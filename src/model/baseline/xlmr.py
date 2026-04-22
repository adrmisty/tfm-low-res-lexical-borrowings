# xlmr.py
# ----------------------------------------------------------------
# XLM-RoBERTa wrapper for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import json
from typing import List, Dict, Any
import torch
from torch import nn
from transformers import (
    XLMRobertaForTokenClassification, 
    TrainingArguments, 
    Trainer,
    pipeline
)
from .dataset import BorrowingDataset, TAG_TO_ID_MULTI, TAG_TO_ID_BINARY
from .prompt import load_gold

class BorrowingXLM:
    """XLM-RoBERTa wrapper class for training and inference."""
    
    def __init__(self, gt: str, model_id: str = "xlm-roberta-base", output_dir: str = "data/model/XLM-RoBERTa/xlmr"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.data_splits = load_gold(gt, verbose=False)

    # first exp. (masked_loss --> 0.6)
    # second exp. (ce_loss --> 0.8)
    def train(self, train_json: str, mask_prob: float = 0.8, task: str = "multi"):
        print(f">>> Initializing XLM-RoBERTa for {task} task...")
        
        tag_dict = TAG_TO_ID_BINARY if task == "binary" else TAG_TO_ID_MULTI
        id_to_tag = {v: k for k, v in tag_dict.items()}        
        train_dataset = BorrowingDataset(train_json, tokenizer_name=self.model_id, mask_prob=mask_prob, task=task)
        model = XLMRobertaForTokenClassification.from_pretrained(
            self.model_id,
            num_labels=len(tag_dict),
            id2label=id_to_tag,
            label2id=tag_dict
        )

        # ** train XLM-RoBERTa on silver standard dataset **
        # disc: only has ONE borrowing detected per sentence
        # so silver data uses partial masked loss
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",
            fp16=True,
            report_to="none"
        )

        """ ** first experiment **
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        """
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        print(">>> Training on silver data...")
        print("\t** extended: added penalization for missing LWs **")
        trainer.train()
        
        print(f">>> Saving final model to {self.output_dir}...")
        trainer.save_model(self.output_dir)
        train_dataset.tokenizer.save_pretrained(self.output_dir)

    def get_borrowings_2step(self, test_data: List[Dict[str, Any]], language: str, path_binary: str, path_multi: str, fallback: str = "Native") -> List[Dict[str, Any]]:
        """Extracts borrowings and classifies them based on 2-step inference process: binary model for detection, multi-class model for classification."""
        
        identifier = pipeline("token-classification", model=path_binary, tokenizer=path_binary, aggregation_strategy="simple", device=0)
        classifier = pipeline("token-classification", model=path_multi, tokenizer=path_multi, aggregation_strategy="simple", device=0)

        results = []
        for case in test_data:
            text = case["text"]
            
            # ** identification **
            id_preds = identifier(text)
            candidate_spans = [p for p in id_preds if p["entity_group"] == "Borrowing"]
            
            # ** classification **
            formatted_preds = []
            if candidate_spans:
                clf_preds = classifier(text)
                
                for cand in candidate_spans:
                    assigned_label = fallback
                    
                    # > overlapping spans
                    for cp in clf_preds:
                        if cp["start"] == cand["start"] and cp["end"] == cand["end"] and cp["entity_group"] != "O":
                            assigned_label = cp["entity_group"]
                            break
                            
                    formatted_preds.append({
                        "span": cand["word"].strip(),
                        "label": assigned_label
                    })
            
            results.append({
                "id": case.get("id"),
                "lang": language,
                "prediction": json.dumps(formatted_preds, ensure_ascii=False)
            })
            
        return results

    @DeprecationWarning
    def get_borrowings_1step(self, test_data: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """Extracts borrowings using a fine-tuned RoBERTa model on a corpus of mined lexical borrowing contexts."""
        ner_pipe = pipeline(
            "token-classification", 
            model=self.output_dir, 
            tokenizer=self.output_dir, 
            aggregation_strategy="simple",
            device=0 # -1 if on CPU
        )

        results = []
        for case in test_data:
            text = case["text"]
            raw_preds = ner_pipe(text)
            
            formatted_preds = []
            for p in raw_preds:
                tag = p["entity_group"]
                if tag != "O":
                    formatted_preds.append({
                        "span": p["word"].strip(),
                        "label": tag
                    })
            
            results.append({
                "id": case.get("id"),
                "lang": language,
                "prediction": json.dumps(formatted_preds, ensure_ascii=False)
            })
            
        return results# ** extension: weighted cross-entropy Loss **

# ** extension: weighted cross-entropy Loss **
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Enforce penalisation of missing loanwords (extended to dynamic weighting based on binary/multi classification)."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        num_labels = model.config.num_labels
        if num_labels == 2:
            # binary 1: [Native, Borrowing]
            class_weights = torch.tensor([1.0, 50.0], device=labels.device)
        else:
            # multi 2: [Native, 7 Borrowing Tags...]
            class_weights = torch.tensor([1.0] + [50.0] * (num_labels - 1), device=labels.device)
        
        # cross-entropy loss, flatten logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        loss = loss_fct(active_logits, active_labels)
        
        return (loss, outputs) if return_outputs else loss