# encoder.py
# ----------------------------------------------------------------
# multilingual encoder wrapper for 2-step lexical borrowing pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jun-2026

import json
import os
from typing import List, Dict, Any
import torch
from torch import nn
from transformers import (
    AutoModelForTokenClassification, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    pipeline
)
from .dataset import (
    IdDataset, 
    ClfDataset, 
    TAG_TO_ID_MULTI, 
    TAG_TO_ID_BINARY,
    conloan_to_jsonl
)
from .prompt import load_gold_data
from .hf import push_models

# ** extension: weighted cross-entropy Loss **
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Dynamic loss penalization that adapts to token vs Sequence classification."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        num_labels = model.config.num_labels
        
        if logits.dim() == 3:
            # token clf (binary): upweight the Borrowing class (1)
            class_weights = torch.tensor([1.0, 50.0], device=labels.device)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            
            # flatten to 2D for cross entropy, ignoring -100 masking
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            
        else:
            # sequence clf (multi): all tokens are borrowings
            # uniform weights
            class_weights = torch.ones(num_labels, device=labels.device)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


class BorrowingEncoder:
    """BERT wrapper class for training and inference (XLM-R, mBERT, etc.):
    Models: 
    - https://huggingface.co/FacebookAI/xlm-roberta-base
    - https://huggingface.co/jhu-clsp/mmBERT-base
    - https://huggingface.co/blog/mmbert
    """
    
    def __init__(self, model_id: str, langs: List[str], gt: str, run_name: str = "standard", task: str = "multi"):
        self.task = task
        self.run_name = run_name
        self.model_type = "mmbert" if "mmbert" in model_id.lower() else "xlmr"
        
        if self.model_type == "mmbert":
            self.model_id = "jhu-clsp/mmBERT-base"
        else:
            self.model_id = "FacebookAI/xlm-roberta-base"
            
        # Dynamically matches Hugging Face sync paths
        self.output_dir = f"results/post_review/model/{self.model_type}/{self.run_name}_{self.task}"

        splits = load_gold_data(gt, target_langs=langs)
        self.data_splits = {}
        for item in splits:
            lang = item["lang"]
            if langs and lang not in langs:
                continue

            if lang not in self.data_splits:
                self.data_splits[lang] = []
            self.data_splits[lang].append(item)    
        
    def train(self, train_json: str, mask_prob: float = 0.8, task: str = "multi"):
        print(f">>> Initializing {self.model_id} for {task.upper()} task...")
        
        if task == "binary":
            tag_dict = TAG_TO_ID_BINARY
            id_to_tag = {v: k for k, v in tag_dict.items()}
            
            # Prevent silent overwrites and safely handle missing silver data
            if not os.path.exists(train_json):
                if self.run_name == "conloan":
                    print(f"\t> ConLoan data not found at {train_json}. Converting raw ConLoan files...")
                    try:
                        conloan_to_jsonl(output_path=train_json)
                    except TypeError:
                        conloan_to_jsonl()
                else:
                    raise FileNotFoundError(
                        f"(!) Training data not found at {train_json}. "
                        "Please run the data mining and cleaning pipeline first to generate the silver standard."
                    )
                    
            train_dataset = IdDataset(train_json, tokenizer_name=self.model_id, mask_prob=mask_prob)
            
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_id, num_labels=len(tag_dict), id2label=id_to_tag, label2id=tag_dict
            )
        else:
            tag_dict = TAG_TO_ID_MULTI
            id_to_tag = {v: k for k, v in tag_dict.items()}
            train_dataset = ClfDataset(train_json, tokenizer_name=self.model_id)
            
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, num_labels=len(tag_dict), id2label=id_to_tag, label2id=tag_dict
            )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
            use_cpu=False,                
            dataloader_pin_memory=False
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        print(f">>> Training {task} model on silver data...")
        trainer.train()
        
        print(f">>> Saving final model to {self.output_dir}...")
        trainer.save_model(self.output_dir)
        train_dataset.tokenizer.save_pretrained(self.output_dir)

        # --- AUTO-PUSH TO HUGGING FACE HUB ---
        try:
            print(f">>> Automatically pushing {task} model to Hugging Face Hub...")
            push_models(specific_path=self.output_dir.replace("\\", "/"))
        except Exception as e:
            print(f"> (!) Auto-push failed: {e}")

def get_borrowings_2step(self, test_data: List[Dict[str, Any]], language: str, path_binary: str, path_multi: str, fallback: str = "Raw") -> List[Dict[str, Any]]:
        """Extracts borrowings and classifies them using sequence cross-encoding context."""
        
        # (1) binary span classifier for identification
        identifier = pipeline("token-classification", model=path_binary, tokenizer=path_binary, aggregation_strategy="simple", device=0)
        # (2) multi-class sequence classifier for contextual classification
        classifier = pipeline("text-classification", model=path_multi, tokenizer=path_multi, device=0)

        results = []
        for case in test_data:
            text = case["text"]
            
            # ** 1. id **
            id_preds = identifier(text)
            candidate_spans = [p for p in id_preds if p["entity_group"] == "Borrowing"]
            
            formatted_preds = []
            
            # ** 2. clf **
            for cand in candidate_spans:
                # >>> take the span directly from the sentence and get its whitespace out
                raw = text[cand["start"]:cand["end"]]
                span_text = raw.strip()
                
                # span + context
                try:
                    clf_pred = classifier({"text": span_text, "text_pair": text})
                    # >>> revert list indexing (clf_pred is a dict, not a list of dicts)
                    assigned_label = clf_pred["label"] 
                except Exception as e:
                    print(f"(!) Classification failed for span '{span_text}': {e}")
                    assigned_label = fallback
                    
                formatted_preds.append({
                    "span": span_text,
                    # adjust offsets
                    "start": cand["start"] + (len(raw) - len(raw.lstrip())),
                    "end": cand["end"] - (len(raw) - len(raw.rstrip())),
                    "label": assigned_label
                })
            
            results.append({
                "id": case.get("id"),
                "lang": language,
                "prediction": json.dumps(formatted_preds, ensure_ascii=False)
            })
            
        return results