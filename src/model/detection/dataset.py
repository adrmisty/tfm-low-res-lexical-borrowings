# dataset.py
# ----------------------------------------------------------------
# masked loss borrowing dataset (silver std.) for XLM-RoBERTa
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import json
import torch
import re
import random
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizerFast


# ** identification of LABEL STUDIO tags **
TAG_TO_ID = {
    "Native": 0, # outside: NATIVE words
    
    # labels: BORROWED words
    "Internationalism": 1,
    "Raw": 2,
    "Adapted_Orthogra": 3,
    "Adapted_Morph": 4,
    "Adapted_Translit": 5,
    "LightVerb_Unintegrated": 6,
    "LightVerb_Integrated": 7
}

# ** mined synth. corpus tags **
TYPE_TO_TAG = {
    # --- Unintegrated / raw ---
    "noun_raw": "Raw",
    "noun_plural_english": "Raw", 
    "cs_latin_raw": "Raw",

    # --- Light verbs ---
    "verb_light_construction": "LightVerb_Unintegrated",
    "verb_light_latin": "LightVerb_Unintegrated",
    "verb_light_greek": "LightVerb_Integrated",

    #TODO: Adapted_Orthogra???
    # --- orthography / morphology ---
    "noun_transliterated": "Adapted_Translit",
    "noun_plural_native": "Adapted_Morph",
    "noun_integrated_sg": "Adapted_Morph",
    "noun_integrated_pl": "Adapted_Morph",
    "verb_morph_prescriptive": "Adapted_Morph",
    "verb_morph_descriptive": "Adapted_Morph",
    "verb_participle_prescriptive": "Adapted_Morph",
    "verb_participle_descriptive": "Adapted_Morph",
    "verb_morph_integrated": "Adapted_Morph",
    "verb_habitual": "Adapted_Morph",
    "verb_morph_aro": "Adapted_Morph",
    "verb_participle": "Adapted_Morph"
}

class BorrowingDataset(Dataset):
    """Silver standard lexical borrowing dataset with partial masked loss
    (a percentage of the unannotated words are O (native), 
    while masking the rest to account for the hidden loanwords in the mined contexts
    that are not annotated in the silver std. unlike in the gold standard test sample."""

    def __init__(self, json_path: str, tokenizer_name: str = "xlm-roberta-base", mask_prob: float = 0.6):
        """Inits the silver standard dataset for XLM-RoBERTa"""
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.mask_prob = mask_prob
        self.data = self._load_silver_data(json_path)

    def _load_silver_data(self, path):
        """Loads data from mined sentence corpus and maps tags to final tagset."""
        valid_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                heuristic_type = item.get("type")
                if heuristic_type in TYPE_TO_TAG:
                    item["gold_tag"] = TYPE_TO_TAG[heuristic_type]
                    valid_data.append(item)
                else:
                    pass 
        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["sentence"]
        seed = item["term"]
        seed_label = item["gold_tag"]

        encoding = self.tokenizer(
            text, 
            truncation=True, 
            max_length=256, 
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = encoding.pop("offset_mapping")[0] 
        labels = []

        match = re.search(re.escape(seed), text)
        if match:
            seed_start_char, seed_end_char = match.span()
        else:
            seed_start_char, seed_end_char = -1, -1

        for i, offset in enumerate(offsets):
            start, end = offset.tolist()

            if start == end:
                labels.append(-100)
                continue

            if seed_start_char != -1 and not (end <= seed_start_char or start >= seed_end_char):
                labels.append(TAG_TO_ID[seed_label])
            else:
                if random.random() < self.mask_prob:
                    labels.append(-100)
                else:
                    labels.append(TAG_TO_ID["Native"])

        encoding["labels"] = torch.tensor(labels)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["labels"].squeeze()
        }