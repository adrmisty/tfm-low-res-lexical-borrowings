# dataset.py
# ------------------------------------------------------------------------
# lexical borrowing datasets (silver std.) for XLM-RoBERTa
# separated by architecture
# 1. token classification
# 2. sequence classification
# ------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import json
import torch
import re
import random
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizerFast
from .prompt import LABELS

# ** 1. binary tags **
TAG_TO_ID_BINARY = {
    "Native": 0,
    "Borrowing": 1
}

# ** 2. multi tags **
TAG_TO_ID_MULTI = {label: idx for idx, label in enumerate(LABELS)}


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

    # --- orthography / morphology ---
    "noun_transliterated": "Adapted_Translit", #TODO: eliminate > Adapted_Orthogra
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

class IdDataset(Dataset):
    """Silver standard dataset for binary token classification (borrowing span identification)."""

    def __init__(self, json_path: str, tokenizer_name: str = "xlm-roberta-base", mask_prob: float = 0.6):
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.mask_prob = mask_prob
        self.data = self._load_silver_data(json_path)

    def _load_silver_data(self, path):
        valid_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                if item.get("type") in TYPE_TO_TAG:
                    valid_data.append(item)
        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["sentence"]
        seed = item["term"]

        encoding = self.tokenizer(
            text, 
            truncation=True, 
            max_length=256, 
            padding="max_length",
            return_offsets_mapping=True
        )

        offsets = encoding.pop("offset_mapping") 
        labels = []

        match = re.search(re.escape(seed), text)
        if match:
            seed_start_char, seed_end_char = match.span()
        else:
            seed_start_char, seed_end_char = -1, -1

        for i, (start, end) in enumerate(offsets):
            if start == end: # special tokens
                labels.append(-100)
                continue

            if seed_start_char != -1 and not (end <= seed_start_char or start >= seed_end_char):
                labels.append(TAG_TO_ID_BINARY["Borrowing"])
            else:
                if random.random() < self.mask_prob:
                    labels.append(-100)
                else:
                    labels.append(TAG_TO_ID_BINARY["Native"])

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels)
        }


class ClfDataset(Dataset):
    """Silver standard dataset for multi-class sequence classification (borrowing morph. classification)."""

    def __init__(self, json_path: str, tokenizer_name: str = "xlm-roberta-base", max_length: int = 256):
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self._load_silver_data(json_path)

    def _load_silver_data(self, path):
        valid_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                heuristic_type = item.get("type")
                
                # only keep items with types that can be mapped to our target taxonomy
                if heuristic_type in TYPE_TO_TAG:
                    gold_label = TYPE_TO_TAG[heuristic_type]
                    if gold_label in TAG_TO_ID_MULTI:
                        valid_data.append({
                            "span": item["term"],
                            "context": item["sentence"],
                            "label": gold_label
                        })
        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # ** tokenizer handles the separator injection for text-pair classification **
        # <s> target_span </s></s> context_sentence </s>
        encoding = self.tokenizer(
            text=item["span"],
            text_pair=item["context"],
            truncation="only_second", # Never truncate the target word, only the context
            max_length=self.max_length,
            padding="max_length"
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(TAG_TO_ID_MULTI[item["label"]], dtype=torch.long)
        }