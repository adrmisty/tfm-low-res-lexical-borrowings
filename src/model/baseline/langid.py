# langid.py
# ----------------------------------------------------------------
# word-level language id baseline  for loanword identification
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import re
import json
import fasttext
from huggingface_hub import hf_hub_download
from typing import List, Dict, Any

from .prompt import load_gold_data

class BorrowingLangId:
    def __init__(self, langs: List[str], gt: str):
        """Language identification at the word level for borrowings using FastText
        (facebook/fasttext-language-identification), which supports:
        Asturian, Greek and Euskera via 3-letter ISO + script codes."""
        self._load_model()
        
        splits = load_gold_data(gt, target_langs=langs)
        self.data_splits = {}
        for item in splits:
            lang = item["lang"]
            if langs and lang not in langs:
                continue

            if lang not in self.data_splits:
                self.data_splits[lang] = []
            self.data_splits[lang].append(item)    
                             
        self.lang_map = {
            "ast": "ast_Latn",
            "eu": "eus_Latn",
            "el": "ell_Grek"
        }

    def get_borrowings(self, test_data: List[Dict[str, Any]], target_lang: str):
        """Extracts borrowings with regards to language identification at the word level."""
        results = []
        
        fasttext_target = self.lang_map.get(target_lang, target_lang)
        
        for case in test_data:
            text = case["text"]
            predictions = []
            
            for match in re.finditer(r'\b[a-zA-ZáéíóúüñΑ-Ωα-ωάέίόύήώϊϋ]+(?:-[a-zA-ZáéíóúüñΑ-Ωα-ωάέίόύήώϊϋ]+)*\b', text):
                word = match.group()
                
                if word.isnumeric() or len(word) < 2:
                    continue

                # predict lang & flag if not the target language
                labels, _ = self.model.predict(word, k=1)
                #print(labels) #debug
                if not labels:
                    continue 
                lang_pred = labels[0].replace('__label__', '')
                
                if lang_pred != fasttext_target:
                    predictions.append({
                        "span": word,
                        "label": "Raw" # ** cannot be used for classification **
                    })
            
            results.append({
                "id": case.get("id"),
                "text": text,
                "lang": target_lang,
                "prediction": json.dumps(predictions, ensure_ascii=False)
            })
            
        return results

    # --- response generation -------------------------------------------------------------------------

    def _load_model(self):
        print("\t> Loading facebook/fasttext-language-identification model from Hugging Face Hub...")
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        
        fasttext.FastText.eprint = lambda x: None
        self.model = fasttext.load_model(model_path)