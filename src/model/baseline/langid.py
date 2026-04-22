# langid.py
# -------------------------------------------------------------------------------------
# word-level language identification for borrowing detection (with facebookai/FastText)
# -------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import re
import json
import fasttext
from huggingface_hub import hf_hub_download
from typing import List, Dict, Any

HF_LANG_MAP = {
    "ast": "ast_Latn",
    "eu":  "eus_Latn",
    "el":  "ell_Grek"
}

class BorrowingLangId:
    def __init__(self, target_langs: list = None):
        """Language identification at the word level for borrowings using FastText."""
        self._load_model()

    def get_borrowings(self, test_data: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
        """Extracts borrowings with regards to language identification at the word level."""
        results = []
        
        hf_target_lang = HF_LANG_MAP.get(target_lang, target_lang)

        for case in test_data:
            text = case["text"]
            predictions = []
            
            for match in re.finditer(r'\b[a-zA-ZáéíóúüñΑ-Ωα-ωάέίόύήώϊϋ]+(?:-[a-zA-ZáéíóúüñΑ-Ωα-ωάέίόύήώϊϋ]+)*\b', text):
                word = match.group()
                
                if word.isnumeric() or len(word) < 2:
                    continue

                # predict language for the **isolated** word
                labels, _ = self.model.predict(word, k=1)
                lang_pred = labels.replace('__label__', '')                
                
                if lang_pred != hf_target_lang:
                    predictions.append({
                        "span": word,
                        "label": "Raw" # ** cannot classify adaptation *
                    })
            
            results.append({
                "id": case.get("id"),
                "lang": target_lang,
                "prediction": json.dumps(predictions, ensure_ascii=False)
            })
            
        return results

    # --- response generation -------------------------------------------------------------------------

    def _load_model(self):
        """Downloads and loads the FastText model (HF)."""
        print(f"> Loading Hugging Face FastText language identification model...")
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        fasttext.FastText.eprint = lambda x: None
        self.model = fasttext.load_model(model_path)