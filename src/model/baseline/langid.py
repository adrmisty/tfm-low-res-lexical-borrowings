# langid.py
# ----------------------------------------------------------------
# word-level language identification modeling for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import os
import re
import json
import urllib.request
import fasttext
from typing import List, Dict, Any

from .prompt import load_gold

class BorrowingLangId:
    def __init__(self, gt: str, model_path: str = "lid.176.bin"):
        """Language identification at the word level for borrowings using FastText,
        (https://fasttext.cc/docs/en/language-identification.html), which supports:
        Asturian, Greek and Euskera."""
        self._load_model(model_path)
        self.data_splits = load_gold(gt, verbose=False)

    def get_borrowings(self, test_data: List[Dict[str, Any]], target_lang: str):
        """Extracts borrowings with regards to language identification at the word level."""
        results = []
        for case in test_data:
            text = case["text"]
            predictions = []
            
            for match in re.finditer(r'\b[a-zA-ZáéíóúüñΑ-Ωα-ωάέίόύήώϊϋ]+(?:-[a-zA-ZáéíóúüñΑ-Ωα-ωάέίόύήώϊϋ]+)*\b', text):
                word = match.group()
                
                if word.isnumeric() or len(word) < 2:
                    continue

                # predict language
                labels, _ = self.model.predict(word, k=1)
                lang_pred = labels[0].split('__label__')[-1]                
                if lang_pred != target_lang:
                    predictions.append({
                        "span": word,
                        "label": "Raw" # no adaptation type for langid
                    })
            
            results.append({
                "id": case.get("id"),
                "lang": target_lang,
                "prediction": json.dumps(predictions, ensure_ascii=False)
            })
            
        return results

    # --- response generation -------------------------------------------------------------------------

    def _load_model(self, model_path: str = "lid.176.bin"):

        FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        if not os.path.exists(model_path):
            print(f"> Loading FastText lang-id. language model to {model_path} (126MB)...")
            urllib.request.urlretrieve(FASTTEXT_URL, model_path)

        fasttext.FastText.eprint = lambda x: None
        self.model = fasttext.load_model(model_path)
