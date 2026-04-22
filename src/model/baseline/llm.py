# llm.py
# ----------------------------------------------------------------------------------------
# LLM wrapper for lexical borrowing 1) identification and 2) joint id + classification
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import re
import json
import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompt import *

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

class BorrowingLLM:
    """LLM wrapper class for lexical borrowing identification."""
    def __init__(self, model_id: str, gt: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.data_splits = load_gold(gt, verbose=True)

        self._load_model()

    def get_borrowings_2step(self, test_data: List[Dict[str, Any]], language: str, examples: Optional[List] = None, fallback: str = "Native"):
        """Extracts borrowings and classifies them based on built chained prompts for each step."""
        results = []
        
        for case in test_data:
            text = case["text"]
            
            # ** identification **
            sys_id = get_system_prompt_id(language)
            prompt_id = get_fewshot_prompt_id(sys_id, text, examples)
            # prefill -> string list
            raw_id_output = self._generate(sys_id, prompt_id, prefill="[\"")   
                     
            # >> identification: parse resulting spans from json
            try:
                clean_out = re.sub(r"<think>.*?</think>", "", str(raw_id_output), flags=re.DOTALL).strip()
                match = re.search(r"\[\s*\".*\"\s*\]|\[.*?\]", clean_out, re.DOTALL)
                candidate_spans = json.loads(match.group()) if match else []
                if not isinstance(candidate_spans, list): candidate_spans = []
            except Exception:
                candidate_spans = []

            # ** classification **
            predictions = []
            sys_clf = get_system_prompt_clf(language)
            
            for span in candidate_spans:
                if not isinstance(span, str): continue
                
                prompt_clf = get_fewshot_prompt_clf(sys_clf, text, span, examples)
                # prefill -> empty, label
                label = self._generate(sys_clf, prompt_clf, prefill="").strip() 

                valid_label = label if label in LABELS else fallback
                predictions.append({"span": span, "label": valid_label})
            
            results.append({
                "id": case.get("id"),
                "lang": language,
                "prompt": prompt_id,
                "prediction": json.dumps(predictions, ensure_ascii=False)
            })
            
        return results

    def get_borrowings_1step(self, test_data: List[Dict[str, Any]], language: str, examples: Optional[List] = None):
        """Extracts borrowings from test and classifies them based on a built prompt (all in 1 step)."""
        system_prompt = get_system_prompt(language)
        results = []
        
        for case in test_data:
            user_prompt = get_fewshot_prompt(system_prompt, case["text"], examples)
            prediction = self._generate(system_prompt, user_prompt)
            results.append({
                "id": case.get("id"),
                "lang": language,
                "prompt": user_prompt,
                "prediction": prediction
            })
            
        return results
    
    # --- response generation -------------------------------------------------------------------------

    def _load_model(self):
        logging.info(f"\t> Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).eval()

    def _generate(self, system: str, user: str, prefill: str = "[\n") -> str:
        """Generates LLM model's response to few-shot prompt."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        out_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True)        
        return prefill + out_text