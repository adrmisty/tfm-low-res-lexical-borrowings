# llm.py
# ----------------------------------------------------------------------------------------
# LLM wrapper for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompt import get_system_prompt, get_fewshot_prompt, load_gold

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

class BorrowingLLM:
    """LLM wrapper class for lexical borrowing identification."""
    def __init__(self, model_id: str, gt: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.data_splits = load_gold(gt, verbose=True)

        self._load_model()

    def get_borrowings(self, test_data: List[Dict[str, Any]], language: str, examples: Optional[List] = None):
        """Extracts borrowings from test and classifies them based on a built prompt."""
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