# llm.py
# ----------------------------------------------------------------------------------------
# LLM/vLLM wrapper for lexical borrowing 1) identification and 2) joint id + classification
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import re
import torch
import json
import logging
from typing import List, Dict, Any
#from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompt import *

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

class BorrowingLLM:
    """LLM wrapper class for lexical borrowing detection and classification."""
    
    def __init__(self, model_id: str, langs: List[str], gt: str, k: int):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if "llm" in model_id:
            self.model_id = "Qwen/Qwen3.5-9B" # HF says compatible with vLLM? :/
        self._load_model()

        self.k = k
        splits = load_gold_data(gt, target_langs=langs)
        self.data_splits = {}
        for item in splits:
            lang = item["lang"]
            if langs and lang not in langs:
                continue

            if lang not in self.data_splits:
                self.data_splits[lang] = []
            self.data_splits[lang].append(item)    

    def get_borrowings_2step(self, test_data: List[Dict[str, Any]], language: str, k: int = 0, fallback: str = "Native"):
        """Extracts borrowings and classifies them based on dynamically built chained prompts."""
        results = []
        
        for case in test_data:
            text = case["text"]
            
            # ** 1. identification **
            sys_id = get_system_prompt_id(language)
            prompt_id = get_fewshot_prompt_id(sys_id, text, language, k)
            
            raw_id_output = self._generate(
                sys_id,
                prompt_id,
                max_new_tokens=128,
                prefill='[\n"'
            )                      
            try:
                clean_out = str(raw_id_output).strip()

                # remove think tags
                clean_out = re.sub(
                    r"</?think>",
                    "",
                    clean_out,
                    flags=re.IGNORECASE
                )

                json_str = self._extract_first_json_list(clean_out)

                if json_str:
                    candidate_spans = json.loads(json_str)
                else:
                    raise ValueError("> (!) Fallback: JSON list extraction failed")

            except Exception:
                # *** truncated list ***
                # json.loads fails
                candidate_spans = re.findall(r'"([^"]+)"', str(raw_id_output))
                
            # validate
            if not isinstance(candidate_spans, list):
                candidate_spans = []

            candidate_spans = [
                s.strip()
                for s in candidate_spans
                if isinstance(s, str) and s.strip()
            ]
            
            
            # ** 2. classification **
            predictions = []
            sys_clf = get_system_prompt_clf(language)
            
            for span in candidate_spans:
                if not isinstance(span, str): continue
                
                prompt_clf = get_fewshot_prompt_clf(sys_clf, text, span, language, k)
                raw_label_output = self._generate(
                    sys_clf,
                    prompt_clf,
                    max_new_tokens=48,
                    prefill='{"label":"'
                ).strip() 
                
                label = fallback
                try:
                    clean_label = str(raw_label_output).strip()
                    clean_label = re.sub(r"</?think>", "", clean_label, flags=re.IGNORECASE)
                    
                    json_str = self._extract_first_json_dict(clean_label)
                    
                    if json_str:
                        parsed_json = json.loads(json_str)
                        label = parsed_json.get("label", fallback)
                    else:
                        raise ValueError("> (!) Fallback: JSON dict extraction failed")

                except Exception:
                    # *** truncated labeling ***
                    match = re.search(r'"label"\s*:\s*"([^"]+)"', '{"label":"' + str(raw_label_output))
                    if match:
                        label = match.group(1)
                
                # > Invalid/Native for evaluation
                valid_label = label if (label in TAGSET or "Invalid" in label) else fallback
                predictions.append({"span": span, "label": valid_label})

            res = {
                "id": case.get("id"),
                "lang": language,
                "prediction": predictions
            }
            print(res)
            results.append(res)
            
        return results
    
    def get_borrowings_1step(self, test_data: List[Dict[str, Any]], language: str, k: int = 0):
        """Extracts borrowings from test and classifies them based on a built prompt."""
        system_prompt = get_system_prompt_1step(language)
        results = []
        
        for case in test_data:
            user_prompt = get_fewshot_prompt_1step(system_prompt, case["text"], language, k)
            prediction = self._generate(system_prompt, user_prompt, max_new_tokens=128, prefill='[{"span":"')
            
            results.append({
                "id": case.get("id"),
                #"prompt": prompt,
                "lang": language,
                "prediction": prediction
            })
            
        return results
        
    # --- response generation -------------------------------------------------------------------------

    def _load_model(self):
        logging.info(f"\t> Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda").eval()
        
    def _generate(self, system: str, user: str, max_new_tokens: int = 64, prefill: str = "") -> str:
        """Generates LLM response."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # --- force structured continuation ---
        if prefill:
            text += prefill

        inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,

                max_new_tokens=max_new_tokens,
                # 1step: 128 (change if truncation happens)
                # 2step: 64(id) + 16(clf)

                # extraction task -> deterministic
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]

        out_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        # remove reasoning remnants
        out_text = re.sub(
            r"(?i)(thinking process|thought process).*",
            "",
            out_text,
            flags=re.DOTALL
        ).strip()

        # avoid duplicate prefill
        if prefill and out_text.startswith(prefill):
            return out_text

        return prefill + out_text    
    

    # ----------------- output parsing util

    def _extract_first_json_list(self, text: str):
        start = text.find("[")

        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            char = text[i]
            if char == "[":
                depth += 1

            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

        return None

    def _extract_first_json_dict(self, text: str):
        start = text.find("{")

        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

        return None

"""
class BorrowingVLLM:
    #TO-BE-DEVELOPED: vLLM wrapper class for lexical borrowing identification with fast batched inference
    
    def __init__(self, model_id: str, langs: List[str], gt: str):
        #TODO: failed to inspect [any architecture I give it]
        if "llm" in model_id:
            self.model_id = "Qwen/Qwen2.5-7B" # Qwen2forCasualLM https://huggingface.co/Qwen/Qwen2.5-7B
        self._load_model()

        splits = load_gold_data(gt, target_langs=langs, few_shot=True)
        self.data_splits = {}
        for item in splits:
            lang = item["lang"]
            if langs and lang not in langs:
                continue

            if lang not in self.data_splits:
                self.data_splits[lang] = []
            self.data_splits[lang].append(item)    

        
    def get_borrowings_2step(self, test_data: List[Dict[str, Any]], language: str, k: int = 0, fallback: str = "Native"):
        # Extracts and classifies borrowings using massive batching for vLLM optimization.
        results = []
        
        # ** 1. identification **
        sys_id = get_system_prompt_id(language)
        id_prompts = [get_fewshot_prompt_id(sys_id, case["text"], language, k) for case in test_data]
        formatted_id_prompts = self._format_prompts(sys_id, id_prompts, prefill="[\n\"")
        id_outputs = self.model.generate(formatted_id_prompts, self.sampling_params)
        
        # _ build clf batch _
        sys_clf = get_system_prompt_clf(language)
        clf_tasks = [] # tuples: (case_index, span, formatted_prompt)
        
        for i, out in enumerate(id_outputs):
            raw_text = "[\"" + out.outputs.text
            
            try:
                clean_out = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                match = re.search(r"[s*\".*\"s*]|[.*?]", clean_out, re.DOTALL)
                candidate_spans = json.loads(match.group()) if match else []
                if not isinstance(candidate_spans, list): candidate_spans = []
            except Exception:
                candidate_spans = []
                
            for span in candidate_spans:
                if not isinstance(span, str): continue
                prompt_clf = get_fewshot_prompt_clf(sys_clf, test_data[i]["text"], span, language, k)
                fmt_clf = self._format_prompts(sys_clf, [prompt_clf], prefill="")
                clf_tasks.append((i, span, fmt_clf))

        # ** 2. batched clf **
        if clf_tasks:
            clf_prompts = [task for task in clf_tasks]
            clf_outputs = self.model.generate(clf_prompts, self.sampling_params)
            
            for j, out in enumerate(clf_outputs):
                predicted_label = out.outputs.text.strip()
                clf_tasks[j] = (*clf_tasks[j], predicted_label)
        
        # ** batched results reconstruction **
        for i, case in enumerate(test_data):
            case_preds = []
            
            for task in clf_tasks:
                if task == i:
                    span = task
                    label = task
                    valid_label = label if (label in TAGSET or "Invalid" in label) else fallback
                    case_preds.append({"span": span, "label": valid_label})
            
            results.append({
                "id": case.get("id"),
                "lang": language,
                "prompt": id_prompts[i],
                "prediction": json.dumps(case_preds, ensure_ascii=False)
            })
            
        return results

    def get_borrowings_1step(self, test_data: List[Dict[str, Any]], language: str, k: int = 0):
        # Batched inference for 1-step pipeline.
        system_prompt = get_system_prompt_1step(language)
        
        user_prompts = [get_fewshot_prompt_1step(system_prompt, case["text"], language, k) for case in test_data]
        formatted_prompts = self._format_prompts(system_prompt, user_prompts, prefill="[\n\n")
        
        outputs = self.model.generate(formatted_prompts, self.sampling_params)
        
        results = []
        for i, case in enumerate(test_data):
            prediction = "[\n" + outputs[i].outputs.text
            results.append({
                "id": case.get("id"),
                "lang": language,
                #"prompt": user_prompts[i],
                "prediction": prediction
            })
            
        return results
    
    # --- backend logic -------------------------------------------------------------------------

    def _load_model(self):
        logging.info(f"\t> Loading {self.model_id} via vLLM...")
        
        # greedy decoding (do_sample=False equivalent)
        self.sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=2048
        )

        # ** vLLM engine **
        self.model = LLM(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=4096, 
            tensor_parallel_size=1 # TODO: change for multiple gpu use
        )
        
        # tokenizer from the vLLM engine
        self.tokenizer = self.model.get_tokenizer()

    def _format_prompts(self, system: str, users: List[str], prefill: str = "") -> List[str]:
        Applies the chat template to a batch of prompts.
        formatted = []
        for user in users:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
            formatted.append(text)
        return formatted
"""