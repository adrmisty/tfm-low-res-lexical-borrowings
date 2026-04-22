# baseline.py
# ----------------------------------------------------------------
# baseline orchestrator for lexical borrowing models
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import os
import json
from datetime import datetime
import logging

from .prompt import load_gold_data
from .llm import BorrowingLLM
from .langid import BorrowingLangId
from .encoder import BorrowingEncoder

OUT_DIR = "results/model"

def run_llm_baseline(langs: list, model_id: str, gt: str, pipeline: str = "2step", k: int = 0):
    logging.info(f"\n--- Running LLM Baseline ({model_id}) | Pipeline: {pipeline} | k: {k} ---")
    
    test_data = load_gold_data(gt, target_langs=langs)
    if not test_data:
        logging.error("\t> (!) No test data found for the specified languages.")
        return
        
    llm = BorrowingLLM(model_id=model_id)
    all_results = []
    
    # ** inference per-language **
    for lang in langs:
        logging.info(f"\t> Processing [{lang.upper()}]...")
        # filter data for current language loop
        lang_data = [item for item in test_data if item["lang"] == lang]
        
        if pipeline == "2step":
            res = llm.get_borrowings_2step(test_data=lang_data, language=lang, k=k)
        else:
            res = llm.get_borrowings_1step(test_data=lang_data, language=lang, k=k)
            
        all_results.extend(res)
        
    model_short_name = model_id.split("/")[-1]
    out_dir = os.path.join(OUT_DIR, model_short_name, pipeline)
    _save_preds(all_results, out_dir, f"{model_short_name}_{pipeline}_{k}shot")

def run_langid_baseline(langs: list, gt: str):
    logging.info("\n--- Running FastText LangID Baseline ---")
    
    test_data = load_gold_data(gt, target_langs=langs)
    if not test_data:
        logging.error("\t> (!) No test data found for the specified languages.")
        return
        
    langid_model = BorrowingLangId()
    all_results = []
    
    for lang in langs:
        logging.info(f"\t> Processing [{lang.upper()}]...")
        lang_data = [item for item in test_data if item["lang"] == lang]
        res = langid_model.get_borrowings(test_data=lang_data, target_lang=lang)
        all_results.extend(res)
        
    _save_preds(all_results, os.path.join(OUT_DIR, "FastText", "1step"), "langid")

def run_xlmr_baseline(langs: list, silver_data: str, gt: str, pipeline: str = "2step", model_id: str = "xlm-roberta-base"):
    logging.info(f"\n--- Running Encoder Baseline ({model_id}) | Pipeline: {pipeline} ---")
    
    test_data = load_gold_data(gt, target_langs=langs)
    if not test_data:
        logging.error("\t> (!) No test data found for the specified languages.")
        return
        
    # checkpoints
    out_dir_bin = os.path.join(OUT_DIR, model_id, "step1_binary")
    out_dir_mul = os.path.join(OUT_DIR, model_id, "step2_multi")
    
    encoder = BorrowingEncoder(gt=gt, model_id=model_id)
    
    # ** train if they don't exist **
    if not os.path.exists(out_dir_bin):
        encoder.output_dir = out_dir_bin
        encoder.train(train_json=silver_data, task="binary")
        
    if pipeline == "2step" and not os.path.exists(out_dir_mul):
        encoder.output_dir = out_dir_mul
        encoder.train(train_json=silver_data, task="multi")
        
    # ** inference **
    all_results = []
    for lang in langs:
        logging.info(f"\t> Processing [{lang.upper()}]...")
        lang_data = [item for item in test_data if item["lang"] == lang]
        
        if pipeline == "2step":
            res = encoder.get_borrowings_2step(
                test_data=lang_data, 
                language=lang, 
                path_binary=out_dir_bin, 
                path_multi=out_dir_mul
            )
        else:
            logging.error("\t>> (!) 1-step not natively supported in the updated encoder sequence classification pipeline (default is 2-step)")
            return
            
        all_results.extend(res)
        
    model_short_name = model_id.split("/")[-1]
    out_dir = os.path.join(OUT_DIR, model_short_name, pipeline)
    _save_preds(all_results, out_dir, f"{model_short_name}_{pipeline}")
    
# --- util

def _save_preds(results: list, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #timestamped json
    out_file = os.path.join(out_dir, f"predictions_{prefix}_{timestamp}.json")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logging.info(f"\t> Predictions saved successfully to: {out_file}")
