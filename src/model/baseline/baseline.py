# baseline.py
# ----------------------------------------------------------------
# baseline runs for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

from .llm import BorrowingLLM
from .langid import BorrowingLangId
from .xlmr import BorrowingXLM

import os
from datetime import datetime
import json

OUT_DIR = "data/model"

def run_llm_baseline(langs: list[str], gt: str, model_id="Qwen/Qwen3.5-9B"):
    """Few-shot prompting on LLM for lexical borrowing identification and classification."""
    llm = BorrowingLLM(model_id, gt)

    all_predictions = []

    for lang in langs:
        print(f"\n>>> Running [few-shot LEXICAL BORROWING IDENTIFICATION and CLASSIFICATION (2-step inference) / ({model_id})] baseline for [{lang.upper()}]...")

        few_shot_items, test_items = llm.data_splits[lang]
        shots = [
            {"text": ex["text"], "output": ex["gold_output"]} 
            for ex in few_shot_items
        ]

        predictions = llm.get_borrowings_2step(
            test_data=test_items, 
            language=lang, 
            examples=shots
        )
        all_predictions.extend(predictions)

    out_dir = f"{OUT_DIR}/{model_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    clean_model_name = model_id.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pred_path = os.path.join(out_dir, f"predictions_{clean_model_name}_{timestamp}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(f">>> INFERENCE COMPLETE. Predictions saved to: {pred_path}")
    print(f">>> To evaluate, run: python main.py --action eval --pred_file {pred_path} --title {clean_model_name}")
    print("="*50)

def run_langid_baseline(langs: list[str], gt: str):
    """Language identification at the word level for lexical borrowing identification and classification
    >>> purely for identification, cannot classify."""
    print(">>> Initializing [word-level LANGUAGE IDENTIFICATION] baseline...")
    langid_model = BorrowingLangId(gt)
    
    all_predictions = []

    for lang in langs:
        print(f"\n>>> Running [LangID LEXICAL BORROWING IDENTIFICATION] for [{lang.upper()}]...")
        _, test_items = langid_model.data_splits[lang]
        predictions = langid_model.get_borrowings(test_items, lang)
        all_predictions.extend(predictions)

    out_dir = f"{OUT_DIR}/FastText"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pred_path = os.path.join(out_dir, f"predictions_langid_{timestamp}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(f">>> INFERENCE COMPLETE. Predictions saved to: {pred_path}")
    print(f">>> To evaluate, run: python main.py --action eval --pred_file {pred_path} --title LANGID")
    print("="*50)

def run_xlmr_baseline(langs: list[str], silver_data: str, gt: str = "data/annotation/final/test_gold_annotations.json"):
    """Trains and runs the 2-step XLM-RoBERTa pipeline using dynamic dataset loading."""
    
    path_binary = f"{OUT_DIR}/XLM-RoBERTa/xlmr_binary"
    path_multi = f"{OUT_DIR}/XLM-RoBERTa/xlmr_multi"
    
    xlm = BorrowingXLM(gt)

    print("\t>>> [XLMR-1]: Training binary classifier (Native vs. Borrowing)...")
    xlm.output_dir = path_binary
    xlm.train(train_json=silver_data, task="binary") 
    
    print("\t>>> [XLMR-2]: Training multi-class classifier (LS tagset)...")
    xlm.output_dir = path_multi
    xlm.train(train_json=silver_data, task="multi")
    
    # 2-step inference
    all_predictions = []
    for lang in langs:
        print(f"\n>>> Running [XLM-RoBERTa 2-step inference] for [{lang.upper()}]...")
        _, test_items = xlm.data_splits[lang]
        
        predictions = xlm.get_borrowings_2step(
            test_data=test_items, 
            language=lang,
            path_binary=path_binary,
            path_multi=path_multi
        )
        all_predictions.extend(predictions)
    
    os.makedirs(f"{OUT_DIR}/XLM-RoBERTa", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(f"{OUT_DIR}/XLM-RoBERTa", f"predictions_xlmr_2step_{timestamp}.json")
    
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(f">>> INFERENCE COMPLETE. Predictions saved to: {pred_path}")
    print(f">>> To evaluate, run: python main.py --action eval --pred_file {pred_path} --title XLMR-2STEP")
    print("="*50)