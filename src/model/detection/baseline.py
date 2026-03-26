# baseline.py
# ----------------------------------------------------------------
# baseline runs for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

from .llm import BorrowingLLM
from .langid import BorrowingLangId
from .xlmr import BorrowingXLM
from .eval import get_metrics

import os
from datetime import datetime
import json

OUT_DIR = "data/model"

def run_llm_baseline(langs: list[str], gt: str, model_id="Qwen/Qwen2.5-7B-Instruct"):
    """Few-shot/Zero-shot prompting on LLM for (step 1) LEXICAL BORROWING IDENTIFICATION."""
    llm = BorrowingLLM(model_id, gt)

    all_predictions = []
    all_ground_truth = []

    for lang in langs:
        print(f"\n>>> Running [zero-shot/few-shot LEXICAL BORROWING IDENTIFICATION] baseline for [{lang.upper()}]...")

        few_shot_items, test_items = llm.data_splits[lang]
        shots = [
            {"text": ex["text"], "output": ex["gold_output"]} 
            for ex in few_shot_items
        ]
        all_ground_truth.extend(
            {"id": item["id"], "annotations": item["raw_annotations"]}
            for item in test_items
        )

        predictions = llm.get_borrowings(
            test_data=test_items, 
            language=lang, 
            examples=shots
        )
        all_predictions.extend(predictions)

    print("\n" + "="*50)
    print("ZERO-SHOT/FEW-SHOT EVALUATION RESULTS")
    print("="*50)
    
    out_dir = f"{OUT_DIR}/{model_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    clean_model_name = model_id.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pred_path = os.path.join(out_dir, f"predictions_{clean_model_name}_{timestamp}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
    print(f">>> Predictions saved to: {pred_path}")
    
    eval_path = os.path.join(out_dir, f"eval_metrics_{clean_model_name}_{timestamp}.txt")
    get_metrics(all_predictions, all_ground_truth, out_file=eval_path)

def run_langid_baseline(langs: list[str], gt: str):
    print(">>> Initializing [world-level LANGUAGE IDENTIFICATION] baseline...")
    langid_model = BorrowingLangId(gt)
    
    all_predictions = []
    all_ground_truth = []

    for lang in langs:
        print(f"\n>>> Running [LangID LEXICAL BORROWING IDENTIFICATION] for [{lang.upper()}]...")
        
        _, test_items = langid_model.data_splits[lang]
        
        all_ground_truth.extend(
            {"id": item["id"], "annotations": item["raw_annotations"]}
            for item in test_items
        )

        predictions = langid_model.get_borrowings(test_items, lang)
        all_predictions.extend(predictions)

    print("\n" + "="*50)
    print("LANGID EVALUATION RESULTS")
    print("="*50)
    
    out_dir = f"{OUT_DIR}/FastText"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pred_path = os.path.join(out_dir, f"predictions_langid_{timestamp}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
    eval_path = os.path.join(out_dir, f"eval_metrics_fasttext_{timestamp}.txt")
    get_metrics(all_predictions, all_ground_truth, out_file=eval_path)

def run_xlmr_baseline(langs: list[str], silver_data: str, gt: str = "data/annotation/final/test_gold_annotations.json"):
    """Trains and evaluates XLM-RoBERTa on Silver Data for LEXICAL BORROWING IDENTIFICATION."""
    
    # train on mined sentence corpus
    xlm = BorrowingXLM(gt)
    xlm.train(train_json=silver_data)
    
    all_predictions = []
    all_ground_truth = []

    for lang in langs:
        print(f"\n>>> Running [XLM-RoBERTa LEXICAL BORROWING IDENTIFICATION] inference for [{lang.upper()}]...")
        
        _, test_items = xlm.data_splits[lang]
        
        all_ground_truth.extend(
            {"id": item["id"], "annotations": item["raw_annotations"]}
            for item in test_items
        )

        predictions = xlm.get_borrowings(test_data=test_items, language=lang)
        all_predictions.extend(predictions)

    print("\n" + "="*50)
    print("XLM-ROBERTA (SILVER DATA-trained) EVALUATION RESULTS")
    print("="*50)
    
    out_dir = f"{OUT_DIR}/XLM-RoBERTa"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pred_path = os.path.join(out_dir, f"predictions_xlmr_{timestamp}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
    eval_path = os.path.join(out_dir, f"eval_metrics_xlmr_{timestamp}.txt")
    get_metrics(all_predictions, all_ground_truth, out_file=eval_path)