# baseline.py
# ----------------------------------------------------------------
# baselines for lexical borrowing detection and classification
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jun-2026

from .llm import BorrowingLLM
from .langid import BorrowingLangId
from .encoder import BorrowingEncoder

import os
from datetime import datetime
import json

OUT_DIR = "results/model"

def run_llm_baseline(langs: list[str], gt: str, model_id="Qwen/Qwen3.5-9B", pipeline: str = "1step", k: int = 2):
    """Few-shot prompting on LLM for lexical borrowing identification and classification."""
    llm = BorrowingLLM(model_id=model_id, langs=langs, gt=gt, k=k)
    #deprec.llm = BorrowingVLLM(model_id, langs=langs, gt=gt)

    all_predictions = []

    for lang in langs:
        print(f"\n>>> Running [{k}-shot LEXICAL BORROWING IDENTIFICATION and CLASSIFICATION ({pipeline} inference) / ({model_id})] baseline for [{lang.upper()}]...")

        test_items = llm.data_splits[lang]
        
        if pipeline == "2step":
            predictions = llm.get_borrowings_2step(
                test_data=test_items, 
                language=lang, 
                k=k
            )
        elif pipeline == "1step":
            predictions = llm.get_borrowings_1step(
                test_data=test_items, 
                language=lang, 
                k=k
            )
        else:
            raise ValueError(f"\t> (!) Invalid pipeline argument: {pipeline}: '1step' or '2step'.")
            
        all_predictions.extend(predictions)
        
    out_dir = f"{OUT_DIR}/{model_id}/{pipeline}"
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
    langid_model = BorrowingLangId(langs, gt)
    
    all_predictions = []

    for lang in langs:
        print(f"\n>>> Running [LangID LEXICAL BORROWING IDENTIFICATION] for [{lang.upper()}]...")
        test_items = langid_model.data_splits[lang]
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

def run_encoder_baseline(model: str, langs: list[str], binary_train_data: str, multi_train_data, gt: str):
    """Trains and runs the 2-step {model} pipeline using dynamic dataset loading."""

    path_binary = f"{OUT_DIR}/{model}/{model}_binary"
    path_multi = f"{OUT_DIR}/{model}/{model}_multi"

    encoder = BorrowingEncoder(model_id=model, langs=langs, gt=gt)

    encoder.output_dir = path_binary
    if not os.path.exists(path_binary):
        print("\t>>> [Encoder-1]: Training binary classifier (Native vs. Borrowing)...")
        encoder.train(train_json=binary_train_data, task="binary")
    else:
        print(f">>> [Encoder-1]: Found existing binary model at {path_binary}; skipping training!")    

    encoder.output_dir = path_multi
    if not os.path.exists(path_multi):
        print("\t>>> [Encoder-2]: Training multi-class classifier (5-tagset)...")
        encoder.train(train_json=multi_train_data, task="multi")
    else:
        print(f">>> [Encoder-2]: Found existing multi-class model at {path_multi}; skipping training!")    
    
    # 2-step inference
    all_predictions = []
    for lang in langs:
        print(f"\n>>> Running [Encoder 2-step inference] for [{lang.upper()}]...")
        test_items = encoder.data_splits[lang]
        
        predictions = encoder.get_borrowings_2step(
            test_data=test_items, 
            language=lang,
            path_binary=path_binary,
            path_multi=path_multi
        )
        all_predictions.extend(predictions)
    
    os.makedirs(f"{OUT_DIR}/encoder", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(f"{OUT_DIR}/encoder", f"predictions_encoder_2step_{timestamp}.json")
    
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(f">>> INFERENCE COMPLETE. Predictions saved to: {pred_path}")
    print(f">>> To evaluate, run: python main.py --action eval --pred_file {pred_path} --title ENCODER-2STEP")
    print("="*50)