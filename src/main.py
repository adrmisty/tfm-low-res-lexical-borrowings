# main.py
# ----------------------------------------------------------------
# lexical borrowing identification pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import os
import argparse
import logging
from src.model.detection.baseline import run_fewzeroshot_baseline

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    gold_standard_path = os.path.join(project_root, "data", "annotation", "final", "test_gold_annotations.json")
    
    parser = argparse.ArgumentParser(description="Run lexical borrowing baselines")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct", 
        help="HuggingFace model ID for the zero/few-shot baseline"
    )
    parser.add_argument(
        "--langs", 
        nargs="+", 
        default=["ast", "eu", "el"], 
        help="List of language codes to process"
    )
    args = parser.parse_args()

    if not os.path.exists(gold_standard_path):
        logging.error(f"> (!) Gold standard file not found at: {gold_standard_path}")
        return
    run_fewzeroshot_baseline(
        langs=args.langs,
        model_id=args.model,
        gt=gold_standard_path
    )

if __name__ == "__main__":
    main()