# main.py
# ----------------------------------------------------------------
# lexical borrowing identification pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import os
import argparse
import logging
from src.model.detection.baseline import run_llm_baseline, run_langid_baseline

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def main():
    
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

    """
    run_llm_baseline(
        langs=args.langs,
        model_id=args.model,
        gt="data/annotation/test_gold_annotations.json"
    )
    """
    run_langid_baseline(langs=args.langs, gt="data/annotation/test_gold_annotations.json")

if __name__ == "__main__":
    main()