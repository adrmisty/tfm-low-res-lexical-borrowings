# main.py
# ----------------------------------------------------------------
# lexical borrowing identification pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import os
import argparse
import logging
from src.model.detection.baseline import run_llm_baseline, run_langid_baseline, run_xlmr_baseline

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

GOLD_STD_PATH = "data/annotation/test_gold_annotations.json"
SILVER_STD_PATH = "data/processed/mined_sentences.clean.jsonl"

def main():
    
    parser = argparse.ArgumentParser(description="Run lexical borrowing baselines")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct", 
        help="HuggingFace model ID for the zero/few-shot baseline"
    )
    parser.add_argument("--type", type=str, choices=["llm", "langid", "xlmr"], default="llm")
    parser.add_argument(
        "--langs", 
        nargs="+", 
        default=["ast", "eu", "el"], 
        help="List of language codes to process"
    )
    args = parser.parse_args()

    if args.type == "langid":
        run_langid_baseline(langs=args.langs, gt=GOLD_STD_PATH)
    elif args.type == "llm":
        run_llm_baseline(langs=args.langs, model_id=args.model, gt=GOLD_STD_PATH)
    elif args.type == "xlmr":
        run_xlmr_baseline(langs=args.langs, silver_data=SILVER_STD_PATH, gt=GOLD_STD_PATH)

if __name__ == "__main__":
    main()