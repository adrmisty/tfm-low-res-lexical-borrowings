# main.py
# ----------------------------------------------------------------
# lexical borrowing identification pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import os
import argparse
import logging
from src.model.baseline.baseline import run_llm_baseline, run_langid_baseline, run_xlmr_baseline
from src.model.baseline.eval import evaluate_pipeline

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

GOLD_STD_PATH = "data/annotation/test_gold_annotations.json"
SILVER_STD_PATH = "data/corpus/processed/mined_sentences.clean.jsonl"

def main():
    parser = argparse.ArgumentParser(description="[TFM] Lexical borrowing detection pipeline")
    parser.add_argument("--action", type=str, choices=["run", "eval"], default="run", help="Choose to run a model or evaluate predictions")
    parser.add_argument("--type", type=str, choices=["llm", "langid", "xlmr"], default="llm")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--langs", nargs="+", default=["ast", "eu", "el"])
    parser.add_argument("--pred_file", type=str, help="Path to prediction JSON (required if --action=eval)")
    parser.add_argument("--title", type=str, default="EXPERIMENT", help="Title for the evaluation plots (e.g., QWEN-FEW-SHOT)")
    args = parser.parse_args()

    if not os.path.exists(GOLD_STD_PATH):
        logging.error(f"(!) Gold standard file not found at: {GOLD_STD_PATH}")
        return

    # --- eval pipeline (2-step) ---
    if args.action == "eval":
        if not args.pred_file or not os.path.exists(args.pred_file):
            logging.error("> (!) Please provide a valid path to a JSON file using --pred_file")
            return
            
        out_dir = os.path.dirname(args.pred_file)
        evaluate_pipeline(
            pred_path=args.pred_file, 
            gold_path=GOLD_STD_PATH, 
            out_dir=out_dir, 
            experiment=args.title
        )
        return

    # --- exec inference pipeline ---
    if args.type == "langid":
        run_langid_baseline(langs=args.langs, gt=GOLD_STD_PATH)
    elif args.type == "llm":
        run_llm_baseline(langs=args.langs, model_id=args.model, gt=GOLD_STD_PATH)
    elif args.type == "xlmr":
        if not os.path.exists(SILVER_STD_PATH):
            logging.error(f"> (!) Silver data file not found at: {SILVER_STD_PATH}")
            return
        run_xlmr_baseline(langs=args.langs, silver_data=SILVER_STD_PATH, gt=GOLD_STD_PATH)

if __name__ == "__main__":
    main()