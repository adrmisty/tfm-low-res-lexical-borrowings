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
from src.model.detection.eval import get_confusion_matrix

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

GOLD_STD_PATH = "data/annotation/test_gold_annotations.json"
SILVER_STD_PATH = "data/processed/mined_sentences.clean.jsonl"
LLM = "Qwen/Qwen3.5-9B"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="[TFM] Lexical borrowing detection pipeline")
    parser.add_argument("--action", type=str, choices=["run", "plot"], default="run", help="Choose to run a model or plot a confusion matrix")
    parser.add_argument("--type", type=str, choices=["llm", "langid", "xlmr"], default="llm")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--langs", nargs="+", default=["ast", "eu", "el"])
    parser.add_argument("--pred_file", type=str, help="Path to prediction JSON (required if --action=plot)")
    parser.add_argument("--title", type=str, help="Path to prediction JSON (required if --action=plot)")
    args = parser.parse_args()

    if not os.path.exists(GOLD_STD_PATH):
        logging.error(f"(!) Gold standard file not found at: {GOLD_STD_PATH}")
        return

    # --- plot results ---
    if args.action == "plot":
        if not args.pred_file or not os.path.exists(args.pred_file):
            logging.error("> (!) Please provide a valid path to a JSON file using --pred_file! e")
            return
            
        output_png = args.pred_file.replace(".json", f"_{args.title.upper()}_cm.png")
        get_confusion_matrix(pred_path=args.pred_file, gold_path=GOLD_STD_PATH, output_img=output_png, experiment=args.title)
        return

    # --- exec pipeline ---
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