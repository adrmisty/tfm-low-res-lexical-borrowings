# main.py
# ----------------------------------------------------------------
# lexical borrowing identification + classification pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jun-2026

import os
import argparse
import logging
from src.model.baseline.baseline import run_llm_baseline, run_langid_baseline, run_encoder_baseline
from src.model.baseline.eval import evaluate_pipeline
import src.model.baseline.hf as hf

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

GOLD_STD_PATH = "data/annotation/test_gold_annotations.json"
SILVER_STD_PATH = "data/corpus/processed/mined_sentences.clean.jsonl"
CONLOAN_STD_PATH = "data/corpus/processed/conloan.clean.jsonl"

def main():
    parser = argparse.ArgumentParser(description="[TFM] Lexical borrowing detection pipeline")
    parser.add_argument("--action", type=str, choices=["run", "eval", "push"], default="run", help="Choose to run a model, evaluate predictions, or push models")
    parser.add_argument("--type", type=str, choices=["llm", "langid", "xlmr", "mmbert"], default="llm")
    
    parser.add_argument("--pipeline", type=str, choices=["1step", "2step"], default="2step", help="Architecture to run")
    parser.add_argument("--k", type=int, default=0, help="Number of few-shot examples to inject per language")
    parser.add_argument("--langs", nargs="+", default=["ast", "eu", "el"], help="List of languages to process")
    
    parser.add_argument("--conloan", action="store_true", help="Fine-tune encoders on ConLoan dataset instead of standard silver data")
    
    parser.add_argument("--pred_file", type=str, help="Path to prediction JSON (required if --action=eval)")
    parser.add_argument("--title", type=str, default="EXPERIMENT", help="Title for the evaluation plots")
    args = parser.parse_args()

    if not os.path.exists(GOLD_STD_PATH):
        logging.error(f"(!) Gold standard file not found at: {GOLD_STD_PATH}")
        return

    if args.action == "push":
        logging.info(">> Pushing trained models to Hugging Face Hub")
        hf.push_models() 
        return

    if args.action == "eval":
        if not args.pred_file or not os.path.exists(args.pred_file):
            logging.error("\t> (!) Warning: provide a valid path to a JSON file using --pred_file")
            return
            
        out_dir = os.path.dirname(args.pred_file)
        
        if len(args.langs) > 1:
            logging.info(f">> Evaluation for: {args.langs}")
            evaluate_pipeline(
                pred_path=args.pred_file, 
                gold_path=GOLD_STD_PATH, 
                img_dir=os.path.join(out_dir, "img"), stats_dir=os.path.join(out_dir, "stats"),
                experiment=f"{args.title}_JOINT"
            )
        
        for lang in args.langs:
            logging.info(f">> Evaluation for: {lang.upper()}")
            evaluate_pipeline(
                pred_path=args.pred_file, 
                gold_path=GOLD_STD_PATH, 
                img_dir=os.path.join(out_dir, "img"), stats_dir=os.path.join(out_dir, "stats"),
                experiment=f"{args.title}_{lang.upper()}"
            )
        return
    
    if args.type == "langid":
        run_langid_baseline(langs=args.langs, gt=GOLD_STD_PATH)
            
    elif args.type == "llm":
        run_llm_baseline(
            langs=args.langs, 
            model_id=args.type, 
            gt=GOLD_STD_PATH, 
            pipeline=args.pipeline, 
            k=args.k
        )
        
    elif args.type in ["xlmr", "mmbert"]:
        if args.conloan:
            logging.info(f">> HYBRID TRAINING: Using ConLoan for Identification training, silver data for Classification training")
            
            run_encoder_baseline(
                model=args.type,
                langs=args.langs, 
                binary_train_data=CONLOAN_STD_PATH, # ** token > ConLoan (binary, LW or not) **
                multi_train_data=SILVER_STD_PATH,   # ** sequence > silver (5-tag classif) **
                gt=GOLD_STD_PATH,
                run_name="conloan"
            )
        else:
            logging.info(f">> Using training data from: {SILVER_STD_PATH}")
            run_encoder_baseline(
                model=args.type,
                langs=args.langs, 
                binary_train_data=SILVER_STD_PATH, # ** all silver **
                multi_train_data=SILVER_STD_PATH, 
                gt=GOLD_STD_PATH
            )
if __name__ == "__main__":
    main()