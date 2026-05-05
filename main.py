# main.py
# ----------------------------------------------------------------
# lexical borrowing identification + classification pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import os
import argparse
import logging
from src.model.baseline.baseline import run_llm_baseline, run_langid_baseline, run_encoder_baseline
from src.model.baseline.eval import evaluate_pipeline
import src.model.baseline.hf as hf

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

GOLD_STD_PATH = "data/annotation/test_gold_annotations.json"
SILVER_STD_PATH = "data/corpus/processed/mined_sentences.clean.jsonl"

def main():
    parser = argparse.ArgumentParser(description="[TFM] Lexical borrowing detection pipeline")
    parser.add_argument("--action", type=str, choices=["run", "eval", "push"], default="run", help="Choose to run a model, evaluate predictions, or push models")
    parser.add_argument("--type", type=str, choices=["llm", "langid", "xlmr", "mmbert"], default="llm")
    
    # ** extend experiments: run 1step/2step, k-shots, different languages **
    parser.add_argument("--pipeline", type=str, choices=["1step", "2step"], default="2step", help="Architecture to run")
    parser.add_argument("--k", type=int, default=0, help="Number of few-shot examples to inject per language")
    parser.add_argument("--langs", nargs="+", default=["ast", "eu", "el"], help="List of languages to process")
    
    parser.add_argument("--pred_file", type=str, help="Path to prediction JSON (required if --action=eval)")
    parser.add_argument("--title", type=str, default="EXPERIMENT", help="Title for the evaluation plots")
    args = parser.parse_args()

    if not os.path.exists(GOLD_STD_PATH):
        logging.error(f"(!) Gold standard file not found at: {GOLD_STD_PATH}")
        return

    # ** push models to huggingface **
    if args.action == "push":
        logging.info(">> Pushing trained models to Hugging Face Hub")
        hf.push_models() 
        return

    # ** evaluation: joint + split steps & language, confusion matrices, metrics **
    if args.action == "eval":
        if not args.pred_file or not os.path.exists(args.pred_file):
            logging.error("\t> (!) Warning: provide a valid path to a JSON file using --pred_file")
            return
            
        out_dir = os.path.dirname(args.pred_file)
        
        # joint language evaluation
        if len(args.langs) > 1:
            logging.info(f">> Evaluation for: {args.langs}")
            evaluate_pipeline(
                pred_path=args.pred_file, 
                gold_path=GOLD_STD_PATH, 
                out_dir=out_dir, 
                experiment=f"{args.title}_JOINT",
                target_langs=args.langs
            )
        
        # single language evaluation
        for lang in args.langs:
            logging.info(f">> Evaluation for: {lang.upper()}")
            evaluate_pipeline(
                pred_path=args.pred_file, 
                gold_path=GOLD_STD_PATH, 
                out_dir=out_dir, 
                experiment=f"{args.title}_{lang.upper()}",
                target_langs=[lang]
            )
        return

    # *** baseline runs: LLM prompting, language identification, XLM-RoBERTa ***
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
        
    elif args.type == "xlmr":
        if not os.path.exists(SILVER_STD_PATH):
            logging.error(f"\t> (!) Silver data file for multilingual encoder not found at: {SILVER_STD_PATH}")
            return
        run_encoder_baseline(
            model=args.type,
            langs=args.langs, 
            silver_data=SILVER_STD_PATH, 
            gt=GOLD_STD_PATH
        )

    elif args.type == "mmbert":
        if not os.path.exists(SILVER_STD_PATH):
            logging.error(f"\t> (!) Silver data file for multilingual encoder not found at: {SILVER_STD_PATH}")
            return
        run_encoder_baseline(
            model=args.type,
            langs=args.langs, 
            silver_data=SILVER_STD_PATH, 
            gt=GOLD_STD_PATH
        )


if __name__ == "__main__":
    main()