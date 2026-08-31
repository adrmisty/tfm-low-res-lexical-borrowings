# main.py
# ----------------------------------------------------------------
# loanword data generation and mining pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# aug-2026

import argparse
import logging
import os

from .pipeline import (
    run_scraping, 
    run_generation, 
    run_mining, 
    run_cleaning, 
    run_truecase,
    run_analysis,
    run_granular_analysis
)

GOLD_STD_PATH = "data/annotation/test_gold_annotations.json"

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="[TFM] Lexical Borrowing Data Pipeline")
    
    parser.add_argument("--action", type=str, choices=["run", "eval", "push", "analyze", "scrape", "generate", "mine", "clean", "stats"], default="run", help="Choose action to perform")
    parser.add_argument("--tokenizer", type=str, default="jhu-clsp/mmBERT-base", help="HuggingFace tokenizer ID for fragmentation analysis")    
    parser.add_argument("--langs", nargs="+", default=["ast", "eu", "el"], help="List of languages to process")
    
    parser.add_argument("--corpus", type=str, default="data/corpus/raw/", help="Path to raw corpus directory")
    parser.add_argument("--input", type=str, help="Input directory or file for cleaning step")
    parser.add_argument("--output", type=str, help="Output directory or file")
    parser.add_argument("--pred_file", type=str, help="Path to prediction JSON (required if --action=analyze)")

    args = parser.parse_args()

    if args.action == "scrape":
        run_scraping(args.langs)
        
    elif args.action == "generate":
        run_generation(args.langs)
        
    elif args.action == "mine":
        if not args.output:
            args.output = "data/corpus/mined/"
        run_mining(args.langs, args.corpus, args.output)
        
    elif args.action == "clean":
        # python -m src.data.main --action clean --input data/corpus/mined/mined_sentences.jsonl --output data/corpus/processed/mined_sentences.clean_2.jsonl
        if not args.input or not args.output:
            logging.error("(!) The 'clean' action requires both --input and --output arguments")
            return
        run_cleaning(args.input, args.output, gold=GOLD_STD_PATH)
        #run_truecase(GOLD_STD_PATH)  
                
    elif args.action == "stats":
        # python src/data/main.py --action stats [for plots and statistics]
        run_analysis()

    elif args.action == "analyze":
        out_dir = os.path.dirname(args.pred_file)
        # python -m src.data.main --action analyze --pred_file results\model\mmBert\predictions_mmbert_2step_20260429_162250.json --langs ast eu el 
        run_granular_analysis(
            gold_path=GOLD_STD_PATH,
            pred_path=args.pred_file,
            tokenizer_id=args.tokenizer,
            target_langs=args.langs,
            output_dir=out_dir
        )
        
if __name__ == "__main__":
    main()