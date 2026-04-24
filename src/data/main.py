# main.py
# ----------------------------------------------------------------
# loanword data generation and mining pipeline
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import argparse
import logging

from pipeline import (
    run_scraping, 
    run_generation, 
    run_mining, 
    run_cleaning, 
    run_analysis
)

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="[TFM] Lexical Borrowing Data Pipeline")
    
    parser.add_argument("--action", type=str, choices=["scrape", "generate", "mine", "clean", "stats"], required=True, help="The data pipeline step to execute")
    parser.add_argument("--langs", nargs="+", default=["ast", "eu", "el"], help="List of languages to process")
    
    parser.add_argument("--corpus", type=str, default="data/corpus/raw/", help="Path to raw corpus directory")
    parser.add_argument("--input", type=str, help="Input directory or file for cleaning step")
    parser.add_argument("--output", type=str, help="Output directory or file")
    
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
        if not args.input or not args.output:
            logging.error("(!) The 'clean' action requires both --input and --output arguments")
            return
        run_cleaning(args.input, args.output)
        
    elif args.action == "stats":
        # python src/data/main.py --action stats [for plots and statistics]
        run_analysis()

if __name__ == "__main__":
    main()