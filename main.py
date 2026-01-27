# main.py
# ----------------------------------------------------------------
# tfm-low-res-lexical-borrowings main pipeline script
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import argparse
import os
import sys
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ALL_ROOTS
from src.domain.generators.asturian import AsturianGenerator
from src.domain.generators.basque import BasqueGenerator
from src.domain.generators.greek import GreekGenerator
from src.domain.scrapers.wiktionary import WiktionaryScraper
from src.mining.miner import WikipediaMiner
from src.mining.cleaner import MiningCleaner
from src.analysis.stats import BorrowingStats
from src.analysis.plot import BorrowingPlots

# --- CONFIGURATION PATHS ---
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
MINED_DIR = os.path.join(DATA_DIR, "mined")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PLOTS_DIR = "plots"

SEEDS_FILE = os.path.join(RAW_DIR, "synthetic_borrowings.csv")
MINED_FILE = os.path.join(MINED_DIR, "mined_sentences.jsonl")
CLEAN_FILE = os.path.join(PROCESSED_DIR, "mined_sentences.clean.jsonl")
STATS_FILE = os.path.join(PLOTS_DIR, "stats")
WIKTIONARY_FILE = os.path.join(RAW_DIR, "wiktionary_borrowings.csv")


for d in [RAW_DIR, MINED_DIR, PROCESSED_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------------------------------------------------------------------

def run_scraping():
    print(f"\n[0] Fetching lexical borrowings from Wiktionary...")
    wiktio = WiktionaryScraper(WIKTIONARY_FILE)
    wiktio.scrape()


def run_generation():
    print(f"\n[1] Generating synthetic seeds...")
    
    generators = [
        AsturianGenerator(),
        BasqueGenerator(),
        GreekGenerator()
    ]
    
    all_seeds = []
    
    # synthetic
    for gen in generators:
        print(f"\tRunning generator for: {gen.lang.upper()}")
        all_seeds.extend(gen.generate_all(ALL_ROOTS))
    
    # wiktionary
    print("\tLoading Wiktionary lexical borrowings...")
    scraper = WiktionaryScraper(WIKTIONARY_FILE)
    wik_seeds = scraper.load_seeds(target_langs=['ast', 'eu', 'el'])
    
    if wik_seeds:
        all_seeds.extend(wik_seeds)
        print(f"\tAdded {len(wik_seeds)} loans from Wiktionary.")
    else:
        print("\tNo Wiktionary seeds loaded (file missing or empty).")
        
    df = pd.DataFrame(all_seeds)
    df_clean = df.drop_duplicates(subset=['term', 'lang', 'type'])
    
    df_clean.to_csv(SEEDS_FILE, index=False)
    
    print(f">>> Generated {len(df_clean)} unique seeds.")
    print(f"\tSaved to: {SEEDS_FILE}")
    print("\tSample:")
    print(df_clean.head(3).to_string(index=False))


def run_mining():
    print(f"\n[2] Mining Wikipedia...")
    
    if not os.path.exists(SEEDS_FILE):
        print(f"(!) > Seeds file not found at {SEEDS_FILE}. Run 'generate' first.")
        return

    df_seeds = pd.read_csv(SEEDS_FILE)
    print(f"\tLoaded {len(df_seeds)} seeds.")
    
    miner = WikipediaMiner(user_agent="LexicalBorrowingsTFM/1.0 (masters_thesis)")
    
    results = miner.search_and_extract(df_seeds, limit_per_seed=5)
    
    with open(MINED_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f">>> Mining complete. Found {len(results)} sentences.")
    print(f"\tSaved to: {MINED_FILE}")


def run_cleaning():
    print(f"\n[3] Cleaning mined data...")
    if not os.path.exists(MINED_FILE):
        print("(!) > Mined file not found. Run 'mine' first.")
        return

    cleaner = MiningCleaner()
    kept, d_eng, d_sem = cleaner.clean_file(MINED_FILE, CLEAN_FILE)
            
    print(f">>> Cleaning complete.")
    print(f"\tKept: {kept}")
    print(f"\tDropped (English): {d_eng}")
    print(f"\tDropped (Semantic/Homonyms): {d_sem}")
    print(f"\tSaved to: {CLEAN_FILE}")


def run_stats():
    print(f"\n[4] Computing data stats...")
    
    if not os.path.exists(SEEDS_FILE):
        print("(!) > Seeds file missing")
        return

    stats = BorrowingStats(SEEDS_FILE, MINED_FILE, CLEAN_FILE)
    stats.report(STATS_FILE)


def run_analysis():
    print(f"\n[5] Generating visualization plots...")
    
    if not os.path.exists(CLEAN_FILE):
        print(f"(!) > Clean file not found at {CLEAN_FILE}. Run 'clean' first.")
        return

    data = []
    with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print("\t(!) > Clean file is empty. No data to plot.")
        return

    df = pd.DataFrame(data)
    viz = BorrowingPlots(df)
    
    print("\tGenerating word-class plot...")
    viz.plot_pos_distribution(os.path.join(PLOTS_DIR, "1_pos_distribution.png"))
    
    print("\tGenerating morphological integration plot...")
    viz.plot_integration_strategies(os.path.join(PLOTS_DIR, "2_integration_strats.png"))
    
    print("\tGenerating visual fidelity plot...")
    viz.plot_spelling_adaptation(os.path.join(PLOTS_DIR, "3_spelling_retained.png"))
    
    print("\tGenerating (synthetic vs. wiktionary) data comparison...")
    viz.plot_data_amounts(os.path.join(PLOTS_DIR, "4_dataset_sizes.png"))
    viz.plot_origin_languages(os.path.join(PLOTS_DIR, "5_origin_langs.png"))
    
    print(f">>> Plots saved to directory: {PLOTS_DIR}/")

# -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lexical Borrowings for Low-Resource Languages (TFM)")
    
    parser.add_argument(
        "step", 
        choices=["scrape", "generate", "mine", "clean", "analyze", "all"],
        help="The pipeline step to execute."
    )
    
    args = parser.parse_args()
    
    if args.step in ["scrape", "all"]:
        run_scraping()

    if args.step in ["generate", "all"]:
        run_generation()
        
    if args.step in ["mine", "all"]:
        run_mining()
        
    if args.step in ["clean", "all"]:
        run_cleaning()
                
    if args.step in ["analyze", "all"]:
        run_stats()
        run_analysis()