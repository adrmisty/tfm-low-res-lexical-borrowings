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
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ALL_ROOTS
from src.domain.generators.asturian import AsturianGenerator
from src.domain.generators.basque import BasqueGenerator
from src.domain.generators.greek import GreekGenerator
from src.pipeline.miner import WikipediaMiner
from src.pipeline.cleaner import EnglishFilter
from src.analysis.visualizer import LoanwordVisualizer

# --- CONFIGURATION PATHS ---
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
MINED_DIR = os.path.join(DATA_DIR, "mined")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PLOTS_DIR = "plots"

SEEDS_FILE = os.path.join(RAW_DIR, "synthetic_borrowings.csv")
MINED_FILE = os.path.join(MINED_DIR, "mined_sentences.jsonl")
CLEAN_FILE = os.path.join(PROCESSED_DIR, "mined_sentences.clean.jsonl")

for d in [RAW_DIR, MINED_DIR, PROCESSED_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


def run_generation():
    print(f"\n[STEP 1] Generating synthetic seeds...")
    
    generators = [
        AsturianGenerator(),
        BasqueGenerator(),
        GreekGenerator()
    ]
    
    all_seeds = []
    for gen in generators:
        print(f"\tRunning generator for: {gen.lang.upper()}")
        all_seeds.extend(gen.generate_all(ALL_ROOTS))
    
    df = pd.DataFrame(all_seeds)
    df_clean = df.drop_duplicates(subset=['term', 'lang', 'type'])
    
    df_clean.to_csv(SEEDS_FILE, index=False)
    
    print(f"[SUCCESS] Generated {len(df_clean)} unique seeds.")
    print(f"\tSaved to: {SEEDS_FILE}")
    print("\tSample:")
    print(df_clean.head(3).to_string(index=False))


def run_mining():
    print(f"\n[STEP 2] Mining Wikipedia...")
    
    if not os.path.exists(SEEDS_FILE):
        print(f"[ERROR] Seeds file not found at {SEEDS_FILE}. Run 'generate' first.")
        return

    df_seeds = pd.read_csv(SEEDS_FILE)
    print(f"\tLoaded {len(df_seeds)} seeds.")
    
    miner = WikipediaMiner(user_agent="LoanwordThesisBot/1.0 (academic_research)")
    
    results = miner.search_and_extract(df_seeds, limit_per_seed=5)
    
    with open(MINED_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"[SUCCESS] Mining complete. Found {len(results)} sentences.")
    print(f"\tSaved to: {MINED_FILE}")


def run_cleaning():
    print(f"\n[STEP 3] Cleaning mined data (English)...")
    
    if not os.path.exists(MINED_FILE):
        print(f"[ERROR] Mined file not found at {MINED_FILE}. Run 'mine' first.")
        return

    cleaner = EnglishFilter(threshold=0.25)
    
    kept = []
    dropped_count = 0
    total_count = 0
    
    print("\tProcessing sentences...")
    with open(MINED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            total_count += 1
            obj = json.loads(line)
            
            if cleaner.is_english(obj['sentence']):
                dropped_count += 1
            else:
                kept.append(obj)
    
    with open(CLEAN_FILE, 'w', encoding='utf-8') as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
            
    print(f"[SUCCESS] Cleaning complete.")
    print(f"\tOriginal: {total_count}")
    print(f"\tDropped:  {dropped_count} (English pollution)")
    print(f"\tKept:     {len(kept)}")
    print(f"\tSaved to: {CLEAN_FILE}")


def run_analysis():
    print(f"\n[STEP 4] Generating visualization plots...")
    
    if not os.path.exists(CLEAN_FILE):
        print(f"[ERROR] Clean file not found at {CLEAN_FILE}. Run 'clean' first.")
        return

    # Load Data
    data = []
    with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print("[ERROR] Clean file is empty. No data to plot.")
        return

    df = pd.DataFrame(data)
    
    viz = LoanwordVisualizer(df)
    
    # Generate Plots
    print("\tGenerating Part-of-Speech Plot...")
    viz.plot_pos_distribution(os.path.join(PLOTS_DIR, "1_pos_distribution.png"))
    
    print("\tGenerating Integration Strategies Plot...")
    viz.plot_integration_strategies(os.path.join(PLOTS_DIR, "2_integration_strategies.png"))
    
    print("\tGenerating Visual Fidelity Plot...")
    viz.plot_visual_fidelity(os.path.join(PLOTS_DIR, "3_visual_fidelity.png"))
    
    print(f"[SUCCESS] Plots saved to directory: {PLOTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loanword Analysis Pipeline (TFM)")
    
    parser.add_argument(
        "step", 
        choices=["generate", "mine", "clean", "analyze", "all"],
        help="The pipeline step to execute."
    )
    
    args = parser.parse_args()
    
    if args.step in ["generate", "all"]:
        run_generation()
        
    if args.step in ["mine", "all"]:
        run_mining()
        
    if args.step in ["clean", "all"]:
        run_cleaning()
        
    if args.step in ["analyze", "all"]:
        run_analysis()