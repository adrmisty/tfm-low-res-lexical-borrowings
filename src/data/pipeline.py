# pipeline.py
# -------------------------------------------------------------------
# data pipeline orchestrator for synthetic seed generation and mining
# -------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import os
import logging
from typing import List

from .domain.scraper.wiktionary import scrape_wiktionary
from .domain.generators.seeds import generate_seeds
from .mining.miner import mine_corpora
from .mining.cleaner import clean_sentences
from .analysis.annotation import get_annotation_stats
from .analysis.plot import generate_plots, plot_token_analysis
from .analysis.stats import generate_token_stats

def run_scraping(langs: List[str]):
    logging.info("\n--- Wikipedia loanword scraping ---")
    for lang in langs:
        logging.info(f"\t> Language: [{lang.upper()}]...")
        scrape_wiktionary(lang)

def run_generation(langs: List[str]):
    logging.info("\n--- Synthetic seed generation ---")
    for lang in langs:
        logging.info(f"\t> Morphological permutations for language: [{lang.upper()}]...")
        generate_seeds(lang)

def run_mining(langs: List[str], corpus_dir: str, output_dir: str):
    """Orchestrates the scanning of monolingual corpora for seed matches."""
    logging.info("\n--- Monolingual corpus mining ---")
    if not os.path.exists(corpus_dir):
        logging.error(f"\t> (!) Corpus directory not found: {corpus_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    for lang in langs:
        logging.info(f"\t> Mining sentences for [{lang.upper()}]...")
        mine_corpora(lang, corpus_dir, output_dir)

def run_cleaning(input_path: str, output_path: str):
    """Orchestrates the formatting and noise-filtering of mined sentences."""
    logging.info("\n--- [cleaning & formatting] ---")
    if not input_path or not os.path.exists(input_path):
        logging.error(f"\t> (!) Input path not found: {input_path}")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"\t> Cleaning mined sentences from {input_path}...")
    clean_sentences(input_path, output_path)

def run_analysis():
    logging.info("\n--- Corpus statistics ---")
    get_annotation_stats()
    generate_plots()

def run_token_analysis(gold_path: str, pred_path: str,tokenizer_id: str, target_langs: List[str], output_dir: str):
    """Orchestrates the granular error analysis for tokenization and taxonomy classification."""
    logging.info("\n--- Granular Tokenization & Taxonomy Analysis ---")
    if not os.path.exists(pred_path):
        logging.error(f"\t> (!) Prediction file not found: {pred_path}")
        return
        
    logging.info(f"\t> Running analysis on predictions: {pred_path}")
    
    tok_csv, clf_csv = generate_token_stats(
        gold_path=gold_path,
        pred_path=pred_path,
        tokenizer_id=tokenizer_id,
        target_langs=target_langs,
        output_dir=output_dir
    )

    if tok_csv and clf_csv:
        plot_token_analysis(tok_csv, clf_csv, output_dir=output_dir)
        logging.info(f"\t> Analysis complete. CSVs and plots saved to: {output_dir}")
    else:
        logging.warning("\t> (!) Granular analysis did not return expected CSVs.")