# pipeline.py
# -------------------------------------------------------------------
# data pipeline orchestrator for synthetic seed generation and mining
# -------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# aug-2026

import json
import os
import logging
from typing import List

from .domain.scraper.wiktionary import scrape_wiktionary
from .domain.generators.seeds import generate_seeds
from .mining.miner import mine_corpora
from .mining.cleaner import clean_sentences
from .analysis.annotation import get_annotation_stats
from .analysis.plot import generate_plots, plot_token_analysis
from .analysis.stats import generate_dataset_stats, generate_granular_stats

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


def run_cleaning(input_path: str, output_path: str, gold: str):
    """Orchestrates formatting, noise-filtering, and test-set exclusion."""
    logging.info("\n--- [cleaning & formatting] ---")
    if not input_path or not os.path.exists(input_path):
        logging.error(f"\t> (!) Input path not found: {input_path}")
        return
        
    # ** gold set exclusion **
    exclude_set = set()
    if os.path.exists(gold):
        logging.info(f"\t> Test =/= subset of train: {gold}")
        with open(gold, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
            for item in gold_data:
                exclude_set.add(item['data']['text'].strip()) 
                
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"\t> Cleaning mined sentences from {input_path}...")
    
    clean_sentences(input_path, output_path, exclude_set=exclude_set)
    
@DeprecationWarning
def run_truecase(gold_path: str):
    """Applies truecasing to the gold standard dataset."""
    #import nltk
    #nltk.download('punkt_tab')
    
    logging.info("\n--- [truecasing gold set] ---")
    if not os.path.exists(gold_path):
        logging.error(f"\t> (!) Gold path not found: {gold_path}")
        return
        
    logging.info(f"\t> Truecasing sentences and spans in {gold_path}...")
    
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        original_text = item['data']['text']
        truecased_text = original_text #truecase.get_true_case(original_text)
        
        item['data']['text'] = truecased_text
        
        for annotation in item.get('annotations', []):
            for result in annotation.get('result', []):
                start = result['value']['start']
                end = result['value']['end']
                result['value']['text'] = truecased_text[start:end]
    
    with open(gold_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    logging.info(f"\t> Truecasing complete for {len(data)} annotations.")
    
def run_analysis():
    logging.info("\n--- Corpus statistics ---")
    generate_dataset_stats()
    get_annotation_stats()
    generate_plots()

def run_granular_analysis(gold_path: str, pred_path: str,tokenizer_id: str, target_langs: List[str], output_dir: str, prefix: str):
    """Orchestrates the granular error analysis and stats computations."""
    logging.info("\n--- Granular tokenization, FPs & taxonomy analysis ---")
    if not os.path.exists(pred_path):
        logging.error(f"\t> (!) Prediction file not found: {pred_path}")
        return
        
    logging.info(f"\t> Running analysis on predictions: {pred_path}")
    
    tok_csv, clf_csv, fp_csv = generate_granular_stats(
        gold_path=gold_path,
        pred_path=pred_path,
        tokenizer_id=tokenizer_id,
        target_langs=target_langs,
        output_dir=output_dir,
        prefix=prefix
    )
        
    if tok_csv and clf_csv:
        plot_token_analysis(tok_csv, clf_csv, output_dir=output_dir, prefix=prefix)
        logging.info(f"\t> Analysis complete. CSVs and plots saved to: {output_dir}")
    else:
        logging.warning("\t> (!) Granular analysis on tokenizer did not return expected CSVs.")