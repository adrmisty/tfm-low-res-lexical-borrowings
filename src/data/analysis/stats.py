# stats.py
# ----------------------------------------------------------------
# statistics for retrieved raw, mined and processed borrowing data
# and tokenization statistics
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan/may-2026

import os
import json
import logging
import pandas as pd
from typing import Dict, Any
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from src.model.baseline.eval import _load_spans
from src.model.baseline.prompt import TAGSET

class BorrowingStats:
    """Computation of statistics of lexical borrowing data, and their distributions per language."""

    def __init__(self, seeds_path: str, mined_path: str, clean_path: str):
        self.paths = {
            "seeds": seeds_path,
            "mined": mined_path,
            "clean": clean_path
        }
        self.data: Dict[str, pd.DataFrame] = {}
        self._load_data()

    def report(self, output_dir: str):
        """Statistics table, for each language, saved to file."""
        langs = ['ast', 'eu', 'el']
        all_stats = {lang: self._get_language_stats(lang) for lang in langs}
        
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        lines.append("="*80)
        lines.append(f"{'LEXICAL BORROWING STATS':^80}")
        lines.append("="*80)
        
        headers = ["Metric", "Asturian", "Basque", "Greek"]
        row_fmt = "{:<30} | {:<12} | {:<12} | {:<12}"
        lines.append(row_fmt.format(*headers))
        lines.append("-" * 80)
        
        metrics = list(all_stats['ast'].keys())
        
        for metric in metrics:
            values = [all_stats[lang][metric] for lang in langs]
            lines.append(row_fmt.format(metric, *values))
            
        lines.append("-" * 80)
        
        total_clean = sum(all_stats[l]["Clean sentences"] for l in langs)
        total_dropped = sum(all_stats[l]["Sentences dropped"] for l in langs)
        lines.append(f"GLOBAL VALID DATASET: {total_clean} sentences.")
        lines.append(f"GLOBAL NOISE REMOVED: {total_dropped} sentences.")
        lines.append("="*80)
        
        report_str = "\n".join(lines)
        logging.info("\n" + report_str) 
        
        txt_path = os.path.join(output_dir, "stats_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_str)
            
        # to CSV
        df_stats = pd.DataFrame(all_stats)
        csv_path = os.path.join(output_dir, "stats_summary.csv")
        df_stats.to_csv(csv_path, index=True)
        
        logging.info(f"\t> Stats saved to: {output_dir}")
        
    def _get_language_stats(self, lang: str) -> Dict[str, Any]:
        """Computes statistics for a specific language."""
        # seed
        seeds = self.data['seeds'][self.data['seeds']['lang'] == lang]
        total_seeds = len(seeds)
        synth_seeds = len(seeds[seeds['source_cat'] == 'Synthetic'])
        wiki_seeds = len(seeds[seeds['source_cat'] == 'Wiktionary'])

        # raw mined
        mined = self.data['mined'][self.data['mined']['lang'] == lang] if not self.data['mined'].empty else pd.DataFrame(columns=['term'])
        raw_sentences = len(mined)
        found_terms_raw = [] if mined.empty else mined['term'].unique()
        seeds_found_count = len(found_terms_raw)

        # processed mined
        clean = self.data['clean'][self.data['clean']['lang'] == lang] if not self.data['clean'].empty else pd.DataFrame(columns=['term'])
        valid_sentences = len(clean)
        found_terms_clean = [] if clean.empty else clean['term'].unique()
        seeds_valid_count = len(found_terms_clean)

        # ratios
        yield_per_seed = round(raw_sentences / seeds_found_count, 1) if seeds_found_count > 0 else 0
        retention_rate = round((valid_sentences / raw_sentences) * 100, 1) if raw_sentences > 0 else 0
        
        dropped = raw_sentences - valid_sentences
        
        return {
            "Language": lang.upper(),
            "Total seeds": total_seeds,
            "  - Synthetic": synth_seeds,
            "  - Wiktionary": wiki_seeds,
            "Seeds mined (Raw)": seeds_found_count,
            "Seeds mined (Clean)": seeds_valid_count,
            "Raw sentences": raw_sentences,
            "Clean sentences": valid_sentences,
            "Sentences dropped": dropped,
            "Retained sentences": f"{retention_rate}%",
            "Yield (sentence/seed)": yield_per_seed
        }

    def _load_data(self):        
        if os.path.exists(self.paths['seeds']):
            self.data['seeds'] = pd.read_csv(self.paths['seeds'])
            self.data['seeds']['source_cat'] = self.data['seeds']['type'].apply(
                lambda x: 'Wiktionary' if 'wiktionary' in str(x).lower() else 'Synthetic'
            )
        else:
            logging.warning(f"\t> (!) Seeds file missing: {self.paths['seeds']}")
            self.data['seeds'] = pd.DataFrame(columns=['term', 'lang', 'type', 'source_cat'])

        self.data['mined'] = self._load_jsonl(self.paths['mined'])
        self.data['clean'] = self._load_jsonl(self.paths['clean'])

    def _load_jsonl(self, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            return pd.DataFrame()
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(data)

# --- extend: into pipeline.py ---

def generate_dataset_stats(seeds_path: str = "data/corpus/raw/synthetic_borrowings.csv",
                           mined_path: str = "data/corpus/mined/mined_sentences.jsonl",
                           clean_path: str = "data/corpus/processed/mined_sentences.clean.jsonl",
                           output_dir: str = "results/plots"):
    stats = BorrowingStats(seeds_path, mined_path, clean_path)
    stats.report(output_dir)

def generate_token_stats(tokenizer_id: str, target_langs: list, output_dir: str,
                         gold_path: str = "data/annotation/test_gold_annotations.json", 
                         pred_path: str = "results/model/mmBert/predictions_mmbert_2step_20260429_162250.json"):
    analyzer = TokenAnalysis(gold_path, pred_path, target_langs)
    tok_csv = analyzer.analyze_tokenization_fragmentation(tokenizer_id, output_dir)
    clf_csv = analyzer.analyze_per_class_performance(output_dir)
    return tok_csv, clf_csv

class TokenAnalysis:
    """Computes advanced diagnostics for lexical borrowing extraction."""
    
    def __init__(self, gold_path: str, pred_path: str, target_langs: list = None):
        self.target_langs = target_langs
        self.true_spans, self.pred_spans = _load_spans(pred_path, gold_path, target_langs)

    def analyze_tokenization_fragmentation(self, tokenizer_id: str, output_dir: str) -> str:
        logging.info(f"\t> Calculating Sub-word Fertility using [{tokenizer_id}]...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        except Exception as e:
            logging.error(f"(!) Could not load tokenizer {tokenizer_id}: {e}")
            return None
        
        stats = {
            "correct": {"1 sub-word": 0, "2 sub-words": 0, "3 sub-words": 0, "4+ sub-words": 0},
            "missed":  {"1 sub-word": 0, "2 sub-words": 0, "3 sub-words": 0, "4+ sub-words": 0}
        }

        for (case_id, text), true_label in self.true_spans.items():
            if true_label not in TAGSET: continue
                
            fertility = len(tokenizer.tokenize(text))
            
            if fertility == 1: bin_key = "1 sub-word"
            elif fertility == 2: bin_key = "2 sub-words"
            elif fertility == 3: bin_key = "3 sub-words"
            else: bin_key = "4+ sub-words"
                
            pred_label = self.pred_spans.get((case_id, text), "Native")
            status = "correct" if pred_label != "Native" else "missed"
            stats[status][bin_key] += 1

        df = pd.DataFrame(stats)
        df.index.name = 'fertility'
        df = df.reset_index()
        df["total"] = df["correct"] + df["missed"]
        df["success_rate"] = (df["correct"] / df["total"]).fillna(0)
        
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "tokenization_fragmentation_stats.csv")
        df.to_csv(out_file, index=False)
        return out_file

    def analyze_per_class_performance(self, output_dir: str) -> str:
        logging.info("\t> Calculating per-class performance breakdown...")
        intersection_keys = set(self.true_spans.keys()).intersection(set(self.pred_spans.keys()))
        y_true, y_pred = [], []
        
        for key in intersection_keys:
            t_lbl = self.true_spans.get(key)
            p_lbl = self.pred_spans.get(key)
            if t_lbl not in TAGSET: continue
            if p_lbl not in TAGSET: p_lbl = "Raw"
            y_true.append(t_lbl)
            y_pred.append(p_lbl)
            
        if not y_true: return None

        report_dict = classification_report(y_true, y_pred, labels=TAGSET, zero_division=0, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        df_report.index.name = 'taxonomy_class'
        df_report = df_report.reset_index()
        
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "per_class_performance_stats.csv")
        df_report.to_csv(out_file, index=False)
        return out_file
