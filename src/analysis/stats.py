# stats.py
# ----------------------------------------------------------------
# statistics for retrieved raw, mined and processed borrowing data
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import pandas as pd
import json
import os

class BorrowingStats:
    """Computation of statistics of lexical borrowing data, and their distributions per language."""

    def __init__(self, seeds_path, mined_path, clean_path):
        self.paths = {
            "seeds": seeds_path,
            "mined": mined_path,
            "clean": clean_path
        }
        self.data = {}
        self._load_data()

    def report(self, output_dir):
        """Statistics table, for each language, saved to file."""
        langs = ['ast', 'eu', 'el']
        all_stats = {lang: self._get_language_stats(lang) for lang in langs}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # cutesy text
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
        print(report_str) # print cutesy text n save it
        txt_path = os.path.join(output_dir, "stats_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_str)
            
        # to CSV
        df_stats = pd.DataFrame(all_stats)
        csv_path = os.path.join(output_dir, "stats_summary.csv")
        df_stats.to_csv(csv_path, index=True)
        
        print(f">>> Stats saved to: {output_dir}.txt and .csv")
        
    # -----------------------------------------------------------------------------------------

    def _get_language_stats(self, lang):
        """Computes statistics for a specific language."""
    
        # seed
        seeds = self.data['seeds'][self.data['seeds']['lang'] == lang]
        total_seeds = len(seeds)
        synth_seeds = len(seeds[seeds['source_cat'] == 'Synthetic'])
        wiki_seeds = len(seeds[seeds['source_cat'] == 'Wiktionary'])

        # raw mined
        if not self.data['mined'].empty:
            mined = self.data['mined'][self.data['mined']['lang'] == lang]
        else:
            mined = pd.DataFrame(columns=['term'])
        raw_sentences = len(mined)
        found_terms_raw = mined['term'].unique() if not mined.empty else []
        seeds_found_count = len(found_terms_raw)

        # processed mined
        if not self.data['clean'].empty:
            clean = self.data['clean'][self.data['clean']['lang'] == lang]
        else:
            clean = pd.DataFrame(columns=['term'])
        valid_sentences = len(clean)
        found_terms_clean = clean['term'].unique() if not clean.empty else []
        seeds_valid_count = len(found_terms_clean)


        # ratios
        yield_per_seed = round(raw_sentences / seeds_found_count, 1) if seeds_found_count > 0 else 0
        retention_rate = round((valid_sentences / raw_sentences) * 100, 1) if raw_sentences > 0 else 0
        
        # actually valid mined sentences
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

    # -----------------------------------------------------------------------------------------

    def _load_data(self):        
        # seed data, classified according to source /synthetic or wiktionary/
        if os.path.exists(self.paths['seeds']):
            self.data['seeds'] = pd.read_csv(self.paths['seeds'])
            
            # classify source based on 'type' column
            self.data['seeds']['source_cat'] = self.data['seeds']['type'].apply(
                lambda x: 'Wiktionary' if 'wiktionary' in str(x).lower() else 'Synthetic'
            )
        else:
            print(f"(!) > Seeds file missing: {self.paths['seeds']}")
            self.data['seeds'] = pd.DataFrame(columns=['term', 'lang', 'type', 'source_cat'])

        # raw mined data
        self.data['mined'] = self._load_jsonl(self.paths['mined'])

        # clean mined data
        self.data['clean'] = self._load_jsonl(self.paths['clean'])

    def _load_jsonl(self, filepath):
        # jsonl data        
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
