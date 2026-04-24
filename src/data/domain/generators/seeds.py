# seeds.py
# ----------------------------------------------------------------
# base for synthetic seed generators / language-decoupled
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import logging
import os
from abc import ABC, abstractmethod
from turtle import pd
from typing import List, Dict

class SeedSynthesizer(ABC):
    """Generates synthetic lexical borrowing forms (nouns, verbs, participle adjs.)."""
    
    def __init__(self, lang_code: str, roots: List[str]):
        self.lang = lang_code
        self.roots = roots

    def generate_all(self, roots: List[str]) -> List[Dict]:
        results = []
        for root in roots:
            results.extend(self.generate_for_root(root))
        return results

    @abstractmethod
    def generate_for_root(self, root: str) -> List[Dict]:
        pass

    def _make_seed(self, term: str, lemma: str, type_: str, pos: str) -> Dict:
        return {
            "term": term, 
            "lemma": lemma, 
            "lang": self.lang, 
            "type": type_, 
            "pos": pos
        }
        
    def is_action_root(self, root: str) -> bool:
        return root not in self.roots

def generate_seeds(lang: str, 
                   input_csv: str = "data/corpus/raw/wiktionary_borrowings.csv",
                   output_csv: str = "data/corpus/raw/synthetic_borrowings.csv"):
    
    from .asturian import Asturian
    from .basque import Basque
    from .greek import Greek

    roots = []
    
    if os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        lang_df = df[df['target_lang'] == lang]
        roots = lang_df['term'].tolist()

    if not roots:
        logging.info(f"\t\t> No Wiktionary roots found for [{lang.upper()}]. Using baseline tech roots.")
        roots = [
            "click", "post", "chat", "link", "tag", "tweet", "scan", "format",
            "hack", "ban", "log", "reset", "download", "stream", "like",
            "scroll", "update", "forward", "spam", "check", "spoiler",
            "pixel", "bug", "server", "cloud", "software", "hardware",
            "online", "interface", "user", "bot", "app", "troll"
        ]

    # 3. Instantiate the correct generator
    if lang == "ast":
        generator = Asturian(roots)
    elif lang == "eu":
        generator = Basque(roots)
    elif lang == "el":
        generator = Greek(roots)
    else:
        logging.error(f"\t> (!) No synthetic generator implemented for language: {lang}")
        return

    logging.info(f"\t\t> Generating synthetic forms for [{lang.upper()}] from {len(roots)} roots...")
    synthetic_data = generator.generate_all(roots)

    if not synthetic_data:
        logging.info(f"\t\t> No synthetic data generated for [{lang.upper()}].")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df = pd.DataFrame(synthetic_data)

    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        combined_df = pd.concat([existing_df, out_df]).drop_duplicates(subset=['term', 'lang', 'type'])
        combined_df.to_csv(output_csv, index=False)
    else:
        out_df.to_csv(output_csv, index=False)

    logging.info(f"\t\t> Saved {len(synthetic_data)} synthetic seeds to {output_csv}")