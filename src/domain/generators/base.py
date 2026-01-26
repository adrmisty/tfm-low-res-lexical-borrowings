# base.py
# ----------------------------------------------------------------
# base for synthetic seed generators / language-decoupled
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

from abc import ABC, abstractmethod
from typing import List, Dict
from src.config import N_ROOTS

class BorrowingGenerator(ABC):
    """Generates synthetic lexical borrowing forms (nouns, verbs)."""
    def __init__(self, lang_code: str):
        self.lang = lang_code

    def generate_all(self, roots: List[str]) -> List[Dict]:
        results = []
        for root in roots:
            results.extend(self.generate_for_root(root))
        return results

    @abstractmethod
    def generate_for_root(self, root: str) -> List[Dict]:
        pass

    def _make_seed(self, term, lemma, type_, pos):
        return {
            "term": term, 
            "lemma": lemma, 
            "lang": self.lang, 
            "type": type_, 
            "pos": pos
        }
        
    def is_action_root(self, root):
        return root not in N_ROOTS