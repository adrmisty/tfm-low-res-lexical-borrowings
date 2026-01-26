# basque.py
# ----------------------------------------------------------------
# synthetic seed generator in Basque
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

from typing import Dict, List
from .base import BorrowingGenerator
from src.config import N_ROOTS

class BasqueGenerator(BorrowingGenerator):
    def __init__(self):
        super().__init__("eu")

    def generate_for_root(self, root: str) -> List[Dict]:
        """Generates Basque synthetic borrowing forms for a given root."""
        H = []
        
        # phonetic adaptations
        phonetic_stem = root.lower().replace("ch", "tx").replace("sh", "x").replace("ck", "k").replace("c", "k").replace("q", "k")
        if "tweet" in phonetic_stem: phonetic_stem = phonetic_stem.replace("tweet", "tuit")
        if phonetic_stem.startswith("s") and len(phonetic_stem) > 1 and phonetic_stem[1] not in "aeiou": 
            phonetic_stem = "e" + phonetic_stem

        # NOUNS
        H.append(self._make_seed(root, root, "noun_raw", "NOUN"))
        # N: definite article -a
        term_a = phonetic_stem if phonetic_stem.endswith("a") else f"{phonetic_stem}a"
        H.append(self._make_seed(term_a, root, "noun_integrated_sg", "NOUN"))
        # N: plural -ak
        term_ak = f"{phonetic_stem}k" if phonetic_stem.endswith("a") else f"{phonetic_stem}ak"
        H.append(self._make_seed(term_ak, root, "noun_integrated_pl", "NOUN"))

        if root in N_ROOTS: return H

        # LIGHT VERBS (with 'egin')
        auxiliaries = [" egin", " egiten", " egingo", " egin du", " egin zen"]
        for aux in auxiliaries:
            H.append(self._make_seed(f"{phonetic_stem}{aux}", root, "verb_light_construction", "VERB"))

        # VERBS (with -tu, -tzen)
        verb_stem = phonetic_stem[:-1] if phonetic_stem.endswith("e") and root in ["like", "update"] else phonetic_stem
        connector = "a" if verb_stem[-1] not in "aeiou" else ""
        
        H.append(self._make_seed(f"{verb_stem}{connector}tu", root, "verb_morph_integrated", "VERB"))
        H.append(self._make_seed(f"{verb_stem}{connector}tzen", root, "verb_habitual", "VERB"))

        return H