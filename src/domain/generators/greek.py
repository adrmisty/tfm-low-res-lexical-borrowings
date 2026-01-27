# greek.py
# ----------------------------------------------------------------
# synthetic seed generator in Greek
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

from typing import Dict, List
from .seeds import BorrowingGenerator
from src.config import GREEK_TRANSLITERATION

class GreekGenerator(BorrowingGenerator):
    def __init__(self):
        super().__init__("el")
        self.trans_map = GREEK_TRANSLITERATION

    def generate_for_root(self, root: str) -> List[Dict]:
        """Generates Greek synthetic borrowing forms for a given root."""
        H = []
        
        # NOUNS (code-switch)
        H.append(self._make_seed(root, root, "cs_latin_raw", "NOUN"))
        
        # LIGHT VERBS (code-switch)
        if self.is_action_root(root):
            auxiliaries = ["κάνω", "κάνει", "έκανε", "κάνοντας"]
            for aux in auxiliaries:
                H.append(self._make_seed(f"{aux} {root}", root, "verb_light_latin", "VERB"))

        # TRANSLITERATION
        gk_stem = self.trans_map.get(root, "")
        
        if gk_stem:
            # NOUNS
            H.append(self._make_seed(gk_stem, root, "noun_transliterated", "NOUN"))
            
            if self.is_action_root(root):
                # LIGHT VERBS
                auxiliaries = ["κάνω", "κάνει", "έκανε", "κάνοντας"]
                for aux in auxiliaries:
                    H.append(self._make_seed(f"{aux} {gk_stem}", root, "verb_light_greek", "VERB"))
                
                # VERBS: productive suffix (-άρω)
                clean_stem = gk_stem.translate(str.maketrans("άέίόύήώ", "αειουηω"))
                
                H.append(self._make_seed(f"{clean_stem}άρω", root, "verb_morph_aro", "VERB"))
                H.append(self._make_seed(f"{clean_stem}αρισμένος", root, "verb_participle", "VERB"))

        return H