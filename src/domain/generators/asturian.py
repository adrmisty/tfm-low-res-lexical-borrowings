# asturian.py
# ----------------------------------------------------------------
# synthetic seed generator in Asturian
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

from typing import Dict, List
from .base import BorrowingGenerator
from src.config import N_ROOTS

class AsturianGenerator(BorrowingGenerator):
    def __init__(self):
        super().__init__("ast")

    def generate_for_root(self, root: str) -> List[Dict]:
        """Generates Asturian synthetic borrowing forms for a given root."""
        H = []
        
        # Base stem logic
        stem = root
        if root.startswith("s") and len(root) > 1 and root[1] not in "aeiou":
            stem = "e" + stem # escan
        if "tweet" in stem: stem = stem.replace("tweet", "tuit")

        # NOUNS -> PoS: NOUN
        H.append(self._make_seed(stem, root, "noun_raw", "NOUN"))
        # N: integrated plural (-es for consonants)
        plural_native = f"{stem}s" if stem[-1] in "aeiou" else f"{stem}es"
        H.append(self._make_seed(plural_native, root, "noun_plural_native", "NOUN"))
        # N: unintegrated plural (-s for everything)
        plural_english = f"{stem}s"
        if plural_english != plural_native: # Avoid dupes if stem ends in vowel
            H.append(self._make_seed(plural_english, root, "noun_plural_english", "NOUN"))
        
        # LIGHT VERBS (COMPOUND aux_V + NOUN) -> PoS: VERB
        if self.is_action_root(root):
            auxiliaries = ["facer", "fizo", "fai", "facemos", "faen", "faciendo", "fechu"]
            for aux in auxiliaries:
                H.append(self._make_seed(f"{aux} {stem}", root, "verb_light_construction", "VERB"))
                # allow for 'facer (un) click/scan/post'
                H.append(self._make_seed(f"{aux} un {stem}", root, "verb_light_construction", "VERB"))
        
        if root in N_ROOTS: return H

        # VERBS (prescriptive) -> PoS: VERB
        def adapt_spelling(base, suffix):
            if base.endswith("g") and suffix.startswith("i"): return base + "u" + suffix
            if base.endswith("ck") and suffix.startswith("i"): return base[:-2] + "qu" + suffix
            if base.endswith("k") and suffix.startswith("i"): return base[:-1] + "qu" + suffix
            return base + suffix

        verb_stem = stem[:-1] if stem.endswith("e") and root in ["like", "update"] else stem
        
        prescriptive = adapt_spelling(verb_stem, "iar")
        H.append(self._make_seed(prescriptive, root, "verb_morph_prescriptive", "VERB"))
        H.append(self._make_seed(adapt_spelling(verb_stem, "iáu"), root, "verb_participle_prescriptive", "VERB"))
        
        # VERBS (Descriptive/Incorrect) -> PoS: VERB
        # click -> clickiar
        descriptive = verb_stem + "iar"
        if descriptive != prescriptive:
            H.append(self._make_seed(descriptive, root, "verb_morph_descriptive", "VERB"))
            H.append(self._make_seed(descriptive.replace("iar", "iáu"), root, "verb_participle_descriptive", "VERB"))

        return H