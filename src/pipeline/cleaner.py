# cleaner.py
# ----------------------------------------------------------------
# cleans mined data from potential English insertions
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import re
from src.config import ENGLISH_STOPWORDS

class EnglishFilter:
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def is_english(self, sentence: str) -> bool:
        """Determines if a sentence's major language is English."""
        text = sentence.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        if not words: return False

        if "references" in text: return True
        if "how" in words and "an" in words and "our" in words: return True

        # stopword counts
        english_count = sum(1 for w in words if w in ENGLISH_STOPWORDS)
        density = english_count / len(words)
        
        if density > self.threshold: return True
        if len(words) < 6 and english_count >= 2: return True
        
        return False