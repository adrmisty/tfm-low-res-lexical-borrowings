# cleaner.py
# ----------------------------------------------------------------
# cleans mined data from English insertions and semantic FPs
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import re
import json
import logging
from typing import Dict, Tuple

# ** ENGLISH TEXTS **
ENGLISH_STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "was", "are", "were", "has", "had", "been",
    "references", "external", "links", "bibliography", "source", "title", "date", "author", "publisher", "retrieved",
    "archived", "original", "page", "pages", "volume", "issue", "doi", "isbn", "issn", "pmid", "journal", "university", "press",
    "abstract", "introduction", "conclusion", "chapter"
}

class EnglishFilter:
    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def is_english(self, sentence: str) -> bool:
        """Determines if a sentence's major language is English."""
        text = sentence.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        if not words: return False

        if "references" in text: return True
        if "retrieved from" in text: return True
        if "how" in words and "an" in words and "our" in words: return True
        if "tongue body" in text: return True 

        english_count = sum(1 for w in words if w in ENGLISH_STOPWORDS)
        density = english_count / len(words)
        
        if density > self.threshold: return True
        if len(words) < 6 and english_count >= 2: return True
        
        return False

class SemanticFilter:
    """Filters contexts that lead to false positives, and native homonyms that clash with lexical borrowings."""
    
    def __init__(self):
        self.false_contexts = {
            "post": ["rugbi", "tenis", "fútbol", "gol", "meta", "washington", "huffington", "diariu", "periódicu", "oficina"],
            "chat": ["mont-du-chat", "chapelle", "lac", "savoie", "saboya", "comuña", "francia", "oise"],
            "bot":  ["bot.", "zool.", "biol.", "sociedá", "nat.", "ser."],
            "bug":  ["bunny", "looney", "river", "rio"],
            "hack": ["hack-a-shaq"],
            "ban":  ["ki-moon", "ki-mun", "banes", "bans", "jura", "cubanu", "croacia", "croatia", "hungary", "hungría", "12th", "xii"],
            "log":  ["logarithm", "logaritmo", "les loges", "equation", "ecuación", "ph", "=", "+", "funtzio", "matemática"],
            "troll": ["mitoloxía", "mythology", "gnome", "dwarf", "fantasy", "tolkien", "harry potter"],
            "check": ["republic", "checa", "chess", "xedrez"],
            "cloud": ["strife", "final fantasy", "saint-cloud"]
        }
        
        self.homonym_terms = {
            "postes", "poste",      # poste -> 'pole'
            "postiar", "postiáu",   # postiar -> 'to place something'
            "bana", "banatu", "banatzen", "banak", # bana -> 'each?', banatu -> 'to distribute',
            "απ", # truncated 'από' preposition 'from' 
        }

    def is_false_positive(self, entry: Dict) -> bool:
        term = entry.get('term', '').lower()
        lemma = entry.get('lemma', '').lower()
        sentence = entry.get('sentence', '').lower()
        lang = entry.get('lang', '')

        if lang == 'eu' and lemma == 'ban':
            return True

        if lang == 'el' and term == 'απ':
            return True

        if lang == 'ast' and term in self.homonym_terms:
            digital_keywords = ["internet", "blog", "web", "rede", "social", "facebook", "twitter", "instagram", "online"]
            if not any(k in sentence for k in digital_keywords):
                return True

        if lemma in self.false_contexts:
            triggers = self.false_contexts[lemma]
            for trigger in triggers:
                if trigger in sentence:
                    return True
                    
        return False

class MiningCleaner:
    """Applies pre-defined cleaning filters to mined data in a file."""
    
    def __init__(self):
        self.eng_filter = EnglishFilter()
        self.sem_filter = SemanticFilter()
    
    def clean_file(self, input_path: str, output_path: str) -> Tuple[int, int, int]:
        kept = []
        dropped_eng = 0
        dropped_sem = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    
                    if self.eng_filter.is_english(obj['sentence']):
                        dropped_eng += 1
                        continue
                    
                    if self.sem_filter.is_false_positive(obj):
                        dropped_sem += 1
                        continue
                        
                    kept.append(obj)
                    
                except json.JSONDecodeError:
                    continue
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for k in kept:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
                
        return len(kept), dropped_eng, dropped_sem

# --- extend: into pipeline.py ---

def clean_sentences(input_path: str, output_path: str):
    cleaner = MiningCleaner()
    kept, dropped_eng, dropped_sem = cleaner.clean_file(input_path, output_path)
    
    logging.info(f"\t> Cleaning complete, final sentences: {kept}")
    logging.info(f"\t> Dropped (English density): {dropped_eng}")
    logging.info(f"\t> Dropped (Semantic noise): {dropped_sem}")