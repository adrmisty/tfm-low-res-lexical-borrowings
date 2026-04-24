# miner.py
# -----------------------------------------------------------------------------
# mines context sentences from Wikipedia using synthetic seeds
# -----------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import os
import re
import time
import json
import logging
import requests
import pandas as pd
import wikipediaapi
from typing import List, Dict

FALSE_POSITIVES = {
    "scan": ["Scania", "Skåne", "VABIS", "Saab", "Volkswagen"], 
    "ban":  ["Ki-moon", "Ki-mun", "Ban Ki", "Naciones Xuníes"],
    "bug":  ["Bugs Bunny", "Bunny", "Looney", "Warner", "Disney", "Rabbit"], 
    "post": ["Washington Post", "The Post", "New York Post", "post reges", "Post-Newsweek", "Post-punk", "post-punk"], 
    "bot":  ["Bot.", "Zool.", "Acta Bot", "Nat.,Bot", "Ser. Bot"], 
    "scroll": ["Elder Scrolls", "Mojang", "Scrolls", "Morrowind", "Oblivion"], 
    "like": ["Like a Rolling Stone", "Like a Prayer", "Like a Virgin", "Nothing's Shocking", "Smells Like Teen Spirit"], 
    "hack": ["Hack-a-Shaq", "Hack and slash", "hack and slash"] 
}

class WikipediaMiner:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.wikis = {} # per-language

    def get_wiki_object(self, lang: str):
        """Lazy loader for WikipediaAPI objects."""
        if lang not in self.wikis:
            self.wikis[lang] = wikipediaapi.Wikipedia(
                user_agent=self.user_agent,
                language=lang,
                extract_format=wikipediaapi.ExtractFormat.WIKI
            )
        return self.wikis[lang]

    def search_and_extract(self, seeds_df: pd.DataFrame, limit_per_seed: int = 2) -> List[Dict]:
        """Iterates seeds, finds pages, extracts sentences, and filters false positives."""
        results = []
        total_seeds = len(seeds_df)
                
        for idx, row in seeds_df.iterrows():
            term = row['term']
            lang = row['lang']
            lemma = row['lemma']
            
            # SEARCH: relevant Wikipedia Titles using Requests
            found_titles = self._search_wiki_titles(term, lang, limit=limit_per_seed)
            
            if not found_titles:
                continue

            # EXTRACT: get page content and find sentences
            wiki = self.get_wiki_object(lang)
            
            for title in found_titles:
                try:
                    page = wiki.page(title)
                    if page.exists():
                        matches = self._get_sentences(page.text, term)
                        
                        for sentence in matches:
                            # FILTER: check for identified false positives
                            if not self._is_semantic_false_positive(lemma, term, sentence):
                                results.append({
                                    "term": term,
                                    "lemma": lemma,
                                    "lang": lang,
                                    "type": row['type'],
                                    "pos": row['pos'],
                                    "sentence": sentence,
                                    "source_page": title
                                })
                except Exception:
                    continue
            
            if (idx + 1) % 50 == 0 or idx == total_seeds - 1:
                logging.info(f"\t\t> Processed {idx+1}/{total_seeds} | Found: {len(results)} sentences (Currently checking: '{term}')")    
                    
        return results

    def _search_wiki_titles(self, term: str, lang: str, limit: int = 3) -> List[str]:
        """Uses standard requests to query the Wikipedia Search API."""
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f'"{term}"', # Exact phrase search
            "format": "json",
            "srlimit": limit
        }
        try:
            time.sleep(0.05)
            r = requests.get(url, params=params, headers={"User-Agent": self.user_agent}, timeout=5)
            data = r.json()
            
            titles = []
            if "query" in data and "search" in data["query"]:
                for item in data["query"]["search"]:
                    titles.append(item["title"])
            return titles
        except Exception:
            return []

    def _get_sentences(self, text: str, term: str) -> List[str]:
        """Splits text into sentences and returns those containing the term."""
        sentences = []
        raw_sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
        pattern = r'\b' + re.escape(term) + r'\b'
        
        for sent in raw_sentences:
            if re.search(pattern, sent, re.IGNORECASE):
                clean_sent = sent.strip()
                if 10 < len(clean_sent) < 500:
                    sentences.append(clean_sent)
        return sentences

    def _is_semantic_false_positive(self, lemma: str, term: str, sentence: str) -> bool:
        """Checks for pre-identified false positives."""
        if lemma == "bot" and term == "bot" and "Bot." in sentence:
            return True

        if lemma in FALSE_POSITIVES:
            triggers = FALSE_POSITIVES[lemma]
            for trigger in triggers:
                if trigger in sentence:
                    return True
        return False

# --- extend: into pipeline.py ---

def mine_corpora(lang: str, corpus_dir: str, output_dir: str):
    seed_file = os.path.join(corpus_dir, "synthetic_borrowings.csv")
    out_file = os.path.join(output_dir, f"mined_sentences_{lang}.jsonl")
    
    if not os.path.exists(seed_file):
        logging.error(f"\t> (!) Seed file not found at {seed_file}: run generation first.")
        return

    df = pd.read_csv(seed_file)
    lang_df = df[df['lang'] == lang]
    
    if lang_df.empty:
        logging.info(f"\t> No seeds found for [{lang.upper()}]. Skipping.")
        return
        
    miner = WikipediaMiner(user_agent="LoanwordThesisBot/1.0")
    results = miner.search_and_extract(lang_df)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    logging.info(f"\t> Mining complete for [{lang.upper()}], saved {len(results)} sentences to {out_file}")