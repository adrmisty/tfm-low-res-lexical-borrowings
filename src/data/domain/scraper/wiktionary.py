# wiktionary.py
# ----------------------------------------------------------------
# scrapes and loads lexical borrowings from Wiktionary dumps
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import os
import time
import logging
import requests
import pandas as pd
from typing import List, Dict, Optional

class WiktionaryScraper:
    """Extractor for lexical borrowings in Wiktionary pages, for target languages from given source languages."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.base_url = "https://en.wiktionary.org/w/api.php"
        self.headers = {"User-Agent": "LoanwordThesisBot/1.0 (academic_research)"}
        
        # (target lang, source lang, wiktionary category)
        self.categories = [
            ("ast", "en", "Category:Asturian_terms_borrowed_from_English"),
            ("ast", "es", "Category:Asturian_terms_borrowed_from_Spanish"),
            
            ("eu",  "en", "Category:Basque_terms_borrowed_from_English"),
            ("eu",  "es", "Category:Basque_terms_borrowed_from_Spanish"),
            ("eu",  "fr", "Category:Basque_terms_borrowed_from_French"),
            
            ("el",  "en", "Category:Greek_terms_borrowed_from_English"),
            ("el",  "tr", "Category:Greek_terms_borrowed_from_Turkish"),
            ("el",  "fr", "Category:Greek_terms_borrowed_from_French"),
        ]
        
    def scrape(self, target_langs: Optional[List[str]] = None):
        """Scrapes and extracts lexical borrowings for the specified categories."""
        all_data = []
        
        for target, origin, category in self.categories:
            if target_langs and target not in target_langs:
                continue
                
            logging.info(f"\t\t> Fetching: {category} ({target} <- {origin})...")
            terms = self._get_category_members(category)
            logging.info(f"\t\t> Found {len(terms)} terms.")
            
            for term in terms:
                all_data.append({
                    "term": term,
                    "target_lang": target,
                    "origin_lang": origin,
                    "source_category": category
                })
                
        if not all_data:
            logging.info("\t\t> No data scraped for the specified languages.")
            return

        df = pd.DataFrame(all_data)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        if os.path.exists(self.csv_path):
            existing_df = pd.read_csv(self.csv_path)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['term', 'target_lang'])
            combined_df.to_csv(self.csv_path, index=False)
        else:
            df.to_csv(self.csv_path, index=False)

    def _get_category_members(self, category_title: str) -> List[str]:
        """Get all members of a category via the MediaWiki API."""
        members = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmlimit": "500",
            "format": "json"
        }
        
        while True:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            data = response.json()
            
            if "query" in data:
                for member in data["query"]["categorymembers"]:
                    if member['ns'] == 0: # namespace 0 is standard articles (words)
                        members.append(member['title'])
            
            if "continue" in data:
                params["cmcontinue"] = data["continue"]["cmcontinue"]
                time.sleep(0.1) # ** delay **
            else:
                break
                
        return members
        
    def load_seeds(self, target_langs: Optional[List[str]] = None) -> List[Dict]:
        """Reads the Wiktionary CSV and converts it into standard Seed format."""
        if not os.path.exists(self.csv_path):
            logging.error(f"\t> (!) Wiktionary file not found at {self.csv_path}")
            return []

        # columns: term, target_lang, origin_lang, source_category
        df = pd.read_csv(self.csv_path)
        seeds = []
        
        for _, row in df.iterrows():
            t_lang = row['target_lang']
            
            if target_langs and t_lang not in target_langs:
                continue

            origin = row['origin_lang']
            seed_type = f"wiktionary_{origin}"
            
            seed = {
                "term": row['term'],
                "lemma": row['term'],
                "lang": t_lang,
                "type": seed_type,
                "pos": "-"  # not specified by Wiktionary
            }
            seeds.append(seed)
            
        return seeds

# --- extend: into pipeline ---

def scrape_wiktionary(lang: str, csv_path: str = "data/corpus/raw/wiktionary_borrowings.csv"):
    scraper = WiktionaryScraper(csv_path=csv_path)
    scraper.scrape(target_langs=[lang])