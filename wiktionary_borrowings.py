# wiktionary_borrowings.py
# ----------------------------------------------------------------
# script to download lexical borrowings from Wiktionary categories
# part of the TFM: "lexical borrowings in low-resource languages"
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import requests
import pandas as pd
import time
import os

TASK = "wiktionary-lexical-borrowings"
OUTPUT_DATA_DIR = "data/raw/"
USER_AGENT = "TFM-LexBorrowings-Research/1.0 (Student Project: adrmisty@github)"
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

# format: https://en.wiktionary.org/wiki/Category:Greek_terms_borrowed_from_Spanish
contacts = [
        # ASTURIAN (low-resource)
        ("Category:Asturian_terms_borrowed_from_English", "ast", "en"),  # target (very few)
        ("Category:Asturian_terms_borrowed_from_Spanish", "ast", "es"),  # dominant language (Spain)
        
        # EUSKERA (morph. rich, low-resource)
        ("Category:Basque_terms_borrowed_from_English", "eu", "en"),     # target
        ("Category:Basque_terms_borrowed_from_Spanish", "eu", "es"),     # dominant lang (Spain)
        ("Category:Basque_terms_borrowed_from_French", "eu", "fr"),      # dominant lang (France)
        
        # GREEK (morph. rich, mid-resource)
        ("Category:Greek_terms_borrowed_from_English", "el", "en"),      # target
        ("Category:Greek_terms_borrowed_from_French", "el", "fr"),       # dominant lang (culturally)
        ("Category:Greek_terms_borrowed_from_Turkish", "el", "tr"),      # dominant lang (Ottoman Emp.)
]


def get_wiktionary_seeds_for(contacts):
    """Gathers lexical borrowing seeds from Wiktionary categories for a set of languages."""
    print(f"\n> [{TASK}] Starting...")
    master_list = []

    for cat, lang, origin in contacts:
        seeds = _get_wiktionary_seeds(cat, lang, origin)
        master_list.extend(seeds)
        
    # save downloaded seeds in comma-separated-value format
    df = pd.DataFrame(master_list)
    df.to_csv(f"{OUTPUT_DATA_DIR}/wiktionary_borrowings.csv", index=False)

    print("\n" + "="*30)
    print(f"[{TASK}] Done")
    print(f"\t> Total seeds: {len(df)}")
    print(f"\t> Saved to: {OUTPUT_DATA_DIR}")
    print("="*30)

    # quick summary
    print(df.groupby(['target_lang', 'origin_lang']).size())

# -------------------------------------------------------------------------------------------------

def _get_wiktionary_seeds(category_name, target, donor, limit=5000):
    """Downloads Wiktionary loanword data for a given language using the API."""
    
    url = "https://en.wiktionary.org/w/api.php"
    base_params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_name, # e.g. "Category:[Target]_terms_borrowed_from_[Donor]", Asturian from Spanish
        "cmlimit": limit,
        "format": "json",
        "cmtype": "page"
    }
    
    headers = {"User-Agent": USER_AGENT}
    all_members = []
    continue_token = None
        
    while True:
        params = base_params.copy()
        if continue_token:
            params["cmcontinue"] = continue_token
            
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            if "error" in data:
                print(f"\t(!) Error API: {data['error']['info']}")
                break
                
            if "query" in data and "categorymembers" in data["query"]:
                items = data["query"]["categorymembers"]
                for item in items:
                    term = item["title"]
                    
                    # filter out subcats
                    if ":" not in term: 
                        all_members.append({
                            "term": term,
                            "target_lang": target,
                            "origin_lang": donor,
                            "source_category": category_name
                        })
            
            if "continue" in data:
                continue_token = data["continue"]["cmcontinue"]
                time.sleep(0.5) # API limits
            else:
                break # no more subpages
                
        except Exception as e:
            print(f"\t(!) Connection exception: {e}")
            break
            
    print(f"\t>{category_name}, seeds: {len(all_members)}")
    return all_members

# -------------------------------------------------------------------------------------------------