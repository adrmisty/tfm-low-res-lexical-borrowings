# wikipedia_miner.py
# -----------------------------------------------------------------------------
# script to mine context sentences from Wikipedia using synthetic seeds
# and quickly analyze results (sentences found, examples, code-switches, etc.)
# part of the TFM: "lexical borrowings in low-resource languages"
# -----------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import json
import pandas as pd
import wikipediaapi
import requests
import time
import re
import os
import sys

# basic config.
TASK = "WIKIPEDIA-MINING"
OUTPUT_DATA_DIR="data/processed/"
os.makedirs("data/processed", exist_ok=True)
INPUT_SEEDS = "data/raw/synthetic_borrowings.csv"
OUTPUT_MINED = "data/processed/mined_sentences.jsonl"
OUTPUT_CLEANED = OUTPUT_MINED.replace(".jsonl",".cleaned.jsonl")
USER_AGENT = "TFM-Borrowing-Research/1.0 (adrmisty@student.ehu.eus)"

# key: synthetic lemma
# value: disqualificators
# from the first run of wikipedia_mining
# TODO extend this ugh
FALSE_POSITIVES = {
    "scan": ["Scania", "Skåne", "VABIS", "Saab", "Volkswagen"], # some region in sweden/some car
    "ban":  ["Ki-moon", "Ki-mun", "Ban Ki", "Naciones Xuníes"],
    "bug":  ["Bugs Bunny", "Bunny", "Looney", "Warner", "Disney", "Rabbit"], # bugs
    "post": ["Washington Post", "The Post", "New York Post", "post reges", "Post-Newsweek", "Post-punk", "post-punk"], # preposition
    "bot":  ["Bot.", "Zool.", "Acta Bot", "Nat.,Bot", "Ser. Bot"], # botanicals
    "scroll": ["Elder Scrolls", "Mojang", "Scrolls", "Morrowind", "Oblivion"], # videogames
    "like": ["Like a Rolling Stone", "Like a Prayer", "Like a Virgin", "Nothing's Shocking", "Smells Like Teen Spirit"], # songs
    "hack": ["Hack-a-Shaq", "Hack and slash", "hack and slash"] # jeremy mentioned hacknslash
}

# -------------------------------------------------------------------------------------------------

def mine_synthetic_seeds():
    print(f"\n> [{TASK}] Starting Wikipedia mining process...")

    if not os.path.exists(INPUT_SEEDS): # get synthetic seeds to look for in Wikipedia
        print(f"Error: {INPUT_SEEDS} not found. Run synthetic_seeds.py!")
        sys.exit(1)

    df_seeds = pd.read_csv(INPUT_SEEDS)
    seeds = df_seeds[['term', 'lang']].drop_duplicates()
    print(f"\tLoaded {len(df_seeds)} seeds, uniques {len(seeds)} seeds")

    results = []
    processed_count = 0

    wikis = {}
    for lang in df_seeds['lang'].unique():
        wikis[lang] = _setup_wiki(lang)

    for index, row in df_seeds.iterrows():
        term = row['term']
        lang = row['lang']
        lemma = row['lemma']
        m_type = row['type']
        
        processed_count += 1
        print(f"\r\t[{TASK}] Searching: {term} ({lang}) [{processed_count}/{len(df_seeds)}]", end="")
        
        # pages containing term
        found_titles = _search_wiki(term, lang, k=2) # Limit 2 to go fast
        if not found_titles:
            continue
            
        # context sentences containing the term
        wiki = wikis[lang]
        for title in found_titles:
            try:
                page = wiki.page(title)
                if page.exists():
                    matches = _get_sentences(page.text, term)
                    for m in matches:
                        results.append({
                            "term": term,
                            "lemma": lemma,
                            "lang": lang,
                            "type": m_type,
                            "sentence": m,
                            "source_page": title
                        })
            except Exception:
                continue

    print(f"\n\n> [{TASK}] Saving mined context sentences with synthetic borrowings...")
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_MINED), exist_ok=True)
    
    if not df_results.empty:
        df_results.to_json(OUTPUT_MINED, orient="records", lines=True, force_ascii=False)
        print(f"\t> Mined {len(df_results)} sentences, saved to {OUTPUT_MINED}")
        print(f"\n\t> Examples:\n{df_results[['term', 'sentence']].head(3).to_string()}\n")
    else:
        print(f"\t> WARNING: 0 sentences extracted :(")
    print("="*30)
    
    # first clean out any false positives
    _clean_mined_results()
    
    # some basic analysis
    _analyze_mined_results()
    
    return df_results

# -------------------------------------------------------------------------------------------------

def _setup_wiki(lang):
    """Initializes Wikipedia API for a specific language with 'wikipediaapi'."""
    return wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

def _get_sentences(text, term):
    """
    Splits text into sentences and returns those containing the individual term.
    """
    sentences = []
    raw_sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' ')) # punct, newl
    
    for sent in raw_sentences:
        if re.search(r'\b' + re.escape(term) + r'\b', sent, re.IGNORECASE): # casing
            sentences.append(sent.strip())
    
    return sentences

def _search_wiki(term, lang, k=3):
    """
    Retrieves top k pages from Wikipedia containing the given term.
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f'"{term}"', # exact phrase search
        "format": "json",
        "srlimit": k
    }
    try:
        time.sleep(0.1) 
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=5)
        data = r.json()
        
        titles = []
        if "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                titles.append(item["title"])
        return titles
    except Exception as e:
        return []

def _analyze_mined_results(top_k=10):
    if not os.path.exists(OUTPUT_MINED):
        print(f"\t(!) Error: {OUTPUT_MINED} does not exist. Run wikipedia_mining.py first!")
        return

    df = pd.read_json(OUTPUT_MINED, lines=True, orient="records")
    
    print(f"> [{TASK}] Total sentences extracted: {len(df)}")

    if len(df) == 0:
        print("\t(!) Nothing found :(")
        return

    print("\t> Per-language results:")
    print(df['lang'].value_counts())

    print(f"\n\tMorphological adaptation type (top {top_k}):")
    print(df['type'].value_counts().head(top_k))

    print("\tMost frequent terms:")
    print(df['term'].value_counts().head(top_k))
    
    morph_hits = df[~df['type'].str.contains("code_switching", case=False, na=False)]
    if not morph_hits.empty:
        for i, row in morph_hits.head(5).iterrows():
            print(f"- [{row['lang']}] {row['term']} ({row['type']}):")
            print(f"  \"{row['sentence'][:150]}...\"")
            print("")
    else:
        print(">t No morphological adaptations found, only code-switching")

def _clean_mined_results(triggers=FALSE_POSITIVES):
    """Cleans mined results from false positives based on predefined triggers
    (that I found in initial executions of this script)."""
    print(f"\n[{TASK}] Cleaning {OUTPUT_MINED} with mined sentences...")
    
    kept_count = 0
    removed_count = 0
    
    if not os.path.exists(OUTPUT_MINED):
        print(f"(!) Error: {OUTPUT_MINED} not found.")
        return

    with open(OUTPUT_MINED, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_CLEANED, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip(): continue
            
            try:
                data = json.loads(line)
                lemma = data.get("lemma")
                sentence = data.get("sentence", "")
                
                discard = False
                
                if lemma in triggers:
                    for trigger in triggers[lemma]:
                        # case-sensitive
                        if trigger in sentence:
                            discard = True
                            break
                            
                if lemma == "bot" and data.get("term") == "bot" and "Bot." in sentence:
                     discard = True

                if not discard:
                    fout.write(line)
                    kept_count += 1
                else:
                    removed_count += 1
                    
            except json.JSONDecodeError:
                continue

    print(f"\t> Cleaned mined sentences from false positives:")
    print(f"\t\tKept {kept_count} sentences; removed {removed_count} false positives")

# -------------------------------------------------------------------------------------------------

mine_synthetic_seeds()