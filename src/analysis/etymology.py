# etymology.py
# ----------------------------------------------------------------
# enriches gold standard sample with etymology info from Wiktextract dumps
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# feb-2026
import json
import os

WIKTEXTRACT_FILES = {
    "ast": "data/external/kaikki.org-dictionary-Asturian.jsonl",
    "eu": "data/external/kaikki.org-dictionary-Basque.jsonl",
    "el": "data/external/kaikki.org-dictionary-Greek.jsonl"
}
GOLD_STD_FILE = "data/annotation/gold_standard.json"
OUTPUT_FILE = "data/gold_standard_etym.json"

def add_etymology_data():
    term_map, gold_data = _load_annotated_loans()
    
    etym_info = {}

    for lang, filepath in WIKTEXTRACT_FILES.items():
        if not os.path.exists(filepath):
            print(f"> (!) Warning: Wiktextract dump for {lang} not found at {filepath}")
            continue
            
        print(f"> Scanning {lang} Wiktextract...")
        
        with open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    word = record.get("word")
                    
                    if word in term_map:
                        # WIKTEXTRACT ETYMOLOGICAL DATA (not necesarily a borrowing, this example is form latin te, but it shows the kind of info we can extract)
                        # "etymology_text": "From Latin tē, from tū.", 
                        # "etymology_templates": [{"name": "inh", "args": {"1": "ast", "2": "la", "3": "te", "4": "tē"}, 
                        # "expansion": "Latin tē"}], "word": "te", "lang": "Asturian", "lang_code": "ast", "senses": [{"links": [["you", "you"]], 
                        # "glosses": ["you (second-person singular direct pronoun)"], "id": "en-te-ast-pron-SiMXsISF", 
                        # "categories": [{"name": "Asturian pronouns", "kind": "other", "parents": [], "source": "w+disamb", "_dis": "49 51"}]}, 

                        templates = record.get("etymology_templates", [])
                        
                        for t in templates:
                            name = t.get("name", "").lower()
                            args = t.get("args", {})
                            
                            if "bor" in name or "loan" in name or "der" in name:
                                # > "2": "la" ---> 2 is source lang for the word
                                source = args.get("2", "unknown")
                                
                                etym_info[word] = {
                                    "found": True,
                                    "source_lang": source,
                                    "template": name
                                }
                                break # next word
                except:
                    continue

    print("> Enriching annotated gold standard data with Wiktextract etymological info...")
    for entry in gold_data:
        for loan in entry['loans']:
            term = loan['term']
            if term in etym_info:
                loan['etymology'] = etym_info[term]
            else:
                loan['etymology'] = {"found": False}

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, indent=4, ensure_ascii=False)
    print(f">>> Saved enriched data to {OUTPUT_FILE}")
    
# -----------------------------------------------------------------------------------------


def _load_annotated_loans(input_file: str = GOLD_STD_FILE):
    """Extracts a set of all unique mined/recognized loanwords."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # entries where the mined terms appear
    term_map = {}
    for entry in data:
        for loan in entry['loans']:
            t = loan['term']
            if t not in term_map: term_map[t] = []
            term_map[t].append(entry['id'])
    return term_map, data
