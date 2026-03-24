# annotation.py
# ----------------------------------------------------------------
# generation, enrichment & format of gold test set for Label Studio
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# feb-2026

from collections import Counter, defaultdict

import pandas as pd
import json
import uuid
import os

# --- PATH CONFIGURATION ---
INPUT_FILE = "data/processed/mined_sentences.clean.jsonl"
OUTPUT_DIR = "data/annotation/final"
OUTPUT_FILE_LABEL = os.path.join(OUTPUT_DIR, "sample_final.json")

WIKTEXTRACT_FILES = {
    "ast": "data/external/kaikki.org-dictionary-Asturian.jsonl",
    "eu": "data/external/kaikki.org-dictionary-Basque.jsonl",
    "el": "data/external/kaikki.org-dictionary-Greek.jsonl"
}

LABEL_MAP = {
    # label-studio format fixes
    "Internationalism_Cognate": "Internationalism",
    "Adapted_Spelling": "Adapted_Orthogra",
    
    # hierarchy (2nd level) fixes
    "LightVerb_Translit": "LightVerb_Integrated",
    "LightVerb_Adapted": "LightVerb_Integrated",
    "LightVerb_Raw": "LightVerb_Unintegrated"
}


PATH_COGNET = "data/external/cognet.tsv"
PATH_UNIMORPH = "data/external/unimorph_eus.tsv"
PATH_CONLOAN = "data/external/conloan_ell.tsv"

ANNOTATIONS_LS = os.path.join(OUTPUT_DIR, "test_gold_annotations.json")
ANNOTATIONS_FLS = os.path.join(OUTPUT_DIR, "fixed-test_gold_annotations.json")

TARGET_TOTAL = 200
TARGET_ESTABLISHED = 100

def sample_for_annotation():
    """Samples data, enriches with etymology/corpus info off of Cognet/Wiktionary, and exports to Label Studio."""
    print("> Sampling data for annotation...")
    sampled_data = _sample_sentences()
    
    if not sampled_data:
        return

    print("> Enriching with etymological data...")
    enriched_data = _add_etymology_data(sampled_data)

    print("> Checking cognates...")
    validated_data = _validate_corpus(enriched_data)

    print("> Exporting to label studio...")
    _export_to_label_studio(validated_data)

def get_annotation_stats():
    """Calculates and prints annotation statistics from the Label Studio JSON."""
    with open(ANNOTATIONS_LS, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        content = "[" + content.replace("}\n{", "},\n{").replace("}\r\n{", "},\n{").strip("[]") + "]"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in content.splitlines() if line.strip()]

    stats = defaultdict(Counter)

    for entry in data:
        # after format fix the language should be in the root 'data' field
        lang = entry.get("data", {}).get("lang") or entry.get("lang", "unknown")
        
        # label studio native format or json-min
        if "annotations" in entry:
            for a in entry["annotations"]:
                if a.get("was_cancelled"): continue
                for result in a.get("result", []):
                    for label in result.get("value", {}).get("labels", []):
                        _label(stats, lang, label)
                        
        elif "label" in entry:
            for label_block in entry["label"]:
                for label in label_block.get("labels", []):
                    _label(stats, lang, label)
                    
    print("=== GOLD STANDARD test set annotations ===")
    for lang_key, counts in stats.items():
        print(f"\nLang: [{lang_key.upper()}]")
        total_tags = sum(counts.values())
        for tag, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {tag}: {count}")
        print(f"  > Total annotations: {total_tags}")

def _label(stats, lang, label):
    """Renaming label studio labels to match the final tagset."""
    stats[lang][LABEL_MAP.get(label, label)] += 1

def _sample_sentences(N=TARGET_TOTAL, E=TARGET_ESTABLISHED):
    """Generates a sample of max N sentences per lang (scaling synthetic if established loans are not enough)."""
    if not os.path.exists(INPUT_FILE):
        print(f"(!) > Error: Cleaned mined sentences file not found at {INPUT_FILE}")
        return []

    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue    

    df = pd.DataFrame(data)
    final_samples = []

    for lang in ['ast', 'eu', 'el']:
        print(f"\nProcessing [{lang}]...")
        subset = df[df['lang'] == lang].copy()

        if subset.empty:
            print(f"(!) > Warning: No lexical borrowing data found for {lang}")
            continue

        mask_wikt = subset['type'].str.contains('wiktionary', case=False, na=False)
        wiktionary = subset[mask_wikt].drop_duplicates(subset=['term'])
        synthetic = subset[~mask_wikt].drop_duplicates(subset=['term'])

        # must ensure the target total per lang
        n_wikt_actual = min(E, len(wiktionary))
        n_syn_target = N - n_wikt_actual
        n_syn_actual = min(n_syn_target, len(synthetic))

        sample_wikt = wiktionary.sample(n=n_wikt_actual, random_state=42).copy()
        sample_wikt['category'] = 'established'
        
        sample_syn = synthetic.sample(n=n_syn_actual, random_state=42).copy()
        sample_syn['category'] = 'synthetic'

        final_samples.extend((sample_wikt, sample_syn))
        print(f"    > Established LWs sampled: {n_wikt_actual}")
        print(f"    > Synthetic LWs sampled:   {n_syn_actual}")
        print(f"    > Total for {lang}:        {n_wikt_actual + n_syn_actual}")

    if not final_samples:
        return []

    # df > jsonl structure for LS
    df_sample = pd.concat(final_samples)
    structured_data = []
    
    for idx, row in df_sample.iterrows():
        entry = {
            "id": idx,
            "lang": row['lang'],
            "sentence": row['sentence'],
            "source": str(row.get('source_page', '')),
            "loans": [{
                "term": str(row['term']),
                "role": "main",
                "category": str(row['category']),
                "type": str(row.get('type', '')),
                "notes": ""
            }]
        }
        structured_data.append(entry)

    return structured_data


def _add_etymology_data(data):
    """Enriches data with Wiktextract etymology."""
    term_map = {}
    for entry in data:
        for loan in entry['loans']:
            t = loan['term']
            if t not in term_map: term_map[t] = []
            term_map[t].append(entry['id'])

    etym_info = {}
    for lang, filepath in WIKTEXTRACT_FILES.items():
        if not os.path.exists(filepath): continue
        
        with open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    word = record.get("word")
                    if word in term_map:
                        templates = record.get("etymology_templates", [])
                        for t in templates:
                            name = t.get("name", "").lower()
                            if "bor" in name or "loan" in name or "der" in name or "inh" in name:
                                etym_info[word] = {
                                    "found": True,
                                    "source_lang": t.get("args", {}).get("2", "unknown")
                                }
                                break
                except Exception as e: continue

    for entry in data:
        for loan in entry['loans']:
            loan['etymology'] = etym_info.get(loan['term'], {"found": False})
            
    return data


def _validate_corpus(data):
    """Validates terms against CogNet, Unimorph, and ConLoan."""
    sources = {'ast_cognates': set(), 'eu_forms': set(), 'el_loans': set()}
    
    # > CogNet (Asturian cognates)
    if os.path.exists(PATH_COGNET):
        with open(PATH_COGNET, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5 and parts[1] == 'ast' and parts[3] in ['spa', 'lat', 'xib']:
                    sources['ast_cognates'].add(parts[2].lower())
                    
    # > Unimorph (Basque morphological forms)
    if os.path.exists(PATH_UNIMORPH):
        with open(PATH_UNIMORPH, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sources['eu_forms'].add(parts[1].lower())

    # > ConLoan (Greek historical loans)
    if os.path.exists(PATH_CONLOAN):
        with open(PATH_CONLOAN, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    sources['el_loans'].add(parts[0].lower())

    # Attach flags
    for entry in data:
        lang = entry['lang']
        for loan in entry['loans']:
            term_lower = loan['term'].lower()
            
            loan['is_cognate'] = (lang == 'ast' and term_lower in sources['ast_cognates'])
            loan['is_integrated'] = (lang == 'eu' and term_lower in sources['eu_forms'])
            loan['is_historical'] = (lang == 'el' and term_lower in sources['el_loans'])

    return data

def _export_to_label_studio(data):
    """Formats the enriched data into Label Studio JSON with pre-populated predictions for annotation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ls_tasks = []

    for entry in data:
        text = entry['sentence']
        task = {
            "data": {
                "text": text,
                "lang": entry['lang'],
                "source": entry.get('source', '')
            },
            "predictions": [{"model_version": "final_tagset", "result": []}]
        }

        for loan in entry['loans']:
            term = loan['term']
            start = text.find(term)
            if start == -1: continue

            type_str = str(loan.get('type', '')).lower()
            notes = str(loan.get('notes', '')).lower()
            lang = entry['lang']
            
            label = "Raw"

            # established or cognates
            if loan.get('is_cognate') or loan.get('etymology', {}).get('source_lang') in ['es', 'lat', 'fr'] or loan.get('is_historical'):
                label = "Internationalism_Cognate"
            
            # light verbs
            elif 'light_greek' in type_str:
                label = "LightVerb_Translit"
            elif 'light_latin' in type_str or ('light_construction' in type_str and lang == 'ast'):
                label = "LightVerb_Raw"
            elif 'light_construction' in type_str and lang == 'eu':
                label = "LightVerb_Adapted"
            
            # morphology and spelling integration
            elif 'transliteration' in notes or (lang == 'el' and 'noun_transliterated' in type_str):
                label = "Adapted_Translit"
            elif 'phonological' in notes or 'morph' in type_str or 'integrated' in type_str:
                label = "Adapted_Morph"
            
            # code switch
            elif 'raw' in type_str or 'cs_latin' in type_str:
                label = "Raw"

            region = {
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": start + len(term),
                    "text": term,
                    "labels": [label]
                }
            }
            task['predictions'][0]['result'].append(region)

        ls_tasks.append(task)

    with open(OUTPUT_FILE_LABEL, 'w', encoding='utf-8') as f:
        json.dump(ls_tasks, f, indent=4, ensure_ascii=False)
    
    print(f">>> Exported {len(ls_tasks)} tasks to {OUTPUT_FILE_LABEL}")


def fix_labels(input_path: str = ANNOTATIONS_LS, output_path: str = ANNOTATIONS_FLS):
    """Fixes label names in the Label Studio annotations to match the final tagset so that they can be (re-)imported."""
    if not os.path.exists(input_path):
        print(f"(!) Error: Could not find the input file at:\n{input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []

    for item in data:
        new_item = {
            # Removed the root 'id' to prevent database collisions during import
            "data": {
                "text": item.get("text", ""),
                "lang": item.get("lang", ""),
                "source": item.get("source", "")
            },
            "annotations": [{"result": []}]
        }

        if "label" in item:
            for span in item["label"]:
                old_labels = span.get("labels", [])
                new_labels = [LABEL_MAP.get(l, l) for l in old_labels]

                region = {
                    "id": str(uuid.uuid4())[:8], # <-- THE MAGIC FIX: Unique UI ID
                    "from_name": "label",  
                    "to_name": "text",     
                    "type": "labels",
                    "value": {
                        "start": span["start"],
                        "end": span["end"],
                        "text": span["text"],
                        "labels": new_labels
                    }
                }
                new_item["annotations"][0]["result"].append(region)

        formatted_data.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
    print(f">>> Processed {len(formatted_data)} tasks with their unique IDs and labels, saved to {output_path}")