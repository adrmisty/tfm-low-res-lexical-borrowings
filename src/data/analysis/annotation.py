# annotation.py
# ----------------------------------------------------------------
# generation, enrichment & format of gold test set for Label Studio
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# feb-2026

import os
import json
import uuid
import logging
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict

# --- path config ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "corpus", "processed", "mined_sentences.clean.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "annotation")
OUTPUT_FILE_LABEL = os.path.join(OUTPUT_DIR, "sample_final.json")

WIKTEXTRACT_FILES = {
    "ast": os.path.join(PROJECT_ROOT, "data", "external", "kaikki.org-dictionary-Asturian.jsonl"),
    "eu": os.path.join(PROJECT_ROOT, "data", "external", "kaikki.org-dictionary-Basque.jsonl"),
    "el": os.path.join(PROJECT_ROOT, "data", "external", "kaikki.org-dictionary-Greek.jsonl")
}

PATH_COGNET = os.path.join(PROJECT_ROOT, "data", "external", "cognet.tsv")
PATH_UNIMORPH = os.path.join(PROJECT_ROOT, "data", "external", "unimorph_eus.tsv")
PATH_CONLOAN = os.path.join(PROJECT_ROOT, "data", "external", "conloan_ell.tsv")

ANNOTATIONS_LS = os.path.join(OUTPUT_DIR, "test_gold_annotations.json") # Final V1 taxonomy version
ANNOTATIONS_FLS = os.path.join(OUTPUT_DIR, "fixed-test_gold_annotations.json")

TARGET_TOTAL = 200
TARGET_ESTABLISHED = 100

LABEL_MAP = {
    "Internationalism_Cognate": "Internationalism",
    "Adapted_Spelling": "Adapted_Orthogra",
    "Adapted_Translit": "Adapted_Orthogra", 
    "LightVerb_Translit": "LightVerb_Integrated",
    "LightVerb_Adapted": "LightVerb_Integrated",
    "LightVerb_Raw": "LightVerb_Unintegrated"
}

def sample_for_annotation():
    """Samples data, enriches with etymology/corpus info, and exports to Label Studio."""
    logging.info("\t> Sampling data for annotation...")
    sampled_data = _sample_sentences()
    
    if not sampled_data:
        return

    logging.info("\t> Enriching with etymological data...")
    enriched_data = _add_etymology_data(sampled_data)

    logging.info("\t> Checking cognates...")
    validated_data = _outside_sources(enriched_data)

    logging.info("\t> Exporting to label studio...")
    _export_to_label_studio(validated_data)

def get_annotation_stats():
    """Calculates and prints annotation statistics from the Label Studio JSON."""
    if not os.path.exists(ANNOTATIONS_LS):
        logging.error(f"\t> (!) Could not find annotation file at: {ANNOTATIONS_LS}")
        return

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
        lang = entry.get("data", {}).get("lang") or entry.get("lang", "unknown")
        
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
                    
    lines = ["\n=== GOLD STANDARD test set annotations ==="]
    for lang_key, counts in stats.items():
        lines.append(f"\nLang: [{lang_key.upper()}]")
        total_tags = sum(counts.values())
        for tag, count in sorted(counts.items(), key=lambda x: x, reverse=True):
            lines.append(f"  - {tag}: {count}")
        lines.append(f"  > Total annotations: {total_tags}")
        
    logging.info("\n".join(lines))

# --- aux functions for sampling, enrichment, and export ---

def _label(stats: Dict, lang: str, label: str):
    stats[lang][LABEL_MAP.get(label, label)] += 1

def _sample_sentences(N: int = TARGET_TOTAL, E: int = TARGET_ESTABLISHED) -> List[Dict]:
    if not os.path.exists(INPUT_FILE):
        logging.error(f"\t> (!) Error: Cleaned mined sentences file not found at {INPUT_FILE}")
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
        subset = df[df['lang'] == lang].copy()

        if subset.empty:
            logging.warning(f"\t> (!) Warning: No lexical borrowing data found for {lang}")
            continue

        mask_wikt = subset['type'].str.contains('wiktionary', case=False, na=False)
        wiktionary = subset[mask_wikt].drop_duplicates(subset=['term'])
        synthetic = subset[~mask_wikt].drop_duplicates(subset=['term'])

        n_wikt_actual = min(E, len(wiktionary))
        n_syn_target = N - n_wikt_actual
        n_syn_actual = min(n_syn_target, len(synthetic))

        sample_wikt = wiktionary.sample(n=n_wikt_actual, random_state=42).copy()
        sample_wikt['category'] = 'established'
        
        sample_syn = synthetic.sample(n=n_syn_actual, random_state=42).copy()
        sample_syn['category'] = 'synthetic'

        final_samples.extend((sample_wikt, sample_syn))
        logging.info(f"\t\t> [{lang.upper()}] Sampled -> Established: {n_wikt_actual} | Synthetic: {n_syn_actual}")

    if not final_samples:
        return []

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

def _add_etymology_data(data: List[Dict]) -> List[Dict]:
    """Checks the etymology templates in Wiktionary dumps for each term and adds source language info if available."""
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
                except Exception: continue

    for entry in data:
        for loan in entry['loans']:
            loan['etymology'] = etym_info.get(loan['term'], {"found": False})
            
    return data

def _outside_sources(data: List[Dict]) -> List[Dict]:
    """Extra: checks for cognates, integrated forms, and historical attestations in external datasets."""
    sources = {'ast_cognates': set(), 'eu_forms': set(), 'el_loans': set()}
    
    if os.path.exists(PATH_COGNET):
        with open(PATH_COGNET, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5 and parts == 'ast' and parts in ['spa', 'lat', 'xib']:
                    sources['ast_cognates'].add(parts.lower())
                    
    if os.path.exists(PATH_UNIMORPH):
        with open(PATH_UNIMORPH, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sources['eu_forms'].add(parts.lower())

    if os.path.exists(PATH_CONLOAN):
        with open(PATH_CONLOAN, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    sources['el_loans'].add(parts.lower())

    for entry in data:
        lang = entry['lang']
        for loan in entry['loans']:
            term_lower = loan['term'].lower()
            loan['is_cognate'] = (lang == 'ast' and term_lower in sources['ast_cognates'])
            loan['is_integrated'] = (lang == 'eu' and term_lower in sources['eu_forms'])
            loan['is_historical'] = (lang == 'el' and term_lower in sources['el_loans'])

    return data

def _export_to_label_studio(data: List[Dict]):
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

            if loan.get('is_cognate') or loan.get('etymology', {}).get('source_lang') in ['es', 'lat', 'fr'] or loan.get('is_historical'):
                label = "Internationalism"
            elif 'light_greek' in type_str:
                label = "LightVerb_Integrated"
            elif 'light_latin' in type_str or ('light_construction' in type_str and lang == 'ast'):
                label = "LightVerb_Unintegrated"
            elif 'light_construction' in type_str and lang == 'eu':
                label = "LightVerb_Integrated"
            elif 'transliteration' in notes or (lang == 'el' and 'noun_transliterated' in type_str):
                label = "Adapted_Orthogra"
            elif 'phonological' in notes or 'morph' in type_str or 'integrated' in type_str:
                label = "Adapted_Morph"
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
            task['predictions']['result'].append(region)

        ls_tasks.append(task)

    with open(OUTPUT_FILE_LABEL, 'w', encoding='utf-8') as f:
        json.dump(ls_tasks, f, indent=4, ensure_ascii=False)
    
    logging.info(f"\t> Exported {len(ls_tasks)} tasks to {OUTPUT_FILE_LABEL}")


def fix_labels(input_path: str = ANNOTATIONS_LS, output_path: str = ANNOTATIONS_FLS):
    if not os.path.exists(input_path):
        logging.error(f"\t> (!) Error: Could not find the input file at: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []

    for item in data:
        new_item = {
            "data": {
                "text": item.get("text", "") if "text" in item else item.get("data", {}).get("text", ""),
                "lang": item.get("lang", "") if "lang" in item else item.get("data", {}).get("lang", ""),
                "source": item.get("source", "") if "source" in item else item.get("data", {}).get("source", "")
            },
            "annotations": [{"result": []}]
        }

        # ** different potential Label Studio export formats **
        annotations_source = item.get("annotations", [])
        if not annotations_source and "label" in item:
            annotations_source = [{"result": item["label"]}]

        for ann in annotations_source:
            for span in ann.get("result", []):
                val = span.get("value", span) 
                old_labels = val.get("labels", [])
                new_labels = [LABEL_MAP.get(l, l) for l in old_labels]

                region = {
                    "id": str(uuid.uuid4())[:8],
                    "from_name": "label",  
                    "to_name": "text",     
                    "type": "labels",
                    "value": {
                        "start": val.get("start", 0),
                        "end": val.get("end", 0),
                        "text": val.get("text", ""),
                        "labels": new_labels
                    }
                }
                new_item["annotations"]["result"].append(region)

        formatted_data.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
    logging.info(f"\t> Processed {len(formatted_data)} tasks with their unique IDs and labels, saved to {output_path}")