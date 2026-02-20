# annotation.py
# ----------------------------------------------------------------
# utility functions for annotation of gold standard testsample
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# feb-2026
import pandas as pd
import json
import os

INPUT_FILE = "data/processed/mined_sentences.clean.jsonl"
OUTPUT_DIR = "data/annotation"
#OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gold_standard_sample.csv")
OUTPUT_FILE_CSV = os.path.join(OUTPUT_DIR, "annotations.csv.xlsx")
OUTPUT_FILE_JSON = os.path.join(OUTPUT_DIR, "gold_standard.json")
OUTPUT_FILE_LABEL = os.path.join(OUTPUT_DIR, "label_studio_import.json")
SAMPLE_SIZE = 100

# columns
# new category: old/new
cols = ['lang', 'term', 'sentence', 'IS_VALID_LOAN', 'OTHER_LOANS', 'NOTES', 'category', 'type', 'source_page']

def sample_in_csv(n: int=SAMPLE_SIZE, split: float=0.5):
    """Generates gold sample to-be-annotated of n random sentences."""
    if not os.path.exists(INPUT_FILE):
        print("(!) > Error: cleaned mined sentences file not found")
        return

    print(f"> Loading data from {INPUT_FILE}...")
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
    target_per_group = n * split  # fifty fifty

    for lang in ['ast', 'eu', 'el']:
        print(f"\nProcessing [{lang}]...")
        subset = df[df['lang'] == lang].copy()

        if subset.empty:
            print(f"(!) > Warning: No lexical borrowing data found for {lang}")
            continue

        # ** WIKTIONARY ** established borrowings
        mask_wikt = subset['type'].str.contains('wiktionary', case=False, na=False)
        pool_wikt = subset[mask_wikt].copy()
        # ** RAW/NEW ** synthetic borrowings
        pool_syn = subset[~mask_wikt].copy()

        wiktionary = pool_wikt.sample(frac=1, random_state=42).drop_duplicates(subset=['term'])
        synthetic = pool_syn.sample(frac=1, random_state=42).drop_duplicates(subset=['term'])

        # sample with the same seed        
        n_wikt = int(min(target_per_group, len(wiktionary)))
        sample_wikt = wiktionary.sample(n=n_wikt, random_state=42).copy()
        sample_wikt['category'] = 'established'
        n_syn = int(min(target_per_group, len(synthetic)))
        sample_syn = synthetic.sample(n=n_syn, random_state=42).copy()
        sample_syn['category'] = 'new'

        final_samples.extend((sample_wikt, sample_syn))
        print(f"    > Wiktionary (Established LWs): Found {n_wikt} unique terms")
        print(f"    > Synthetic  (New LWs): Found {n_syn} unique terms")

        if n_syn < target_per_group:
            print(f">> (!) Only found {n_syn} synthetic terms for {lang}, so sample includes all of them")

    if not final_samples:
        print("(!) > Warning: no samples generated")
        return

    df_sample = pd.concat(final_samples)

    df_sample['IS_VALID_LOAN'] = ''
    df_sample['OTHER_LOANS'] = ''
    df_sample['NOTES'] = ''
    df_sample = df_sample[cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_sample.to_csv(OUTPUT_FILE_CSV, index=False, sep=';', encoding='utf-8-sig')

    print(f"\n>>> Split gold standard sample saved to: {OUTPUT_FILE_CSV}")
    print(f">>> Total rows: {len(df_sample)}")

def sample_in_json(csv_path: str = OUTPUT_FILE_CSV, json_path: str = OUTPUT_FILE_JSON, version: str = "v1"):
    # sourcery skip: low-code-quality
    """Converts an already-annotated CSV sample into a structured JSON format."""
    try:
        print(f"> Loading annotations CSV from {csv_path}...")
        df = pd.read_excel(csv_path, sheet_name=f"Annotations ({version})")
    except Exception as e:
        print(f"(!) > Error: Annotations CSV file not found or could not be read: {e}")
        return
    json_data = []

    for idx, row in df.iterrows():
        if pd.isna(row['sentence']): continue
        # main mined term off wikipedia context
        # 'is_valid': 0 (incorrect, proper nouns) or 1 (correct, vlaid loan)
        main_loan = {
            "term": str(row['term']),
            "role": "main",
            "is_valid": (
                row['IS_VALID_LOAN'] == 1.0
                if pd.notna(row['IS_VALID_LOAN'])
                else False
            ),
            "category": (
                str(row['category']) if pd.notna(row['category']) else None
            ),
            "type": str(row['type']) if pd.notna(row['type']) else None,
            "notes": str(row['NOTES']) if pd.notna(row['NOTES']) else "",
        }

        # substructure off main mined term
        loans_list = [main_loan]

        # other loans for mined sentence
        if pd.notna(row['OTHER_LOANS']):
            others = str(row['OTHER_LOANS']).split(',')
            for term in others:
                if term_clean := term.strip():
                    # quick check for potential NEs (uppercase: iPad, App Store)
                    is_NE = term_clean[0].isupper()

                    loans_list.append({
                        "term": term_clean,
                        "role": "secondary",
                        "is_valid": not is_NE, 
                        "category": None,
                        "type": "manual_annotation",
                        "notes": "NE (adversarial)" if is_NE else "in-context loan (manual annotation)"
                    })
        # sentence object containing these mined terms
        entry = {
            "id": idx,
            "lang": row['lang'],
            "sentence": row['sentence'],
            "source": str(row['source_page']) if pd.notna(row['source_page']) else None,
            "loans": loans_list
        }
        json_data.append(entry)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f">>> Annotations converted to JSON format and saved to: {json_path}")


def sample_in_labelstudio(input_path: str = OUTPUT_FILE_JSON, output_path: str = OUTPUT_FILE_LABEL):
    # sourcery skip: low-code-quality
    """Converts an already-annotated JSON sample into a structured LabelStudio format."""
    try:
        print(f"> Loading annotated JSON from {input_path} for Label Studio import...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"(!) > Error: Annotated JSON file not found or could not be read: {e}")
        return

    ls_tasks = []

    # ** import my annotations to label studio to make my life easier **
    for entry in data:
        text = entry['sentence']
        task = {
            "data": {
                "text": text,
                "lang": entry['lang'],
                "source": entry.get('source', '')
            },
            "predictions": [{"model_version": "v1_import", "result": []}]
        }

        for loan in entry['loans']:
            term = loan['term']
            start = text.find(term)

            if start == -1: 
                continue

            # ** INVALID LOANS ** (adversarial examples: proper nouns/NEs, false positives)
            if not loan.get('is_valid', True):
                if 'uppercase' in str(loan.get('notes', '')).lower() or loan['term'][0].isupper():
                    label = "Invalid_NE"
                else:
                    label = "Invalid_FalsePositive"

            elif loan.get('etymology', {}).get('source_lang') in ['es', 'lat']:
                label = "Romance_Cognate"

            elif 'raw' in str(loan.get('type', '')).lower() or 'cs_latin' in str(loan.get('type', '')).lower():
                label = "Raw_Latin_Script" if entry['lang'] == 'el' else "Raw_English"
            else:
                type_str = str(loan.get('type', '')).lower()
                notes = str(loan.get('notes', '')).lower()
                
                if 'light_greek' in type_str:
                    label = "Adapted_LightVerb_Translit" # kanei klik
                elif 'light_latin' in type_str in type_str or ('light_construction' in type_str and entry['lang'] == 'ast'): # facer click, kanei click
                    label = "Adapted_LightVerb_Latin" # kanei click
                elif 'light_construction' in type_str and entry['lang'] == 'eus': # klik egin
                    label = "Adapted_LightVerb_Phono" 

                elif 'transliteration' in notes or entry['lang'] == 'el' and 'noun_transliterated' in type_str:
                    label = "Adapted_Translit" # transliteration adaptation
                elif 'phonological' in notes:
                    label = "Adapted_Phono"   # letter changes
                else:
                    label = "Adapted_Morph"   # default
            
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

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ls_tasks, f, indent=4, ensure_ascii=False)
    print(f">>> Exported {len(ls_tasks)} tasks to {output_path} for Label Studio labeling")
