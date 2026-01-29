# goldsampler.py
# ----------------------------------------------------------------
# samples random sentences from mined to generate a gold std. sample
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026
import pandas as pd
import json
import os

INPUT_FILE = "data/processed/mined_sentences.clean.jsonl"
OUTPUT_DIR = "data/annotation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gold_standard_sample.csv")
SAMPLE_SIZE = 100

# columns
cols = ['lang', 'term', 'sentence', 'IS_VALID_LOAN', 'OTHER_LOANS', 'NOTES', 'type', 'source_page']

def get_unannotated_gold_sample(n=SAMPLE_SIZE):
    """Generates gold sample to-be-annotated of n random sentences."""
    if not os.path.exists(INPUT_FILE):
        print("(!) > Error: cleaned mined sentences file not found")
        return

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
            print(f"(!) Warning: No data found for {lang}")
            continue
                
        unique_terms_df = subset.sample(frac=1, random_state=42).drop_duplicates(subset=['term'])
            
        n = min(SAMPLE_SIZE, len(unique_terms_df))
        sample = unique_terms_df.sample(n=n, random_state=42)
            
        final_samples.append(sample)    
        print(f"    > [{lang}] {n} unique terms (from {len(unique_terms_df)} available)")
    
    df_sample = pd.concat(final_samples)
    
    df_sample['IS_VALID_LOAN'] = ''   
    df_sample['OTHER_LOANS'] = ''     
    df_sample['NOTES'] = ''        
    
    df_sample = df_sample[cols]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_sample.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
    
    print(f">>> Gold standard sample created: {OUTPUT_FILE}")
    print(f"\tTotal rows: {len(df_sample)} [{n} per language]")

get_unannotated_gold_sample()