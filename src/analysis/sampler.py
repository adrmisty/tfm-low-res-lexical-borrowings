# goldsampler.py
# ----------------------------------------------------------------
# samples random sentences from mined to generate a gold std. sample
# 50/50 split (wikt/synthetic) to ensure equal representation
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# feb-2026
import pandas as pd
import json
import os

INPUT_FILE = "data/processed/mined_sentences.clean.jsonl"
OUTPUT_DIR = "data/annotation"
#OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gold_standard_sample.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gold_standard_split.csv")
SAMPLE_SIZE = 100

# columns
# new category: old/new
cols = ['lang', 'term', 'sentence', 'IS_VALID_LOAN', 'OTHER_LOANS', 'NOTES', 'category', 'type', 'source_page']

def get_unannotated_gold_sample(n=SAMPLE_SIZE, split=0.5):
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
        
        final_samples.append(sample_wikt)
        final_samples.append(sample_syn)
        
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
    df_sample.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
    
    print(f"\n>>> Split gold standard sample saved to: {OUTPUT_FILE}")
    print(f">>> Total rows: {len(df_sample)}")
    