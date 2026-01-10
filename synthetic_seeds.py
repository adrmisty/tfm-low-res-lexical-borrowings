# synthetic_seeds.py
# ----------------------------------------------------------------
# script to generate hypothetical loanword forms for low-res langs
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import pandas as pd
import os

# basic config.
task = "synthetic-seeds-generation"
os.makedirs("data/raw", exist_ok=True)
euskera = "eu"
asturian = "ast"

# foreign roots
V_roots = [
    "click", "post", "chat", "link", "tag", "tweet", 
    "scan", "format", "hack", "ban", "log", "reset", 
    "download", "stream", "like", "scroll", "update", 
    "forward", "spam", "check", "spoiler", "pixel"
]
N_roots = [
    "bug", "server", "cloud", "software", "hardware", 
    "online", "interface", "user", "bot", "app"
]
roots = V_roots + N_roots

def synthesize_AST_seeds(root, lang=asturian):
    """
    Generates potential Asturian adaptations.
    """
    H = []
    
    # ** PHONETIC CHANGES (epenthesis) **
    # word-initial S_ (scan -> (escan) -> escan)
    stems = set([root])
    if root.startswith("s") and len(root) > 1 and root[1] not in "aeiou":
        stems.add("e" + root)
    
    for r in stems:
        # ** H1: NOUNS / CODE-SWITCHING ** both sg. and pl.
        H.append({"term": r, "lemma": root, "lang": lang, "type": "code_switching_sg"})
        plural_suffix = "es" if r[-1] not in "aeiou" and r[-1] != "s" else "s"
        H.append({"term": f"{r}s", "lemma": root, "lang": lang, "type": "plural_s"})
        if r[-1] not in "aeiou":
             H.append({"term": f"{r}es", "lemma": root, "lang": lang, "type": "plural_es"})

        if root in N_roots: continue # stop for nouns here<

        # H2: Verbs (-iar / -ear)
        H.append({"term": f"{r}iar", "lemma": root, "lang": lang, "type": "morph_infinitive_iar"}) 
        H.append({"term": f"{r}ear", "lemma": root, "lang": lang, "type": "morph_infinitive_ear"}) 
        
        # H3: Participles (-iáu / -éu, dialectal)
        suf_iar = ["iáu", "iau", "iao", "iua", "iada", "ia"]
        for suf in suf_iar: H.append({"term": f"{r}{suf}", "lemma": root, "lang": lang, "type": "morph_participle_i_base"})

        suf_ear = ["eáu", "eau", "eao", "éu", "eada", "eá"]
        for suf in suf_ear: H.append({"term": f"{r}{suf}", "lemma": root, "lang": lang, "type": "morph_participle_e_base"})

    return H

def synthesize_EU_seeds(root, lang=euskera):
    """
    Generates potential Basque adaptations.
    
    Morphological adaptation hypotheses taken from asking L1-Basque speakers,
    covering different dialects (Gatika, Zarautz, Gasteiz, Bilbo, Donosti).
    
    Sources:
    - Ane: Zarautz (Gipuzkoa)
    - Josu: Gatika (Bizkaia)
    - Ander: Gasteiz (Batua)
    - Asier: Lekeitio (Bizkaia)
    - Iker: Bilbo (Batua)
    - Sofia: Donosti (Batua, w/ some Gipuzkoan features)
    
    #TODO: extend hypothesis once friends send me their examples/adaptations list!
    
    Hypotheses implemented:
      1. Raw code-switching
      2. Phonetic shifts: c->k, ch->tx, sC->esC
      3. Verbs: Participle (-tu standard, -eu western)
      4. Aspect: Habitual (-tzen)
      5. Adjectives/Resultative: -tuta / -tatu
      6. Nouns: -a at the end as definite article
    """
    H = []
    
    # ** PHONETIC CHANGES (digraphs, single sounds) **
    phonetic_stem = root.lower()
    
    # digraphs
    phonetic_stem = phonetic_stem.replace("ch", "tx") # chat -> txat
    phonetic_stem = phonetic_stem.replace("sh", "x")  # crash -> krax
    phonetic_stem = phonetic_stem.replace("ck", "k")  # check -> txek, click -> clik
    phonetic_stem = phonetic_stem.replace("c", "k")   # clik -> klik
    phonetic_stem = phonetic_stem.replace("q", "k")
    
    # epenthesis (scan -> eskan)
    if phonetic_stem.startswith("s") and len(phonetic_stem) > 1 and phonetic_stem[1] not in "aeiou":
        phonetic_stem = "e" + phonetic_stem

    # ** MORPH. PRODUCTIVITY **
    
    # H1: raw code switch
    H.append({"term": root, "lemma": root, "lang": lang, "type": "code_switching_raw"})
    
    # H2: phonetic changes to the stem
    if phonetic_stem != root:
        H.append({"term": phonetic_stem, "lemma": root, "lang": lang, "type": "phonetic_adaptation_base"})
        # singular article -a, definite
        H.append({"term": f"{phonetic_stem}a".replace("aa", "a"), "lemma": root, "lang": lang, "type": "phonetic_adaptation_base_a"})

    # H3: plural article -a + k
    connector_noun = "a" if phonetic_stem[-1] not in "aeiou" else "" 
    H.append({"term": f"{phonetic_stem}{connector_noun}ak".replace("aa","a"), "lemma": root, "lang": lang, "type": "morph_plural_ak"})

    if root in N_roots:
        return H

    # H4: Verbs
    #  if ends in consonant, needs connecting vowel 'a' (txat-a-tu)
    connector_verb = "a" if phonetic_stem[-1] not in "aeiou" else ""
    
    # batua
    H.append({"term": f"{phonetic_stem}{connector_verb}tu", "lemma": root, "lang": lang, "type": "morph_participle_batua"})
    # according to Josu and Ane (dialects)
    H.append({"term": f"{phonetic_stem}{connector_verb}u", "lemma": root, "lang": lang, "type": "morph_participle_bizkaia_u"}) 
    if connector_verb == "a": # txat-a-u -> txatau -> reduced to txateu?
        H.append({"term": f"{phonetic_stem}eu", "lemma": root, "lang": lang, "type": "morph_participle_bizkaia_eu"})

    # H5: aspect (-tzen)
    H.append({"term": f"{phonetic_stem}{connector_verb}tzen", "lemma": root, "lang": lang, "type": "morph_habitual"})
    
    # H6: Adjs/Resultative (-tuta / -tatu)
    H.append({"term": f"{phonetic_stem}{connector_verb}tuta", "lemma": root, "lang": lang, "type": "morph_resultative_batua"})
    H.append({"term": f"{phonetic_stem}{connector_verb}tatu", "lemma": root, "lang": lang, "type": "morph_resultative_variant"})

    return H

# -------------------------------------------------------------------------------------------------

print(f"\n> [{task}] Starting...")

all_seeds = []

print(f"\tGenerating Asturian synthetic seeds...")
for root in roots:
    all_seeds.extend(synthesize_AST_seeds(root))

print(f"\tGenerating Basque synthetic seeds (w/ clean phonetic rules)...")
for root in roots:
    all_seeds.extend(synthesize_EU_seeds(root))

df = pd.DataFrame(all_seeds)
df_clean = df.drop_duplicates(subset=['term', 'lang'])

# save as .csv format
output_file = "data/raw/synthetic_borrowings.csv"
df_clean.to_csv(output_file, index=False)

print("\n" + "="*30)
print(f"[{task}] Done")
print(f"\t> Total seeds generated: {len(df_clean)}")
print(df_clean.groupby('lang')['type'].count().to_string())
print(f"\n\t> Examples (Basque 'chat'):\n{df_clean[(df_clean['lemma']=='chat') & (df_clean['lang']=='eu')]['term'].to_string(index=False)}")
print(f"\n\t> Examples (Basque 'click'):\n{df_clean[(df_clean['lemma']=='click') & (df_clean['lang']=='eu')]['term'].to_string(index=False)}")
print(f"\n\t> Examples (Basque 'check'):\n{df_clean[(df_clean['lemma']=='check') & (df_clean['lang']=='eu')]['term'].to_string(index=False)}")
print("\n" + "="*30)
