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
    "forward", "spam", "check", "spoiler", "pixel", "troll"
]
N_roots = [
    "bug", "server", "cloud", "software", "hardware", 
    "online", "interface", "user", "bot", "app"
]
roots = V_roots + N_roots

transliterations_el = {
        "click": "κλικ",
        "post": "ποστ",
        "chat": "τσατ",
        "link": "λινκ",
        "tag": "ταγκ",
        "tweet": "τουίτ",
        "scan": "σκαν",
        "format": "φορμάτ",
        "hack": "χακ",
        "ban": "μπαν",
        "log": "λογκ",
        "reset": "ρισέτ",
        "download": "νταουνλόουντ",
        "stream": "στριμ",
        "like": "λάικ",
        "scroll": "σκρολ",
        "update": "απντέιτ",
        "forward": "φόργουορντ",
        "spam": "σπαμ",
        "check": "τσεκ",
        "spoiler": "σπόιλερ",
        "pixel": "πίξελ",
        "bug": "μπαγκ",
        "server": "σέρβερ",
        "cloud": "κλάουντ",
        "software": "σόφτγουερ",
        "hardware": "χάρντγουερ",
        "online": "ονλάιν",
        "interface": "ιντερφέις",
        "user": "γιούζερ",
        "bot": "μποτ",
        "app": "απ", # TODO: review results to check if this gets confused with the preposition από, same for others
        "troll": "τρολ"
}

# -------------------------------------------------------------------------------------------------

def synthesize_seeds():
    print(f"\n> [{task}] Starting...")

    all_seeds = []

    print(f"\tGenerating Asturian synthetic seeds...")
    for root in roots:
        all_seeds.extend(_synthesize_AST_seeds(root))

    print(f"\tGenerating Basque synthetic seeds...")
    for root in roots:
        all_seeds.extend(_synthesize_EU_seeds(root))

    print(f"\tGenerating Greek synthetic seeds...")
    for root in roots:
        all_seeds.extend(_synthesize_EL_seeds(root))

    df = pd.DataFrame(all_seeds)
    df_clean = df.drop_duplicates(subset=['term', 'lang'])

    # save as .csv format
    output_file = "data/raw/synthetic_borrowings.csv"
    df_clean.to_csv(output_file, index=False)

    print("\n" + "="*30)
    print(f"[{task}] Done")
    print(f"\t> Total seeds generated: {len(df_clean)}")
    print(df_clean.groupby('lang')['type'].count().to_string())
    print(f"\n\t> Examples (Asturian 'tweet'):\n{df_clean[(df_clean['lemma']=='tweet') & (df_clean['lang']=='ast')]['term'].to_string(index=False)}")
    print(f"\n\t> Examples (Basque 'chat'):\n{df_clean[(df_clean['lemma']=='chat') & (df_clean['lang']=='eu')]['term'].to_string(index=False)}")
    print(f"\n\t> Examples (Greek 'troll'):\n{df_clean[(df_clean['lemma']=='troll') & (df_clean['lang']=='el')]['term'].to_string(index=False)}")
    print("\n" + "="*30)

# -------------------------------------------------------------------------------------------------


def _synthesize_AST_seeds(root, lang=asturian):
    """
    Generates potential Asturian adaptations.

    Hypotheses implemented:
      0. Raw code-switching
      + Phonetic changes: ck->qu, ge/gi->gue/gui, sC->esC, vowel changes e.g. tuit
      1. Nouns (sg., pl.)
      2. Verbs: infinitive (-iar / -ear)
      4. Adjectives (-iáu / -eáu)
    """
    H = []
    
    # ** PHONETIC CHANGES (epenthesis) **    
    # (scan -> escan)
    stem = root
    if root.startswith("s") and len(root) > 1 and root[1] not in "aeiou":
        stem = "e" + stem
    
    # ** PHONETIC CHANGES (vowels) **    
    # TODO: think of other cases where we adapt like in Spanish? also this is in Euskera according to Ane
    if "tweet" in stem:
        stem = stem.replace("tweet", "tuit")

    for r in [stem]:
        # H1: NOUNS (sg., pl.)
        H.append({"term": r, "lemma": root, "lang": lang, "type": "code_switching_sg"})
        if r.endswith(("a", "e", "i", "o", "u")):
            H.append({"term": f"{r}s", "lemma": root, "lang": lang, "type": "plural_s"})
        else:
            H.append({"term": f"{r}es", "lemma": root, "lang": lang, "type": "plural_es"})

        if root in N_roots: continue 

        # H2: VERBS (sg., pl.)
        # drop 'e' for verbs ending in 'e' before adding suffixes
        verb_stem = r
        if verb_stem.endswith("e") and root in ["like", "update", "share"]:
             verb_stem = verb_stem[:-1]

        # ** ORTHOGRAPHY FOR VERBS **
        def adapt_spelling(base_stem, suffix):
            # (log -> logu-iar)
            if base_stem.endswith("g") and suffix[0] in ["e", "i"]:
                return base_stem + "u" + suffix
            
            # CK/K -> QU (click -> cliqu-iar, like -> liqu-iar)
            if base_stem.endswith("ck"):
                if suffix[0] in ["e", "i"]:
                    return base_stem[:-2] + "qu" + suffix
            
            if base_stem.endswith("k"):
                if suffix[0] in ["e", "i"]:
                    return base_stem[:-1] + "qu" + suffix
            return base_stem + suffix

        # (-iar / -ear)
        H.append({"term": adapt_spelling(verb_stem, "iar"), "lemma": root, "lang": lang, "type": "morph_infinitive_iar"}) 
        H.append({"term": adapt_spelling(verb_stem, "ear"), "lemma": root, "lang": lang, "type": "morph_infinitive_ear"}) 
        
        # H3: Participles
        suf_iar = ["iáu", "iau", "iao", "iua", "iada", "ia"]
        for suf in suf_iar: 
            H.append({"term": adapt_spelling(verb_stem, suf), "lemma": root, "lang": lang, "type": "morph_participle_i_base"})

        suf_ear = ["eáu", "eau", "eao", "éu", "eada", "eá"]
        for suf in suf_ear: 
            H.append({"term": adapt_spelling(verb_stem, suf), "lemma": root, "lang": lang, "type": "morph_participle_e_base"})

    return H

def _synthesize_EU_seeds(root, lang=euskera):
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
    
    # ** PHONETIC CHANGES **
    phonetic_stem = root.lower()
    
    # adapt vowels like in ES/AST: tweet -> tuit, wee -> ui
    if "tweet" in phonetic_stem:
        phonetic_stem = phonetic_stem.replace("tweet", "tuit")
    
    # digraphs and usual consonant changes
    phonetic_stem = phonetic_stem.replace("ch", "tx") # chat -> txat
    phonetic_stem = phonetic_stem.replace("sh", "x")  # crash -> krax
    phonetic_stem = phonetic_stem.replace("ck", "k")  # check -> txek
    phonetic_stem = phonetic_stem.replace("c", "k")   # click -> klik
    phonetic_stem = phonetic_stem.replace("q", "k")
    
    # ** epenthesis (scan -> eskan), same strategy **
    if phonetic_stem.startswith("s") and len(phonetic_stem) > 1 and phonetic_stem[1] not in "aeiou":
        phonetic_stem = "e" + phonetic_stem

    # ** NOUN MORPHOLOGY **
    # full stem with some changes
    # H1: raw code-switch
    H.append({"term": root, "lemma": root, "lang": lang, "type": "code_switching_raw"})
    
    # H2: noun
    if phonetic_stem != root:
        H.append({"term": phonetic_stem, "lemma": root, "lang": lang, "type": "phonetic_adaptation_base"})
    
    # H3: definite article singular (-a)
    # if ends in 'a', do not double it (data -> data); if consonant, add 'a' (txat -> txata).
    term_a = phonetic_stem if phonetic_stem.endswith("a") else f"{phonetic_stem}a"
    H.append({"term": term_a, "lemma": root, "lang": lang, "type": "phonetic_adaptation_base_a"})

    # definite article plural  (-ak)
    # if ends in 'a', add 'k' (data -> datak); otherwise add 'ak' (txat -> txatak, etxe -> etxeak).
    term_ak = f"{phonetic_stem}k" if phonetic_stem.endswith("a") else f"{phonetic_stem}ak"
    H.append({"term": term_ak, "lemma": root, "lang": lang, "type": "morph_plural_ak"})

    if root in N_roots:
        return H

    # ** VERB MORPHOLOGY **
    # like in Spanish/Asturian, drop silent 'e' before adding suffixes
    verb_stem = phonetic_stem
    silent_e_candidates = ["like", "update", "share", "save", "mute", "code"]
    
    if root in silent_e_candidates and verb_stem.endswith("e"):
        verb_stem = verb_stem[:-1]

    # connecting vowel: lik -> lik-a-tu, txat -> txat-a-tu.
    connector_verb = "a" if verb_stem[-1] not in "aeiou" else ""

    # H4: Participles
    # batua: -tu
    H.append({"term": f"{verb_stem}{connector_verb}tu", "lemma": root, "lang": lang, "type": "morph_participle_batua"})
    
    # dialectal (Bizkaia/Western): -u / -eu
    # -u replaces -tu. (likatu -> likau)
    H.append({"term": f"{verb_stem}{connector_verb}u", "lemma": root, "lang": lang, "type": "morph_participle_bizkaia_u"}) 
    
    # -eu is common when the connector was 'a' (likatu -> likeu, updateu)
    if connector_verb == "a": 
        H.append({"term": f"{verb_stem}eu", "lemma": root, "lang": lang, "type": "morph_participle_bizkaia_eu"})

    # H5: aspect (-tzen)
    H.append({"term": f"{verb_stem}{connector_verb}tzen", "lemma": root, "lang": lang, "type": "morph_habitual"})
    
    # H6: resultative (-tuta / -tatu)
    H.append({"term": f"{verb_stem}{connector_verb}tuta", "lemma": root, "lang": lang, "type": "morph_resultative_batua"})
    H.append({"term": f"{verb_stem}{connector_verb}tatu", "lemma": root, "lang": lang, "type": "morph_resultative_variant"})

    return H


def _synthesize_EL_seeds(root, lang="el"):
    """
    Generates potential Greek adaptations.
    
    Hypotheses: 
      1. Raw code-switching.
      2. Transliteration into Greek based on common rules.
      3. Verbs: adding productive suffixes, for sg. present and sg. past.
      4. Adjectives: participle with productive suffix.
      5. Nouns: from verbs (act of verb+ing)
    
    Notes: on transliteration, Greeks do literal pronunciation, see these consonant clusters::
      - b  -> μπ (mp)   (blog -> μλόγκ)
      - d  -> ντ (nt)   (digital -> ντίτζιταλ)
      - g  -> γκ (gk)   (game -> γκέιμ)
      - j  -> τζ (tz)   (jogging -> τζόγκινγκ)
      - sh -> σ  (s)    (shopping -> σόπινγκ)
      - th -> θ  (th)   (thriller -> θρίλερ)
      - ch -> τσ (ts)   (check -> τσέκ)
      - f  -> φ         (facebook -> φέισμπουκ)
      - v  -> β         (video -> βίντεο)
      - w  -> ου (ou)   (web -> ουέμπ)
    """
    H = []
    
    # H1: code-switching
    # εγώ κάνω follow -> φόλλου
    H.append({"term": root, "lemma": root, "lang": lang, "type": "code_switching_latin"})
    H.append({"term": root + "s", "lemma": root, "lang": lang, "type": "code_switching_latin_plural"})

    # H2: transliteration of base nouns
    gk_stem = transliterations_el.get(root)
    
    if not gk_stem:
        return H

    # guessing singular/plural is usually the same?
    # το κλικ, τα κλικς
    H.append({"term": gk_stem, "lemma": root, "lang": lang, "type": "transliteration_noun"})
    
    # αλλά μπορεις να βρεις αυτα τα λαικς
    # coloquially, u can add sigma to these plurals: τα likes -> τα λάικς)
    if not gk_stem.endswith("ς"):
        H.append({"term": gk_stem + "ς", "lemma": root, "lang": lang, "type": "transliteration_noun_plural_s"})

    if root in N_roots:
        return H

    # H3: VERBS (add suffix -άρω)
    # move stress to the suffix: κλικ -> κλικάρω
    unstressed_stem = gk_stem.replace("ά", "α").replace("έ", "ε").replace("ί", "ι").replace("ό", "ο").replace("ύ", "υ").replace("ή", "η").replace("ώ", "ω")
    
    # present: εγό ()-άρω (klikárw)
    H.append({"term": f"{unstressed_stem}άρω", "lemma": root, "lang": lang, "type": "morph_verb_aro_present"})
    H.append({"term": f"{unstressed_stem}άρει", "lemma": root, "lang": lang, "type": "morph_verb_aro_3sg"})
    H.append({"term": f"{unstressed_stem}άρεις", "lemma": root, "lang": lang, "type": "morph_verb_aro_2sg"})
    H.append({"term": f"{unstressed_stem}άρουν", "lemma": root, "lang": lang, "type": "morph_verb_aro_3pl"})
    
    # past: -αρα (κλικ[αρα])
    H.append({"term": f"{unstressed_stem}αρα", "lemma": root, "lang": lang, "type": "morph_verb_ara_past"})
    
    # H4: participle -αρισμένος (Clicked κλικαρισμένος)
    H.append({"term": f"{unstressed_stem}αρισμένος", "lemma": root, "lang": lang, "type": "morph_participle_menos_masc"})
    H.append({"term": f"{unstressed_stem}αρισμένη", "lemma": root, "lang": lang, "type": "morph_participle_menos_fem"})
    H.append({"term": f"{unstressed_stem}αρισμένο", "lemma": root, "lang": lang, "type": "morph_participle_menos_neut"})

    # H5: act of -άρισμα (the click, κλικαρισμα, τρολαρισμα)
    H.append({"term": f"{unstressed_stem}άρισμα", "lemma": root, "lang": lang, "type": "morph_noun_arisma"})

    return H

# -------------------------------------------------------------------------------------------------

synthesize_seeds()