# prompt.py
# ----------------------------------------------------------
# configurations for loanword identification & classification
# ----------------------------------------------------------
# adriana r.f. (@adrmisty)
# may-2026

import json
import os
import hashlib
from typing import List, Dict

FEW_SHOT_PATH = "data/icl/few_shot_examples.json"

TAGSET = [ 
    "Raw", 
    "Adapted_Orthogra", 
    "Adapted_Morph", 
    "LightVerb_Unintegrated", 
    "LightVerb_Integrated"
]

TAGSET_DEF = """--- TAGSET ---
1. "Raw": Unassimilated borrowings that retain their exact original foreign spelling and morphology without any adaptation.
2. "Adapted_Orthogra": Borrowings adapted to the target language's spelling or phonological rules, but without native morphological inflection.
3. "Adapted_Morph": Borrowings that have been fully integrated by taking on native suffixes, prefixes, plural markers, or grammatical gender.
4. "LightVerb_Unintegrated": A multi-word construction pairing a native verb with a completely raw, unassimilated foreign loanword.
5. "LightVerb_Integrated": A multi-word construction pairing a native verb with a foreign loanword that has undergone orthographic or morphological adaptation."""

def load_gold_data(filepath: str, target_langs: list = None) -> List[Dict]:
    """Loads the gold standard data as a test set."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    test_set = []
    for item in raw_data:
        lang = item["data"]["lang"]
        if target_langs and lang not in target_langs:
            continue
            
        text = item["data"]["text"]
        stable_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        test_set.append({
            "id": str(item.get("id", stable_id)),
            "text": text,
            "lang": lang,
            "raw_annotations": item.get("annotations", [])
        })
    
    return test_set

def _get_few_shots(lang: str, k: int, examples_path: str = FEW_SHOT_PATH) -> list:
    """Loads k examples for a specific language from the external JSON."""
    if k == 0 or not os.path.exists(examples_path):
        return []
    with open(examples_path, "r", encoding="utf-8") as f:
        all_examples = json.load(f)
    return all_examples.get(lang.lower(), [])[:k]


# --- 1-step pipeline (identification + classification in one prompt) ---

def get_system_prompt_1step(language: str) -> str:
    SYSTEM_PROMPT = """You are an expert computational linguist analyzing text in {language}.
    Your task is to identify lexical borrowings and classify their adaptation.
    
    CRITICAL RULE: It is a FATAL ERROR to extract standard native {language} vocabulary.
    
    STRICT INSTRUCTIONS:
    1. Extract ONLY lexical borrowings, technical loanwords, or foreign-origin lexical items.
    2. DO NOT extract native {language} verbs, adjectives, prepositions, or basic nouns.
    3. DO NOT extract named entities, brands, organizations, or locations.
    4. If unsure if a word is a loanword, DO NOT include the span.
    5. Spans may contain multiple words for light verb constructions.
    
    OUTPUT FORMAT:
    - Output ONLY a valid JSON list.
    - Each item must contain EXACTLY:
        - "span" 
        - "label"
        
        Allowed labels:
        {TAGSET_DEF}
        
        Example output: [{{"span": "router", "label": "Raw"}}]
        If no borrowings are found, output EXACTLY: []
        
        DO NOT explain. DO NOT output reasoning. DO NOT output "Thinking/Output Process". DO NOT use markdown."""
    
    return SYSTEM_PROMPT.format(
        language=language.upper(),
        TAGSET_DEF=TAGSET_DEF
    )

def get_fewshot_prompt_1step(system_prompt: str, text: str, lang: str, k: int) -> str:
    prompt = "" 
    examples = _get_few_shots(lang, k)
    if examples:
        prompt += "--- EXAMPLES start:\n"
        for ex in examples:
            formatted_output = []
            for span_data in ex['spans']:
                formatted_output.append({
                    "span": span_data['span'],
                    "label": span_data['label']
                })
            prompt += f"Text:\n{ex['text']}\nOutput:\n{json.dumps(formatted_output, ensure_ascii=False)}\n\n"
        prompt += "EXAMPLES end ---\n\n"
    
    prompt += f"Text to analyze:\n{text}\n\nOutput:\n"
    return prompt

# --- 2-step pipeline (identification + classification in 2 prompts) ---

def get_system_prompt_id(language: str) -> str:
    SYSTEM_PROMPT_ID = """You are an expert computational linguist analyzing text in {language}.
    Your task is to identify lexical borrowings ONLY.
    
    CRITICAL RULE: It is a FATAL ERROR to extract standard native {language} vocabulary.
    
    STRICT INSTRUCTIONS:
    1. Extract ONLY lexical borrowings, technical loanwords, or foreign-origin items.
    2. DO NOT extract native {language} words (e.g., standard verbs, prepositions, numbers).
    3. DO NOT extract named entities, person names, locations, organizations, or brands.
    4. DO NOT extract metalinguistic mentions or quoted foreign text.
    5. If unsure if a word is foreign, do NOT include the span.
    6. Typical number of spans: 0-5.
    
    OUTPUT FORMAT:
    - Output ONLY a valid JSON list of strings.
    - Example: ["software", "router"]
    - If no borrowings are found, output EXACTLY: []
    
    DO NOT explain. DO NOT output reasoning. DO NOT output "Thinking Process". DO NOT use markdown. The output MUST end with "]"."""
    
    return SYSTEM_PROMPT_ID.format(language=language.upper())

def get_fewshot_prompt_id(system_prompt: str, text: str, lang: str, k: int) -> str:
    prompt = "" 
    examples = _get_few_shots(lang, k)
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            prompt += f"Text:\n{ex['text']}\nOutput:\n{ex['output_id']}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Text to analyze:\n{text}\n\nOutput:"
    return prompt


def get_system_prompt_clf(language: str) -> str:
    SYSTEM_PROMPT_CLF = """You are an expert computational linguist analyzing text in {language}.
    Your task is to classify the morphological adaptation of ONE borrowing.
    
    LABEL DEFINITIONS:
    {TAGSET_DEF}
    
    STRICT INSTRUCTIONS:
    1. Choose EXACTLY ONE label.
    2. Output ONLY a valid JSON object. Use EXACTLY this schema: {{"label": "Raw"}}
    
    DO NOT explain.
    DO NOT output reasoning.
    DO NOT output "Thinking Process".
    DO NOT use markdown."""
    
    return SYSTEM_PROMPT_CLF.format(
        language=language.upper(),
        TAGSET_DEF=TAGSET_DEF
    )
    
def get_fewshot_prompt_clf(system_prompt: str, text: str, target_span: str, lang: str, k: int) -> str:
    prompt = "" 
    examples = _get_few_shots(lang, k)
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            for span_data in ex['spans']:
                prompt += f"Context:\n{ex['text']}\nTarget word:\n{span_data['span']}\nOutput:\n{{\"label\": \"{span_data['label']}\"}}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Context:\n{text}\nTarget word:\n{target_span}\n\nOutput:\n"
    return prompt