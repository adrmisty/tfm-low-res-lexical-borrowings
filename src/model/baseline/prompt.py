# prompt.py
# ----------------------------------------------------------
# configurations for loanword identification & classification
# ----------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import json
import os
import hashlib
from typing import List, Dict

FEW_SHOT_PATH = "data/icl/few_shot_examples.json"

TAGSET = [ # eliminated "Internationalism" // "Adapted_Translit" merged onto "Adapted_Orthogra" // excl. Invalid_*
    "Raw", 
    "Adapted_Orthogra", 
    "Adapted_Morph", 
    #"Adapted_Translit", 
    "LightVerb_Unintegrated", 
    "LightVerb_Integrated",
    #"Internationalism",
    #"Invalid_NE",
    #"Invalid_FalsePos"
]

TAGSET_DEF = """--- TAGSET ---
1. "Raw": Unassimilated borrowings that retain their exact original foreign spelling and morphology without any adaptation.
2. "Adapted_Orthogra": Borrowings adapted to the target language's spelling or phonological rules, but without native morphological inflection.
3. "Adapted_Morph": Borrowings that have been fully integrated by taking on native suffixes, prefixes, plural markers, or grammatical gender.
4. "LightVerb_Unintegrated": A multi-word construction pairing a native verb with a completely raw, unassimilated foreign loanword.
5. "LightVerb_Integrated": A multi-word construction pairing a native verb with a foreign loanword that has undergone orthographic or morphological adaptation.
6. "Invalid_NE": Proper nouns, corporate brands, geographical names, or specific entities that are not general lexical borrowings.
7. "Invalid_FalsePos": Native homonyms, metalinguistic explanations, or raw English strings that are not actually functioning as borrowings in the sentence."""


def load_gold_data(filepath: str, target_langs: list = None, few_shot=False) -> List[Dict]:
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
    
    if few_shot:
        return _get_few_shots
    return test_set


# --- in-context-learning: k few-shot examples ---
# stored in external .json
# test with different k to see how it affects performance


def _get_few_shots(lang: str, k: int, examples_path: str = FEW_SHOT_PATH) -> list:
    """Loads k examples for a specific language from the external JSON."""
    if k == 0 or not os.path.exists(examples_path):
        return []
    with open(examples_path, "r", encoding="utf-8") as f:
        all_examples = json.load(f)
    return all_examples.get(lang.lower(), [])[:k]


# --- 1-step pipeline (identification + classification in one prompt) ---

def get_system_prompt_1step(language: str) -> str:
    """Returns the system prompt for the 1-step pipeline, which combines identification and classification in a single prompt."""
    
    SYSTEM_PROMPT = """You are an expert computational linguist analyzing text in {language}. 
    Your task is to identify lexical borrowings (loanwords) in the provided text and classify their morphological adaptation into the target language.
    You must evaluate the text and extract ALL loanwords, as well as any tricky entities or false candidates. 
    For every span you extract, you must classify it using STRICTLY one of the following tags:
    {TAGSET_DEF}
    
    You must output a raw JSON list of dictionaries. 
    Each dictionary must contain exactly three keys: "span" (the exact text), "reasoning" (your step-by-step analysis), and "label" (STRICTLY one of the 8 taxonomy tags above). 
    Example: [{{"span": "click", "reasoning": "Borrowed from English...", "label": "Raw"}}]
    
    Note: Even if the examples below omit the 'reasoning' key, YOUR final output MUST include it.
    Do not wrap the JSON in markdown blocks."""
    return SYSTEM_PROMPT.format(language=language, TAGSET_DEF=TAGSET_DEF)

def get_fewshot_prompt_1step(system_prompt: str, text: str, lang: str, k: int) -> str:
    """Builds the few-shot prompt for the 1-step pipeline, which includes k examples of combined identification and classification."""
    prompt = system_prompt + "\n\n"
    examples = _get_few_shots(lang, k)
    if examples:
        prompt += "--- EXAMPLES start:\n"
        for ex in examples:
            prompt += f"Text:\n{ex['text']}\nOutput:\n{ex['output_1step']}\n\n"
        prompt += "EXAMPLES end ---\n\n"
    
    prompt += f"Text to analyze:\n{text}\n\nOutput:"
    return prompt


# --- 2-step pipeline (identification + classification in 2 prompts) ---

def get_system_prompt_id(language: str) -> str:
    """Returns the system prompt for the 2-step pipeline, which focuses only on identification in the first step."""
    
    SYSTEM_PROMPT_ID = """You are an expert computational linguist analyzing text in {language}. 
    Your task is exclusively to IDENTIFY lexical borrowings (loanwords) in the provided text.
    You must evaluate the text and extract ALL loanwords. You should also extract tricky proper nouns or brand names so they can be filtered later. Native vocabulary MUST NOT be extracted. 
    
    You must output a raw JSON list of strings, where each string is the exact text span of a borrowing or entity. 
    Example: ["click", "software", "Microsoft"]
    If there are no borrowings in the text, output an empty list: []
    Do not output thinking processes, explanations, or markdown blocks. Output ONLY the raw JSON list."""
    
    return SYSTEM_PROMPT_ID.format(language=language.upper())

def get_fewshot_prompt_id(system_prompt: str, text: str, lang: str, k: int) -> str:
    """Builds the few-shot prompt for the identification step of the 2-step pipeline, which includes k examples of identification only."""
    prompt = system_prompt + "\n\n"
    examples = _get_few_shots(lang, k)
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            prompt += f"Text:\n{ex['text']}\nOutput:\n{ex['output_id']}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Text to analyze:\n{text}\n\nOutput:"
    return prompt


def get_system_prompt_clf(language: str) -> str:
    """Returns the system prompt for the classification step of the 2-step pipeline, which focuses only on classification of a given target span."""
    SYSTEM_PROMPT_CLF = """You are an expert computational linguist analyzing text in {language}. 
    Your task is to classify the morphological adaptation of a specific target loanword found within a context sentence.
    You must classify the target word using STRICTLY one of the following tags:
    {TAGSET_DEF}
    
    You must output a raw JSON dictionary with exactly two keys: "reasoning" (your step-by-step analysis) and "label" (STRICTLY the exact tag name from the list above).
    Example: {{"reasoning": "The word takes the native plural suffix...", "label": "Adapted_Morph"}}
    Do not wrap the JSON in markdown blocks."""
    return SYSTEM_PROMPT_CLF.format(language=language.upper(), TAGSET_DEF=TAGSET_DEF)

def get_fewshot_prompt_clf(system_prompt: str, text: str, target_span: str, lang: str, k: int) -> str:
    """Builds the few-shot prompt for the classification step of the 2-step pipeline, which includes k examples of classification only."""
    prompt = system_prompt + "\n\n"
    examples = _get_few_shots(lang, k)
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            for span_data in ex['spans']:
                prompt += f"Context:\n{ex['text']}\nTarget word:\n{span_data['span']}\nOutput:\n{{\"reasoning\": \"Matches the {span_data['label']} criteria.\", \"label\": \"{span_data['label']}\"}}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Context:\n{text}\nTarget word:\n{target_span}\n\nOutput:\n"
    return prompt