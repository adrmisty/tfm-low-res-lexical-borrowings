# prompts.py
# ----------------------------------------------------------
# configurations for loanword identification & classification
# ----------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import json
from typing import List, Dict, Tuple
import hashlib

LABELS = [
    "Internationalism", 
    "Raw", 
    "Adapted_Orthogra", 
    "Adapted_Morph", 
    "Adapted_Translit", 
    "LightVerb_Unintegrated", 
    "LightVerb_Integrated"
]

# ** 1) borrowing identification **

def get_system_prompt_id(language: str) -> str:
    """System prompt for inference[1]: borrowing identification at the span level."""
    return (
        f"""You are an expert computational linguist analyzing text in {language.upper()}. 
        Your task is exclusively to IDENTIFY lexical borrowings (loanwords) in the provided text.

        You must evaluate the text and extract ALL loanwords. Native vocabulary MUST NOT be extracted. 
        
        --- OUTPUT ---
        You must output a raw JSON list of strings, where each string is the exact text span of a borrowing. 
        Example: ["click", "software"]
        If there are no borrowings in the text, output an empty list: []
        Do not wrap the JSON in markdown blocks. Do not explain your reasoning.
        """
    )

def get_fewshot_prompt_id(system_prompt: str, text: str, examples: list = None) -> str:
    """Few-shot prompt for in-context-learning inference[1]: borrowing identification at the span level."""
    prompt = system_prompt + "\n\n"
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            gold_dicts = json.loads(ex['gold_output'])
            gold_spans = [d["span"] for d in gold_dicts]
            
            prompt += f"Text:\n{ex['text']}\n"
            prompt += f"Output:\n{json.dumps(gold_spans, ensure_ascii=False)}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Text to analyze:\n{text}\n\nOutput:"
    return prompt

def get_fewshot_prompt_id(system_prompt: str, text: str, examples: list = None) -> str:
    """Few-shot formatting for identification."""
    prompt = system_prompt + "\n\n"
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            try:
                gold_dicts = json.loads(ex.get('output', '[]'))
                gold_spans = [d.get("span", "") for d in gold_dicts]
            except Exception:
                gold_spans = []
            
            prompt += f"Text:\n{ex['text']}\n"
            prompt += f"Output:\n{json.dumps(gold_spans, ensure_ascii=False)}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Text to analyze:\n{text}\n\nOutput:"
    return prompt


# ** 2) borrowing classification **

def get_system_prompt_clf(language: str) -> str:
    """System prompt for inference[2]: borrowing classification at the span level."""
    return (
        f"""You are an expert computational linguist analyzing text in {language.upper()}. 
        Your task is to classify the morphological adaptation of a specific target loanword found within a context sentence.

        You must classify the target word using STRICTLY one of the following tags:

        --- TAGSET ---
        1. "Raw": Unassimilated borrowings that retain their exact original foreign spelling and morphology without any adaptation.
        2. "Adapted_Orthogra": Borrowings adapted to the target language's spelling or phonological rules, but lacking native morphological inflection.
        3. "Adapted_Morph": Borrowings that have been fully integrated by taking on native suffixes, prefixes, plural markers, or grammatical gender.
        4. "Adapted_Translit": Borrowings that have been transliterated into a different alphabet to match the target language's script.
        5. "LightVerb_Unintegrated": A multi-word construction pairing a native verb with a completely raw, unassimilated foreign loanword.
        6. "LightVerb_Integrated": A multi-word construction pairing a native verb with a foreign loanword that has undergone orthographic or morphological adaptation.
        7. "Internationalism": Widely recognized global vocabulary with shared Greco-Latin roots, deeply integrated into the language's core lexicon.

        --- OUTPUT ---
        Output ONLY the exact tag name from the list above. Do not output anything else. Do not explain your reasoning.
        """
    )


def get_fewshot_prompt_clf(system_prompt: str, text: str, target_span: str, examples: list = None) -> str:
    """Few-shot formatting for classification."""
    prompt = system_prompt + "\n\n"
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            try:
                gold_dicts = json.loads(ex.get('output', '[]'))
            except Exception:
                gold_dicts = []
            for gd in gold_dicts:
                prompt += f"Context:\n{ex['text']}\n"
                prompt += f"Target word:\n{gd.get('span', '')}\n"
                
                # Handle list-type labels from Label Studio
                lbl = gd.get('label', 'Raw')
                if isinstance(lbl, list) and len(lbl) > 0:
                    lbl = lbl
                
                prompt += f"Output:\n{lbl}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Context:\n{text}\nTarget word:\n{target_span}\n\nOutput:\n"
    return prompt


# ** loading gold data **

def load_gold(filepath: str, verbose: bool = True) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """Loads gold standard data, to use annotations as few-shot examples and test set splits per language."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    data_by_lang = {'ast': [], 'eu': [], 'el': []}
    
    # ** target tags as per Label Studio xml > LABELS **
    
    for item in raw_data:
        text = item["data"]["text"]
        stable_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        case_id = str(item.get("id", stable_id))
        lang = item["data"]["lang"]

        gold_spans = []
        if "annotations" in item and len(item["annotations"]) > 0:
            for result in item["annotations"][0].get("result", []):
                val = result.get("value", {})
                if "text" in val and "labels" in val:
                    gold_spans.append({
                        "span": val["text"].strip(),
                        "label": val["labels"]
                    })
        
        processed_item = {
            "id": case_id,
            "text": text,
            "lang": lang,
            "gold_output": json.dumps(gold_spans, ensure_ascii=False),
            "raw_annotations": item["annotations"] 
        }
        
        if lang in data_by_lang:
            data_by_lang[lang].append(processed_item)

    # ** enforce one example of each class if any **
    splits = {}
    for lang, items in data_by_lang.items():
        import random
        random.seed(42)
        random.shuffle(items)
        
        few_shot = []
        test_set = []
        
        found_tags = {tag: False for tag in LABELS}
        
        for item in items:
            item_tags = set()

            if item["raw_annotations"]:
                for result in item["raw_annotations"][0].get("result", []):
                    labels = result["value"].get("labels", [])
                    item_tags.update(labels)    
                     
            provides_new_tag = False
            for tag in item_tags:
                if tag in found_tags and not found_tags[tag]:
                    found_tags[tag] = True
                    provides_new_tag = True         
                       
            if provides_new_tag:
                few_shot.append(item)
            else:
                test_set.append(item)
                
        splits[lang] = (few_shot, test_set)
        if verbose:
            print(f"[{lang.upper()}] Few-shot prompt created with {len(few_shot)} examples covering: {[t for t, v in found_tags.items() if v]}")
        
    return splits

# ** single prompt for both tasks **

@DeprecationWarning
def get_system_prompt(language: str) -> str:
    """Crafts the system prompt for the LLM, with instructions and tagset definitions."""
    return (
        f"""You are an expert computational linguist analyzing text in {language.upper()}. 
        Your task is to identify lexical borrowings (loanwords) in the provided text and classify their morphological adaptation into the target language.

        You must evaluate the text and extract ALL loanwords. Native vocabulary MUST NOT be extracted. 
        For every loanword you find, you must classify it using STRICTLY one of the following tags:

        --- TAGSET ---
        1. "Raw": Unassimilated borrowings that retain their exact original foreign spelling and morphology without any adaptation.
        2. "Adapted_Orthogra": Borrowings adapted to the target language's spelling or phonological rules, but lacking native morphological inflection.
        3. "Adapted_Morph": Borrowings that have been fully integrated by taking on native suffixes, prefixes, plural markers, or grammatical gender.
        4. "Adapted_Translit": Borrowings that have been transliterated into a different alphabet to match the target language's script.
        5. "LightVerb_Unintegrated": A multi-word construction pairing a native verb with a completely raw, unassimilated foreign loanword.
        6. "LightVerb_Integrated": A multi-word construction pairing a native verb with a foreign loanword that has undergone orthographic or morphological adaptation.
        7. "Internationalism": Widely recognized global vocabulary with shared Greco-Latin roots, deeply integrated into the language's core lexicon.

        --- OUTPUT ---
        You must output a raw JSON list of dictionaries. Each dictionary must contain exactly two keys: "span" (the exact text of the borrowing) and "label" (one of the 7 taxonomy tags above). 
        Do not wrap the JSON in markdown blocks. Do not explain your reasoning.
        """
    )


@DeprecationWarning
def get_fewshot_prompt(prompt: str, text: str, examples: list = None) -> str:
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            prompt += f"Text:\n{ex['text']}\n"
            prompt += f"Output:\n{ex['output']}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Text to analyze:\n{text}\n\nOutput:"
    return prompt
