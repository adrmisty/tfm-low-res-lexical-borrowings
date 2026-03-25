# prompts.py
# ----------------------------------------------------------
# configurations for loanword identification & classification
# ----------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import json
import random
from typing import List, Dict, Tuple

def get_system_prompt(language: str) -> str:
    return (
        f"You are an expert computational linguist analyzing text in {language.upper()}. "
        "Identify all lexical borrowings and historical loans in the text. "
        "Classify each into ONE of the following tags: "
        "['Internationalism', 'Raw', 'Adapted_Orthogra', 'Adapted_Morph', 'Adapted_Translit', 'LightVerb_Unintegrated', 'LightVerb_Integrated']. "
        "Respond STRICTLY with a JSON array of objects, where each object has 'span' (the exact text span) and 'label' (the tag). "
        "If no borrowings exist, return []."
    )

def get_fewshot_prompt(text: str, examples: list = None) -> str:
    prompt = "WARNING: Only generate the required JSON output, no explanations or thinking process.\n\n"
    if examples:
        prompt += "--- EXAMPLES start ---\n"
        for ex in examples:
            prompt += f"Text:\n{ex['text']}\n"
            prompt += f"Output:\n{ex['output']}\n\n"
        prompt += "--- EXAMPLES end ---\n\n"
    
    prompt += f"Text to analyze:\n{text}\n\nOutput:"
    return prompt

def load_gold(filepath: str, num_few_shot: int = 3, verbose: bool = True) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """Loads gold standard data, to use annotations as few-shot examples and test set splits per language."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    data_by_lang = {'ast': [], 'eu': [], 'el': []}
    
    # ** target tags as per Label Studio xml **
    target_tags = [
        'Internationalism', 'Raw', 'Adapted_Orthogra', 
        'Adapted_Morph', 'Adapted_Translit', 
        'LightVerb_Unintegrated', 'LightVerb_Integrated'
    ]
    
    for item in raw_data:
        case_id = item.get("id", str(hash(item["data"]["text"]))) 
        lang = item["data"]["lang"]
        text = item["data"]["text"]
        
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
        
        found_tags = {tag: False for tag in target_tags}
        
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