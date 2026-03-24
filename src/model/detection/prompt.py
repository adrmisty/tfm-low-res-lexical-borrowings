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

def load_gold(filepath: str, num_few_shot: int = 3) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """Loads gold standard data, to use annotations as few-shot examples and test set splits per language."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    data_by_lang = {'ast': [], 'eu': [], 'el': []}
    
    for item in raw_data:
        case_id = item.get("id", str(hash(item["data"]["text"]))) 
        lang = item["data"]["lang"]
        text = item["data"]["text"]
        
        # ** gold standard spans into the JSON format the LLM will use to identify borrowings **
        gold_spans = []
        if "annotations" in item and len(item["annotations"]) > 0:
            for result in item["annotations"].get("result", []):
                val = result.get("value", {})
                if "text" in val and "labels" in val:
                    gold_spans.append({
                        "span": val["text"].strip(),
                        "label": val["labels"]
                    })
        
        item = {
            "id": case_id,
            "text": text,
            "lang": lang,
            "gold_output": json.dumps(gold_spans, ensure_ascii=False), # gold ex
            "raw_annotations": item["annotations"]                     # eval
        }
        
        if lang in data_by_lang:
            data_by_lang[lang].append(item)

    # ** few shot examples and data splits **
    splits = {}
    for lang, items in data_by_lang.items():
        random.seed(42)
        random.shuffle(items)
        
        few_shot = items[:num_few_shot]
        test_set = items[num_few_shot:]
        
        splits[lang] = (few_shot, test_set)
        
    return splits