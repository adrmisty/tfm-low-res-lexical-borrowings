# eval.py
# ----------------------------------------------------------------------------------------
# LLM output evaluation for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import json
import re
from typing import List, Dict

def _parse_llm_output(prediction_str: str) -> List[Dict[str, str]]:
    """Parses JSON output for an LLM prompted to carry out the task of lexical borrowing identification."""
    try:
        prediction_str = re.sub(r"<think>.*?</think>", "", prediction_str, flags=re.DOTALL).strip()
        match = re.search(r"\[.*\]", prediction_str, re.DOTALL)
        return json.loads(match.group()) if match else json.loads(prediction_str)
    except Exception:
        return []

def eval_borrowings(predictions: List[dict], ground_truth: List[dict]):
    """Computes exact match precision, recall, and F1 for lexical borrowing identification."""
    gt_map = {item["id"]: item["annotations"] for item in ground_truth}

    id_tp, id_fp, id_fn = 0, 0, 0
    clf_tp, clf_fp, clf_fn = 0, 0, 0

    for record in predictions:
        case_id = record["id"]
        if case_id not in gt_map: continue

        pred_items = _parse_llm_output(record["prediction"])
        true_items = gt_map[case_id].get("result", []) if gt_map[case_id] and len(gt_map[case_id]) > 0 else []

        true_set = [(_normalize_text(t["value"]["text"]), t["value"]["labels"]) for t in true_items]
        pred_set = [(_normalize_text(p.get("span", "")), p.get("label", "")) for p in pred_items]

        true_terms = set(true_set)
        pred_terms = set(pred_set)

        id_tp += len(true_terms & pred_terms)
        id_fp += len(pred_terms - true_terms)
        id_fn += len(true_terms - pred_terms)

        clf_tp += len(set(true_set) & set(pred_set))
        clf_fp += len(set(pred_set) - set(true_set))
        clf_fn += len(set(true_set) - set(pred_set))
        
    def calc_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    id_p, id_r, id_f1 = calc_f1(id_tp, id_fp, id_fn)
    clf_p, clf_r, clf_f1 = calc_f1(clf_tp, clf_fp, clf_fn)

    print("--- [LEXICAL BORROWING ID.]: Exact span ---")
    print(f"Precision: {id_p:.4f} | Recall: {id_r:.4f} | F1: {id_f1:.4f}")

    print(f"\n--- [LEXICAL BORROWING ID.]: Exact span + label ---")
    print(f"Precision: {clf_p:.4f} | Recall: {clf_r:.4f} | F1: {clf_f1:.4f}")

def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())
