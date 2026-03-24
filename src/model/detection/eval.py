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

def eval_borrowings(predictions: List[dict], ground_truth: List[dict], out_file: str = None):
    """Computes exact match precision, recall, and F1 for lexical borrowing identification.."""

    id_tp, id_fp, id_fn = 0, 0, 0
    clf_tp, clf_fp, clf_fn = 0, 0, 0

    gt_map = {item["id"]: item for item in ground_truth}  # or index-based

    for record in predictions:
        case_id = record["id"]
        if case_id not in gt_map:
            continue

        pred_items = _parse_llm_output(record["prediction"])
        
        item = gt_map.get(case_id, {})
        true_items = []

        for ann in item.get("annotations", []):
            true_items.extend(ann.get("result", []))    
                    
        true_set = [
            (_normalize_text(t["value"]["text"]), _normalize_labels(t["value"]["labels"]))
            for t in true_items
        ]

        pred_set = [
            (_normalize_text(p.get("span", "")), _normalize_labels(p.get("label", "")))
            for p in pred_items
        ]

        # identification (span only)
        true_terms = {t[0] for t in true_set}
        pred_terms = {p[0] for p in pred_set}

        # classification (span + label)
        clf_true = set(true_set)
        clf_pred = set(pred_set)
        
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

    output_lines = [
        "--- [LEXICAL BORROWING ID.]: Exact span ---",
        f"Precision: {id_p:.4f} | Recall: {id_r:.4f} | F1: {id_f1:.4f}\n",
        "--- [LEXICAL BORROWING ID.]: Exact span + label ---",
        f"Precision: {clf_p:.4f} | Recall: {clf_r:.4f} | F1: {clf_f1:.4f}\n"
    ]
    
    output_str = "\n".join(output_lines)
    print(output_str)

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f">>> Evaluation saved to: {out_file}")
        
def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def _normalize_labels(labels):
    if isinstance(labels, list):
        return tuple(labels)
    return (labels,)