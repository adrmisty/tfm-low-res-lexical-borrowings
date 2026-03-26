# eval.py
# ----------------------------------------------------------------------------------------
# LLM output evaluation for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# and plotting of confusion matrices
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

import json
import re
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import List, Dict
import os
import hashlib

LABELS = [
    "O", # outside, false pos/neg
    "Internationalism", 
    "Raw", 
    "Adapted_Orthogra", 
    "Adapted_Morph", 
    "Adapted_Translit", 
    "LightVerb_Unintegrated", 
    "LightVerb_Integrated"
]

def get_confusion_matrix(pred_path: str, gold_path: str, output_img: str, experiment: str):
    """Generates and plot confusion matrix for prediction comparison against gold data."""
    
    print(f"\n> Generating confusion matrix for: {pred_path} [{experiment}]")
    
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # ** GROUND TRUTH IDs ** with hashlib
    # same hashing mech as gold data loading
    gt_map = {}
    for item in ground_truth:
        text = item["data"]["text"]
        stable_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        case_id = str(item.get("id", stable_id))
        gt_map[case_id] = item.get("annotations", [])

    # ** LABELS **
    y_true = []
    y_pred = []
    debug_true_positives = 0

    for record in predictions:
        case_id = str(record["id"])
        if case_id not in gt_map:
            continue

        # predicted/model-generated annotations
        pred_items = _parse_llm_output(record.get("prediction", "[]"))
        pred_dict = {}
        for p in pred_items:
            if p.get("span") and p.get("label"):
                txt = _normalize_text(p["span"])
                lbl = _extract_label(p["label"])
                pred_dict[txt] = lbl

        # manual gold annotations
        true_items = []
        for ann in gt_map[case_id]:
            if isinstance(ann, dict):
                true_items.extend(ann.get("result", []))
        true_dict = {}
        for t in true_items:
            val = t.get("value", {})
            if "text" in val and "labels" in val:
                txt = _normalize_text(val["text"])
                lbl = _extract_label(val["labels"])
                true_dict[txt] = lbl

        # print("TRUE:", true_dict)
        # print("PRED:", pred_dict)

        # 5. Align spans
        all_spans = set(true_dict.keys()).union(set(pred_dict.keys()))
        for span in all_spans:
            # all invalid/unmatched labels collapse to O
            t_label = true_dict.get(span, "O")
            p_label = pred_dict.get(span, "O")

            if t_label not in LABELS:
                t_label = "O"
            if p_label not in LABELS:
                p_label = "O"
            y_true.append(t_label)
            y_pred.append(p_label)

            if t_label != "O":
                debug_true_positives += 1

    print(f"\t>[*] Debug: {debug_true_positives} valid labeled spans")

    if not y_true:
        print("> (!) Warning: No overlapping spans")
        return

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABELS,
        yticklabels=LABELS
    )

    plt.title(f"Span-level confusion matrix\n{os.path.basename(pred_path)}\n[{experiment.upper()}]")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)

    print(f">>> CM saved to {output_img}\n")
            
def get_metrics(predictions: List[dict], ground_truth: List[dict], out_file: str = None):
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

# --- aux -------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def _normalize_labels(labels):
    if isinstance(labels, list):
        return tuple(labels)
    return (labels,)

def _extract_label(lbl):
    if isinstance(lbl, list) and len(lbl) > 0:
        return lbl[0].strip()
    elif isinstance(lbl, str):
        return lbl.strip()
    return "O"

def _parse_llm_output(prediction_str: str) -> List[Dict[str, str]]:
    try:
        prediction_str = re.sub(r"<think>.*?</think>", "", prediction_str, flags=re.DOTALL).strip()
        match = re.search(r"\[.*\]", prediction_str, re.DOTALL)
        return json.loads(match.group()) if match else json.loads(prediction_str)
    except Exception:
        return []