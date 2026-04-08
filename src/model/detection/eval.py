# eval.py
# ----------------------------------------------------------------------------------------
# LLM output evaluation for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# and plotting of confusion matrices
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import json
import re
from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from typing import List, Dict
import os
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

def evaluate_pipeline(pred_path: str, gold_path: str, out_dir: str, experiment: str):
    """2-step evaluation pipeline."""
    print(f"\n{'='*60}\nEvaluating pipeline: {experiment}\n{pred_path}\n{'='*60}")
    os.makedirs(out_dir, exist_ok=True)
    
    true_spans, pred_spans = _load_spans(pred_path, gold_path)
    
    eval_id(true_spans, pred_spans, out_dir, experiment)
    eval_clf(true_spans, pred_spans, out_dir, experiment)

def eval_id(true_spans: dict, pred_spans: dict, out_dir: str, experiment: str):
    """Evaluates the binary detection task (Native vs. Borrowing)."""
    true_keys = set(true_spans.keys())
    pred_keys = set(pred_spans.keys())
    all_keys = true_keys.union(pred_keys)

    y_true_id, y_pred_id = [], []
    
    for key in all_keys:
        y_true_id.append("Borrowing" if key in true_keys else "Native")
        y_pred_id.append("Borrowing" if key in pred_keys else "Native")

    get_metrics(y_true_id, y_pred_id, labels=["Borrowing"], average="binary")
    _plot_cm(
        y_true=y_true_id, y_pred=y_pred_id, labels=["Native", "Borrowing"],
        title=f"Borrowing span identification\n[{experiment.upper()}]",
        out_path=os.path.join(out_dir, f"{experiment}_step1_identification_cm.png")
    )

def eval_clf(true_spans: dict, pred_spans: dict, out_dir: str, experiment: str):
    """Evaluates the morphological tagging task."""
    # ** only correctly identified spans considered for classification **
    true_positives = set(true_spans.keys()).intersection(set(pred_spans.keys()))
    
    if not true_positives:
        print("\n--- STEP 2: CLASSIFICATION ---")
        print("(!) Cannot calculate Step 2: The model failed to identify any correct spans in Step 1.")
        return

    y_true_clf, y_pred_clf = [], []
    
    for key in true_positives:
        t_lbl = true_spans[key]
        p_lbl = pred_spans[key]
        
        # ** FALLBACK: always raw? **
        if t_lbl not in LABELS: t_lbl = "Raw" 
        if p_lbl not in LABELS: p_lbl = "Raw"
        
        y_true_clf.append(t_lbl)
        y_pred_clf.append(p_lbl)

    get_metrics(y_true_clf, y_pred_clf, labels=LABELS, average="macro", task="CLASSIFICATION")
    _plot_cm(
        y_true=y_true_clf, y_pred=y_pred_clf, labels=LABELS,
        title=f"Borrowing adaptation classification\n[{experiment.upper()}]",
        out_path=os.path.join(out_dir, f"{experiment}_step2_classification_cm.png")
    )
                
def get_metrics(ground_truth: List[dict], predictions: List[dict], labels: list, average: str = "binary", out_file: str = None, task: str = "IDENTIFICATION"):
    """Computes exact match precision, recall, and F1 for lexical borrowing identification.."""

    acc = accuracy_score(ground_truth, predictions)
    p, r, f1, _ = precision_recall_fscore_support(
        ground_truth, 
        predictions, 
        labels=labels, 
        average=average, 
        pos_label="Borrowing" if average == "binary" else None,
        zero_division=0
    )
    
    output_str = f"[LEXICAL BORROWING {task}.] -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}\n"
    print(output_str)
    
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f">>> Evaluation saved to: {out_file}")

    
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    }

# --- aux -------------------------------------------------------------------------

def _load_spans(pred_path: str, gold_path: str) -> Tuple[Dict, Dict]:
    """Loads JSON data and parses text [multi-word] spans into aligned dictionaries."""
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    gt_map = {}
    for item in ground_truth:
        text = item["data"]["text"]
        stable_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        case_id = str(item.get("id", stable_id))
        gt_map[case_id] = item.get("annotations", [])

    true_spans_dict = {} 
    pred_spans_dict = {} 

    for record in predictions:
        case_id = str(record["id"])
        if case_id not in gt_map: continue

        # ** predictions **
        pred_items = _parse_llm_output(record.get("prediction", "[]"))
        if isinstance(pred_items, list):
            for p in pred_items:
                if isinstance(p, dict) and p.get("span") and p.get("label"):
                    txt = _normalize_text(p["span"])
                    lbl = _normalize_label(p["label"]) 
                    if lbl != "O": 
                        pred_spans_dict[(case_id, txt)] = lbl
                        
        # ** true annotations **
        true_items = []
        for ann in gt_map[case_id]:
            if isinstance(ann, dict):
                true_items.extend(ann.get("result", []))
                
        for t in true_items:
            val = t.get("value", {})
            if "text" in val and "labels" in val:
                txt = _normalize_text(val["text"])
                lbl = _normalize_label(val["labels"])
                true_spans_dict[(case_id, txt)] = lbl
                
    return true_spans_dict, pred_spans_dict

def _plot_cm(y_true, y_pred, labels, title, out_path):
    """Plots confusion matrix for given true/pred labels and saves to file."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title, pad=15)
    plt.ylabel('True label (gold standard)', fontweight='bold')
    plt.xlabel('Predicted label (model output)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def _normalize_label(lbl):
    if isinstance(lbl, list) and len(lbl) > 0:
        return lbl[0].strip()
    elif isinstance(lbl, str):
        return lbl.strip()
    return "O"

def _parse_llm_output(prediction_str: str) -> List[Dict]:
    try:
        prediction_str = re.sub(r"<think>.*?</think>", "", str(prediction_str), flags=re.DOTALL).strip()
        match = re.search(r"\[\s*\{.*\}\s*\]", prediction_str, re.DOTALL)
        if match:
            return json.loads(match.group())
        data = json.loads(prediction_str)
        return data if isinstance(data, list) else []
    except Exception:
        return []