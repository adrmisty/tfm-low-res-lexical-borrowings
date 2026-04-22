# eval.py
# --------------------------------------------------------------------------------------------------------
# lexical borrowing identification and classification output evaluation and plotting of confusion matrices
# --------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026

import json
import re
from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import os
import hashlib
from .prompt import LABELS

def evaluate_pipeline(pred_path: str, gold_path: str, out_dir: str, experiment: str):
    """Evaluation pipeline generating 3 views."""
    print(f"\n{'='*60}\nEvaluating pipeline: {experiment}\n{pred_path}\n{'='*60}")
    os.makedirs(out_dir, exist_ok=True)
    
    true_spans, pred_spans = _load_spans(pred_path, gold_path)
    
    print("\n*** [LW 1]: Borrowing detection ***")
    eval_step1_id(true_spans, pred_spans, out_dir, experiment)
    
    print("\n*** [LW 2]: Borrowing classification ***")
    eval_step2_clf(true_spans, pred_spans, out_dir, experiment)
    
    print("\n*** [LW 1+2]: JOINT PIPELINE: ID + CLASSIFICATION ***")
    eval_joint(true_spans, pred_spans, out_dir, experiment)


def eval_step1_id(true_spans: dict, pred_spans: dict, out_dir: str, experiment: str):
    """Evaluates the binary detection task (native vs. borrowing)."""
    true_keys = set(true_spans.keys())
    pred_keys = set(pred_spans.keys())
    all_keys = true_keys.union(pred_keys)

    y_true_id, y_pred_id = [], []
    
    for key in all_keys:
        y_true_id.append("Borrowing" if key in true_keys else "Native")
        y_pred_id.append("Borrowing" if key in pred_keys else "Native")

    get_metrics(y_true_id, y_pred_id, labels=["Borrowing"], average="binary", task="IDENTIFICATION")
    
    _plot_cm(
        y_true=y_true_id, y_pred=y_pred_id, labels=["Native", "Borrowing"],
        title=f"Borrowing detection\n[{experiment.upper()}]",
        out_path=os.path.join(out_dir, f"{experiment}_step1_cm.png")
    )

def eval_step2_clf(true_spans: dict, pred_spans: dict, out_dir: str, experiment: str):
    """Evaluates morphological classification ONLY on spans where the boundary was correctly detected."""
    intersection_keys = set(true_spans.keys()).intersection(set(pred_spans.keys()))

    y_true_clf, y_pred_clf = [], []

    for key in intersection_keys:
        t_lbl = true_spans.get(key)
        p_lbl = pred_spans.get(key)

        # if the gold label is 'Invalid_NE' or 'Invalid_FalsePos', it's not a real borrowing.
        # it's a False Positive from the identification step
        if t_lbl not in LABELS:
            continue

        # fallback: hallucinated tags
        if p_lbl not in LABELS: 
            p_lbl = "Raw"

        y_true_clf.append(t_lbl)
        y_pred_clf.append(p_lbl)

    if len(y_true_clf) == 0:
        print(">>> Skipping classification CM: No valid tagset borrowings intersected")
        return

    get_metrics(
        ground_truth=y_true_clf, 
        predictions=y_pred_clf, 
        labels=LABELS, 
        average="macro", 
        task="CLASSIFICATION"
    )

    _plot_cm(
        y_true=y_true_clf, y_pred=y_pred_clf, labels=LABELS,
        title=f"Borrowing classification (Exact matches)\n[{experiment.upper()}]",
        out_path=os.path.join(out_dir, f"{experiment}_step2_cm.png")
    )

def eval_joint(true_spans: dict, pred_spans: dict, out_dir: str, experiment: str):
    """Evaluates joint identification and classification."""
    all_keys = set(true_spans.keys()).union(set(pred_spans.keys()))

    y_true_joint, y_pred_joint = [], []

    for key in all_keys:
        t_lbl = true_spans.get(key, "Native")
        p_lbl = pred_spans.get(key, "Native")

        # gold label is an Invalid_NE/FalsePos noise tag, map it back to Native
        if t_lbl not in LABELS and t_lbl != "Native": 
            t_lbl = "Native" 

        # fallback:hallucinated borrowing tags
        if p_lbl not in LABELS and p_lbl != "Native": 
            p_lbl = "Raw"

        y_true_joint.append(t_lbl)
        y_pred_joint.append(p_lbl)

    get_metrics(
        ground_truth=y_true_joint, 
        predictions=y_pred_joint, 
        labels=LABELS, 
        average="macro", 
        task="JOINT (Macro, ex. Native)"
    )

    labels_with_native = ["Native"] + LABELS
    _plot_cm(
        y_true=y_true_joint, y_pred=y_pred_joint, labels=labels_with_native,
        title=f"Joint borrowing identification & classification\n[{experiment.upper()}]",
        out_path=os.path.join(out_dir, f"{experiment}_joint_cm.png")
    )
                
def get_metrics(ground_truth: List[str], predictions: List[str], labels: list, average: str = "binary", out_file: str = None, task: str = "IDENTIFICATION"):
    acc = accuracy_score(ground_truth, predictions)
    p, r, f1, _ = precision_recall_fscore_support(
        ground_truth, 
        predictions, 
        labels=labels, 
        average=average, 
        pos_label="Borrowing" if average == "binary" else None,
        zero_division=0
    )
    
    output_str = f"[{task}] -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}\n"
    print(output_str)
    
    if out_file:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(output_str)

    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def _load_spans(pred_path: str, gold_path: str) -> Tuple[Dict, Dict]:
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

        pred_items = _parse_llm_output(record.get("prediction", "[]"))
        if isinstance(pred_items, list):
            for p in pred_items:
                if isinstance(p, dict) and p.get("span") and p.get("label"):
                    txt = _normalize_text(p["span"])
                    lbl = _normalize_label(p["label"]) 
                    if lbl != "O": 
                        pred_spans_dict[(case_id, txt)] = lbl
                        
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
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(11, 9))
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
    """Regex to extract clean string tag, stripping lists, quotes, and brackets."""
    if not lbl: return "O"
    lbl_str = str(lbl)
    lbl_str = re.sub(r"[\[\]\'\"]", "", lbl_str)
    return lbl_str.strip() or "O"

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