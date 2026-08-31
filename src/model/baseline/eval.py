# eval.py
# ----------------------------------------------------------------------------
# standardized plotting and statistics generation
# evaluates predictions and corpus data, and generates all thesis figures with 
# standardized aesthetics (Large fonts, blues)
# -----------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# aug-2026

import os
import re
import json
import glob
import hashlib
import logging
from typing import List, Dict, Tuple

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib

matplotlib.use('Agg')
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. AESTHETICS & THEME SETTINGS
# ==========================================
FONT_TITLE = 28
FONT_LABEL = 24
FONT_TICK = 20
FONT_LEGEND = 20
FONT_CM_ANNOTATION = 32  # Large numbers inside the CM boxes
FONT_CM_TICK = 24        # Large tags on X/Y axes of CM

sns.set_theme(style="whitegrid")
plt.rc('font', size=FONT_TICK)
plt.rc('axes', titlesize=FONT_TITLE, labelsize=FONT_LABEL)
plt.rc('xtick', labelsize=FONT_TICK)
plt.rc('ytick', labelsize=FONT_TICK)
plt.rc('legend', fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND)

BLUE_PALETTE = ["#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1"]
sns.set_palette(sns.color_palette(BLUE_PALETTE))

# ==========================================
# 2. TAGSET MAPPINGS
# ==========================================
TAGSET_MAP_5_TAG = {
    "noun_raw": "Raw", 
    "noun_plural_english": "Raw", 
    "cs_latin_raw": "Raw",
    "verb_light_construction": "LightVerb_Unintegrated", 
    "verb_light_latin": "LightVerb_Unintegrated", 
    "verb_light_greek": "LightVerb_Integrated",
    "noun_transliterated": "Adapted_Orthogra", 
    "noun_plural_native": "Adapted_Morph", 
    "noun_integrated_sg": "Adapted_Morph",
    "noun_integrated_pl": "Adapted_Morph", 
    "verb_morph_prescriptive": "Adapted_Morph", 
    "verb_morph_descriptive": "Adapted_Morph",
    "verb_participle_prescriptive": "Adapted_Morph", 
    "verb_participle_descriptive": "Adapted_Morph",
    "verb_morph_integrated": "Adapted_Morph", 
    "verb_habitual": "Adapted_Morph", 
    "verb_morph_aro": "Adapted_Morph", 
    "verb_participle": "Adapted_Morph"
}

TAGSET = [
    "Raw", 
    "Adapted_Orthogra", 
    "Adapted_Morph", 
    "LightVerb_Unintegrated", 
    "LightVerb_Integrated"
]

TAGSET_ORDER_3 = ["Raw", "Adapted (Morph/Ortho)", "Light Verb"]

# ==========================================
# 3. DATASET STATISTICS PLOTS
# ==========================================
class DatasetPlots:
    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df = df.copy()
        self.out_dir = output_dir
        self.df['data_source'] = self.df['type'].apply(
            lambda t: "Established" if "wiktionary" in str(t).lower() else "Synthetic"
        )
        self.df_synth = self.df[self.df['data_source'] == "Synthetic"].copy()

    def plot_dataset_sizes(self):
        plt.figure(figsize=(12, 10))
        ax = sns.countplot(data=self.df, x="lang", hue="data_source", palette=BLUE_PALETTE[:2])
        plt.title("Silver-standard dataset sizes", pad=20, fontweight='bold', fontsize=FONT_TITLE)
        plt.ylabel("Total sentences", fontsize=FONT_LABEL)
        plt.xlabel("Language", fontsize=FONT_LABEL)
        plt.legend(title="Data source", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND)
        plt.xticks(fontsize=FONT_TICK)
        plt.yticks(fontsize=FONT_TICK)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "4_dataset_sizes.png"), dpi=300)
        plt.close()

    def plot_pos_distribution(self):
        if self.df_synth.empty: return
        plt.figure(figsize=(14, 10))
        sns.countplot(data=self.df_synth, x="lang", hue="pos", palette=BLUE_PALETTE)
        plt.title("Part-of-speech distribution \n [tech neologisms]", pad=20, fontweight='bold', fontsize=FONT_TITLE)
        plt.ylabel("Contexts found", fontsize=FONT_LABEL)
        plt.xlabel("Language", fontsize=FONT_LABEL)
        plt.legend(title="Part of Speech", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND)
        plt.xticks(fontsize=FONT_TICK)
        plt.yticks(fontsize=FONT_TICK)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "1_pos_distribution.png"), dpi=300)
        plt.close()

    def plot_spelling_adaptation(self):
        if self.df_synth.empty: return
        def map_spelling(t):
            tag = TAGSET_MAP_5_TAG.get(str(t), "Other")
            if tag in ["Raw", "LightVerb_Unintegrated"]: return "Retained (Foreignization)"
            return "Modified (Nativization)"
            
        self.df_synth['spelling'] = self.df_synth['type'].apply(map_spelling)
        plt.figure(figsize=(12, 10))
        sns.countplot(data=self.df_synth, x="lang", hue="spelling", palette=BLUE_PALETTE[1:3])
        plt.title("Orthographic nativization vs. foreignization", pad=20, fontweight='bold', fontsize=FONT_TITLE)
        plt.ylabel("Contexts found", fontsize=FONT_LABEL)
        plt.xlabel("Language", fontsize=FONT_LABEL)
        plt.legend(title="Spelling strategy", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND)
        plt.xticks(fontsize=FONT_TICK)
        plt.yticks(fontsize=FONT_TICK)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "3_spelling_retained.png"), dpi=300)
        plt.close()

    def plot_integration_strategies(self):
        if self.df_synth.empty: return
        self.df_synth['tag_5'] = self.df_synth['type'].apply(lambda t: TAGSET_MAP_5_TAG.get(str(t), "Other"))
        df_5 = self.df_synth[self.df_synth['tag_5'] != "Other"]
        
        plt.figure(figsize=(16, 10))
        ax = sns.countplot(data=df_5, x="lang", hue="tag_5", hue_order=TAGSET, palette=BLUE_PALETTE)
        ax.set_yscale("log")
        plt.title("Integration strategies \n(5-tag classification)", pad=20, fontweight='bold', fontsize=FONT_TITLE)
        plt.ylabel("Contexts found [log scale]", fontsize=FONT_LABEL)
        plt.xlabel("Language", fontsize=FONT_LABEL)
        plt.legend(title="Tag", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND)
        plt.xticks(fontsize=FONT_TICK)
        plt.yticks(fontsize=FONT_TICK)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "2.2_integration_strats.png"), dpi=300)
        plt.close()

        def map_to_3_tag(tag):
            if tag == "Raw": return "Raw"
            if "LightVerb" in tag: return "Light Verb"
            if "Adapted" in tag: return "Adapted (Morph/Ortho)"
            return "Other"
            
        df_5['tag_3'] = df_5['tag_5'].apply(map_to_3_tag)
        plt.figure(figsize=(14, 10))
        ax = sns.countplot(data=df_5, x="lang", hue="tag_3", hue_order=TAGSET_ORDER_3, palette=BLUE_PALETTE[:3])
        ax.set_yscale("log")
        plt.title("Broad integration+adaptation strategies \n (3-tag classification)", pad=20, fontweight='bold', fontsize=FONT_TITLE)
        plt.ylabel("Contexts found [log scale]", fontsize=FONT_LABEL)
        plt.xlabel("Language", fontsize=FONT_LABEL)
        plt.legend(title="Strategy", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND)
        plt.xticks(fontsize=FONT_TICK)
        plt.yticks(fontsize=FONT_TICK)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "2.1_integration_strats.png"), dpi=300)
        plt.close()

# ==========================================
# 4. STANDARDIZED PLOTTING FUNCTIONS
# ==========================================
def plot_confusion_matrix(y_true, y_pred, labels: list, title: str, output_path: str):
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    plot_labels = [l for l in labels if l in unique_labels]
    
    if not plot_labels:
        logging.warning(f"Skipping CM {output_path}: No matching labels found.")
        return
        
    cm = confusion_matrix(y_true, y_pred, labels=plot_labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=plot_labels, yticklabels=plot_labels,
                annot_kws={"size": FONT_CM_ANNOTATION, "weight": "bold"})
                
    plt.title(title, pad=20, fontweight='bold', fontsize=FONT_TITLE)
    plt.ylabel('True tag (gold)', fontweight='bold', fontsize=FONT_LABEL)
    plt.xlabel('Predicted tag (model)', fontweight='bold', fontsize=FONT_LABEL)
    
    plt.xticks(rotation=45, ha='right', fontsize=FONT_CM_TICK, fontweight='bold')
    plt.yticks(rotation=0, fontsize=FONT_CM_TICK, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_per_class_f1(y_true, y_pred, labels: list, title: str, output_path: str):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    df = pd.DataFrame({'Taxonomy Class': labels, 'F1-Score': f1})
    
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=df, x="F1-Score", y="Taxonomy Class", order=labels, palette=list(reversed(BLUE_PALETTE)))
    
    for i, row in df.iterrows():
        ax.text(row['F1-Score'] + 0.02, i, f"{row['F1-Score']:.2f}", 
                color='black', va="center", fontsize=20, fontweight='bold')

    plt.title(title, pad=20, fontweight='bold', fontsize=FONT_TITLE)
    plt.xlabel("F1-Score", fontsize=FONT_LABEL)
    plt.ylabel("", fontsize=FONT_LABEL)
    plt.xlim(0, 1.1)
    
    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_CM_TICK, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_token_fragmentation(csv_path: str, output_path: str, title_prefix: str):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty: return
    
    df['Success rate (%)'] = df['success_rate'] * 100
    
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=df, x="fertility", y="Success rate (%)", palette=BLUE_PALETTE)
    
    for i, row in df.iterrows():
        ax.text(i, row['Success rate (%)'] + 2, f"n={int(row['total'])}", 
                color='black', ha="center", fontsize=20, fontweight='bold')

    plt.title(f"{title_prefix}: Impact of sub-word fragmentation", pad=20, fontweight='bold', fontsize=FONT_TITLE)
    plt.ylabel("Identification success rate (%)", fontsize=FONT_LABEL)
    plt.xlabel("Sub-word fertility (tokenizer splits)", fontsize=FONT_LABEL)
    plt.ylim(0, 115)
    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def export_and_plot_false_positives(true_spans: dict, pred_spans: dict, tagset: list, title: str, img_dir: str, stats_dir: str, experiment: str):
    fp_counts = {}
    for (case_id, txt), p_lbl in pred_spans.items():
        if p_lbl in tagset:
            t_lbl = true_spans.get((case_id, txt), "Native")
            if t_lbl not in tagset: 
                fp_counts[txt] = fp_counts.get(txt, 0) + 1
                
    if not fp_counts: return
        
    df_fp = pd.DataFrame(list(fp_counts.items()), columns=['Word/Span', 'False positives']).sort_values(by='False positives', ascending=False)
    
    csv_path = os.path.join(stats_dir, f"{experiment}_top_FalsePositives.csv")
    df_fp.head(50).to_csv(csv_path, index=False)
    
    df_top10 = df_fp.head(10)
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df_top10, x="False positives", y="Word/Span", palette="Blues_r")
    
    plt.title(f"Top 10 false positives \n [{title}]", pad=20, fontweight='bold', fontsize=FONT_TITLE)
    plt.xlabel("Frequency", fontsize=FONT_LABEL)
    plt.ylabel("Hallucinated / Over-extracted span", fontsize=FONT_LABEL)
    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_CM_TICK, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{experiment}_top_FPs_plot.png"), dpi=300)
    plt.close()

# ==========================================
# 5. EVALUATION LOGIC
# ==========================================
def evaluate_pipeline(pred_path: str, gold_path: str, img_dir: str, stats_dir: str, experiment: str):
    logging.info(f"\n{'='*60}\nEvaluating pipeline: {experiment}\n{pred_path}\n{'='*60}")
    
    stat_file = os.path.join(stats_dir, f"{experiment}_stats.txt")
    with open(stat_file, "w", encoding="utf-8") as f:
        f.write(f"=== EVALUATION STATS: {experiment} ===\n")
    
    true_spans, pred_spans, lang_map = _load_spans(pred_path, gold_path)
    all_keys = set(true_spans.keys()).union(set(pred_spans.keys()))
    
    # export FPs globally
    export_and_plot_false_positives(true_spans, pred_spans, TAGSET, experiment.upper(), img_dir, stats_dir, experiment)

    # over ALL plus the three languages
    target_subsets = ["ALL", "ast", "eu", "el"]

    for subset_name in target_subsets:
        if subset_name == "ALL":
            subset_keys = all_keys
        else:
            # lang_map keys are case_ids (strings), k is a tuple (case_id, txt)
            subset_keys = {k for k in all_keys if lang_map.get(k) == subset_name}

        if not subset_keys:
            logging.info(f"  > Skipping subset {subset_name.upper()} (No data matched)")
            continue
        
        prefix = f"{experiment}" if subset_name == "ALL" else f"{experiment}_{subset_name}"
        title_suffix = "" if subset_name == "ALL" else f" ({subset_name.upper()})"
        
        with open(stat_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Subset: {subset_name.upper()} ---\n")

        # 1. IDENTIFICATION
        y_true_id = ["Borrowing" if true_spans.get(k, "Native") in TAGSET else "Native" for k in subset_keys]
        y_pred_id = ["Borrowing" if pred_spans.get(k, "Native") in TAGSET else "Native" for k in subset_keys]
        
        get_metrics(y_true_id, y_pred_id, labels=["Borrowing"], average="binary", out_file=stat_file, task=f"ID {subset_name}")
        plot_confusion_matrix(y_true_id, y_pred_id, labels=["Native", "Borrowing"],
                              title=f"Borrowing detection{title_suffix}\n[{experiment.upper()}]",
                              output_path=os.path.join(img_dir, f"{prefix}_step1_cm.png"))

        # 2. CLASSIFICATION
        intersection_keys = subset_keys.intersection(set(true_spans.keys())).intersection(set(pred_spans.keys()))
        y_true_clf, y_pred_clf = [], []
        for key in intersection_keys:
            t_lbl = true_spans.get(key)
            p_lbl = pred_spans.get(key)
            if t_lbl not in TAGSET: continue
            if p_lbl not in TAGSET: p_lbl = "Raw"
            y_true_clf.append(t_lbl)
            y_pred_clf.append(p_lbl)

        if y_true_clf:
            # average="macro" >>> to average="micro"
            get_metrics(y_true_clf, y_pred_clf, labels=TAGSET, average="micro", out_file=stat_file, task=f"CLF {subset_name}")
            plot_confusion_matrix(y_true_clf, y_pred_clf, labels=TAGSET,
                                  title=f"Classification (exact matches){title_suffix}\n[{experiment.upper()}]",
                                  output_path=os.path.join(img_dir, f"{prefix}_step2_cm.png"))
                                  
            plot_per_class_f1(y_true_clf, y_pred_clf, labels=TAGSET,
                              title=f"Per-class classification F1{title_suffix}\n[{experiment.upper()}]",
                              output_path=os.path.join(img_dir, f"{prefix}_per_class_f1.png"))
            
        # 3. JOINT
        y_true_joint = [true_spans.get(k, "Native") if true_spans.get(k, "Native") in TAGSET else "Native" for k in subset_keys]
        y_pred_joint = [pred_spans.get(k, "Native") if pred_spans.get(k, "Native") in TAGSET else "Native" for k in subset_keys]

        # average="macro" >>> to average="micro"
        get_metrics(y_true_joint, y_pred_joint, labels=TAGSET, average="micro", out_file=stat_file, task=f"JOINT {subset_name}")
        plot_confusion_matrix(y_true_joint, y_pred_joint, labels=["Native"] + TAGSET,
                              title=f"Joint ID & classification{title_suffix}\n[{experiment.upper()}]",
                              output_path=os.path.join(img_dir, f"{prefix}_joint_cm.png"))
        
# --- helper metrics & parsing ---
def get_metrics(ground_truth, predictions, labels, average="binary", out_file=None, task="IDENTIFICATION"):
    p, r, f1, _ = precision_recall_fscore_support(ground_truth, predictions, labels=labels, 
                                                  average=average, pos_label="Borrowing" if average == "binary" else None, zero_division=0)
    output_str = f"[{task}] -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}\n"
    logging.info(output_str.strip())
    if out_file:
        with open(out_file, "a", encoding="utf-8") as f: f.write(output_str)

def _load_spans(pred_path: str, gold_path: str) -> Tuple[Dict, Dict, Dict]:
    with open(pred_path, "r", encoding="utf-8") as f: predictions = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f: ground_truth = json.load(f)

    gt_map = {}
    lang_map = {}
    for item in ground_truth:
        lang = item.get("lang", "") if "lang" in item else item.get("data", {}).get("lang", "")
        text = item.get("text", "") if "text" in item else item.get("data", {}).get("text", "")
        
        stable_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        case_id = str(item.get("id", stable_id))
        
        gt_map[case_id] = item.get("annotations", [])
        lang_map[case_id] = lang

    true_spans_dict, pred_spans_dict = {}, {}

    for record in predictions:
        case_id = str(record.get("id"))
        if case_id not in gt_map: continue

        pred_items = _parse_llm_output(record.get("prediction", []))
        if isinstance(pred_items, list):
            for p in pred_items:
                if isinstance(p, dict) and p.get("span") and p.get("label"):
                    start = p.get("start", -1)
                    end = p.get("end", -1)
                    lbl = _normalize_label(p["label"]) 
                    if lbl != "O": 
                         # incl. start/end boundaries for span matching to avoid overwriting matches
                        pred_spans_dict[(case_id, start, end)] = lbl
                        
        true_items = []
        for ann in gt_map[case_id]:
            if isinstance(ann, dict): true_items.extend(ann.get("result", []))
                
        for t in true_items:
            val = t.get("value", {})
            if "text" in val and "labels" in val:
                start = val.get("start", -1)
                end = val.get("end", -1)
                lbl = _normalize_label(val["labels"])
                # Use start/end boundaries as the key
                true_spans_dict[(case_id, start, end)] = lbl
                                
    return true_spans_dict, pred_spans_dict, lang_map

def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def _normalize_label(lbl):
    if not lbl: return "O"
    return re.sub(r"[\[\]\'\"]", "", str(lbl)).strip() or "O"

def _parse_llm_output(prediction_data) -> List[Dict]:
    if isinstance(prediction_data, list): return prediction_data
    try:
        prediction_str = re.sub(r"<think>.*?</think>", "", str(prediction_data), flags=re.DOTALL).strip()
        match = re.search(r"\[\s*\{.*\}\s*\]", prediction_str, re.DOTALL)
        if match: return json.loads(match.group())
        data = json.loads(prediction_str)
        return data if isinstance(data, list) else []
    except Exception: return []

# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    data_dir = "data"
    img_dir = "img"
    stats_dir = "stats"
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    logging.info("--- THESIS PLOT & EVALUATION GENERATOR ---")
    
    # 1. Dataset Statistics
    silver_file = os.path.join(data_dir, "silver_standard_set.jsonl") 
    if os.path.exists(silver_file):
        data = [json.loads(line) for line in open(silver_file, 'r', encoding='utf-8') if line.strip()]
        df = pd.DataFrame(data)
        if not df.empty:
            logging.info("\t> Generating dataset statistics plots...")
            plotter = DatasetPlots(df, img_dir)
            plotter.plot_dataset_sizes()
            plotter.plot_pos_distribution()
            plotter.plot_spelling_adaptation()
            plotter.plot_integration_strategies()

    # 2. Pipeline Evaluation & Confusion Matrices
    gold_file = os.path.join(data_dir, "gold_standard_set.json")
    if os.path.exists(gold_file):
        pred_files = glob.glob(os.path.join(data_dir, "predictions_*.json"))
        for pred_file in pred_files:
            experiment_name = os.path.basename(pred_file).replace("predictions_", "").replace(".json", "")
            evaluate_pipeline(pred_file, gold_file, img_dir, stats_dir, experiment_name)
    else:
        logging.warning(f"\t> (!) Missing gold standard: {gold_file}. Skipping evaluations.")

    # 3. Token Fragmentation Plots
    frag_files = glob.glob(os.path.join(data_dir, "*fragmentation*.csv"))
    for f in frag_files:
        model_name = os.path.basename(f).replace("_fragmentation", "").replace(".csv", "").upper()
        out_name = os.path.join(img_dir, f"{model_name}_token_fragmentation.png")
        plot_token_fragmentation(f, out_name, model_name)

    logging.info("--- GENERATION COMPLETE. Check 'img/' and 'stats/' folders. ---")

if __name__ == "__main__":
    main()