# plot.py
# ----------------------------------------------------------------
# analyzes and plots lexical borrowing data 
# (standardized coloring, bigger font size)
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jun-2026

import os
import json
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set global theme for larger fonts and standard color palette
sns.set_theme(style="whitegrid", font_scale=1.3, palette="viridis")

# Official Taxonomy Mapping
TAGSET_MAP = {
    "noun_raw": "Raw", 
    "noun_plural_english": "Raw", 
    "cs_latin_raw": "Raw",
    "verb_light_construction": "LightVerb_Unintegrated", 
    "verb_light_latin": "LightVerb_Unintegrated", 
    "verb_light_greek": "LightVerb_Integrated",
    "noun_transliterated": "Adapted_Orthogra",  # ** TODO: write in thesis that orthographical changes are IMPLIED **
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

TAGSET_ORDER = [
    "Raw", 
    "Adapted_Orthogra", 
    "Adapted_Morph", 
    "LightVerb_Unintegrated", 
    "LightVerb_Integrated"
]

class BorrowingPlots:
    """Plots for lexical borrowing statistics, depending on source, language, degree of integration..."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        self.df['data_source'] = self.df['type'].apply(
            lambda t: "Established" if "wiktionary" in str(t) else "Synthetic"
        )

    def plot_pos_distribution(self, output_path: str):
        df_synth = self.df[self.df['data_source'] == "Synthetic"].copy()
        if df_synth.empty:
            logging.warning("\t> (!) Warning: No synthetic data for PoS plot.")
            return

        plt.figure(figsize=(10, 8))
        sns.countplot(data=df_synth, x="lang", hue="pos")
        plt.title("Part-of-Speech distribution [tech neologisms]")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_integration_strategies(self, output_path: str):
        """Visualizes the distribution of the official taxonomy tags in the synthetic data."""
        df_synth = self.df[self.df['data_source'] == "Synthetic"].copy()
        df_synth['taxonomy_tag'] = df_synth['type'].apply(lambda t: TAGSET_MAP.get(str(t), "Other"))
        
        # Filter out "Other" just in case any weird types slipped through
        df_synth = df_synth[df_synth['taxonomy_tag'] != "Other"]
        
        plt.figure(figsize=(14, 8))
        ax = sns.countplot(
            data=df_synth, 
            x="lang", 
            hue="taxonomy_tag", 
            hue_order=TAGSET_ORDER
        )
        ax.set_yscale("log")
        plt.title("Tag distribution [synthetic silver standard]")
        plt.ylabel("Sentences found (log scale)")
        plt.legend(title="Integration strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_spelling_adaptation(self, output_path: str):
        """Plots broad foreignization vs. nativization."""
        df_synth = self.df[self.df['data_source'] == "Synthetic"].copy()
        df_synth['spelling'] = df_synth['type'].apply(self._map_spelling)
        
        plt.figure(figsize=(10, 8))
        sns.countplot(
            data=df_synth, 
            x="lang", 
            hue="spelling", 
            hue_order=["Retained (foreignization)", "Modified (nativization)"]
        )
        plt.title("Spelling adaptation strategies [tech neologisms]")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_data_amounts(self, output_path: str):
        plt.figure(figsize=(10, 8))
        sns.countplot(data=self.df, x="lang", hue="data_source")
        plt.title("Dataset size comparison")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_origin_languages(self, output_path: str):
        df_wik = self.df[self.df['data_source'] == "Established"].copy()
        if df_wik.empty: return

        df_wik['origin'] = df_wik['type'].apply(lambda t: t.split('_')[-1] if '_' in t else "unknown")
        
        plt.figure(figsize=(10, 8))
        sns.countplot(data=df_wik, x="lang", hue="origin")
        plt.title("Origin of established loans")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _map_spelling(self, t: str) -> str:
        tag = TAGSET_MAP.get(str(t), "Other")
        if tag in ["Raw", "LightVerb_Unintegrated"]:
            return "Retained (foreignization)"
        return "Modified (nativization)"


def generate_plots(clean_path: str = "data/corpus/processed/mined_sentences.clean.jsonl", 
                   output_dir: str = "results/plots/post_review"):
    if not os.path.exists(clean_path):
        logging.error(f"\t> (!) Cannot plot, missing file: {clean_path}")
        return
        
    data = []
    with open(clean_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
                    
    df = pd.DataFrame(data)
    if df.empty:
        logging.error("\t> (!) Dataset is empty, nothing to plot")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    plots = BorrowingPlots(df)
    
    logging.info(f"\t> Generating dataset plots in {output_dir}/...")
    plots.plot_pos_distribution(os.path.join(output_dir, "1_pos_distribution.png"))
    plots.plot_integration_strategies(os.path.join(output_dir, "2_integration_strats.png"))
    plots.plot_spelling_adaptation(os.path.join(output_dir, "3_spelling_retained.png"))
    plots.plot_data_amounts(os.path.join(output_dir, "4_dataset_sizes.png"))
    plots.plot_origin_languages(os.path.join(output_dir, "5_origin_langs.png"))

# --- tokenization analysis plots ---

def plot_token_analysis(tok_csv: str, clf_csv: str, output_dir: str):
    logging.info(f"\t> Generating Granular Analysis plots in {output_dir}/...")
    _plot_token_fragmentation(tok_csv, os.path.join(output_dir, "analysis_token_fragmentation.png"))
    _plot_per_class_f1(clf_csv, os.path.join(output_dir, "analysis_per_class_f1.png"))

def _plot_token_fragmentation(csv_path: str, output_path: str):
        df = pd.read_csv(csv_path)
        if df.empty: return
        
        df['Success rate (%)'] = df['success_rate'] * 100
        
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(data=df, x="fertility", y="Success rate (%)")
        
        for i, row in df.iterrows():
            ax.text(i, row['Success rate (%)'] + 2, f"n={row['total']}", 
                    color='black', ha="center", fontsize=12)

        plt.title("Impact of Sub-Word fragmentation on identification", pad=15, fontweight='bold')
        plt.ylabel("Identification success rate (%)")
        plt.xlabel("Sub-Word fertility (tokenizer Splits)")
        plt.ylim(0, 110)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

def _plot_per_class_f1(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    if df.empty: return
        
    df_classes = df[df['taxonomy_class'].isin(TAGSET_ORDER)].copy()
        
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_classes, x="f1-score", y="taxonomy_class", order=TAGSET_ORDER)
        
    plt.title("Morphological classification performance (Exact-Match F1)", pad=15, fontweight='bold')
    plt.xlabel("Macro F1-Score")
    plt.ylabel("")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
