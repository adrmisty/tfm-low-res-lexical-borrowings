# plot.py
# ----------------------------------------------------------------
# analyzes and plots lexical borrowing data
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import os
import json
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
        sns.set_theme(style="whitegrid")
        
        self.df['data_source'] = self.df['type'].apply(
            lambda t: "Established" if "wiktionary" in str(t) else "Synthetic"
        )

    def plot_pos_distribution(self, output_path: str):
        df_synth = self.df[self.df['data_source'] == "Synthetic"].copy()
        if df_synth.empty:
            logging.warning("\t> (!) Warning: No synthetic data for PoS plot.")
            return

        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_synth, x="lang", hue="pos", palette="viridis")
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
        
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(
            data=df_synth, 
            x="lang", 
            hue="taxonomy_tag", 
            palette="Spectral",
            hue_order=TAGSET_ORDER
        )
        ax.set_yscale("log")
        plt.title("Taxonomy Tag Distribution [Synthetic Silver Standard]")
        plt.ylabel("Sentences found (log scale)")
        plt.legend(title="Integration Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_spelling_adaptation(self, output_path: str):
        """Plots broad foreignization vs. nativization."""
        df_synth = self.df[self.df['data_source'] == "Synthetic"].copy()
        df_synth['spelling'] = df_synth['type'].apply(self._map_spelling)
        
        plt.figure(figsize=(8, 6))
        sns.countplot(
            data=df_synth, 
            x="lang", 
            hue="spelling", 
            palette="Set2",
            hue_order=["Retained (foreignization)", "Modified (nativization)"]
        )
        plt.title("Spelling adaptation strategies [tech neologisms]")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_data_amounts(self, output_path: str):
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x="lang", hue="data_source", palette="mako")
        plt.title("Dataset size comparison")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_origin_languages(self, output_path: str):
        df_wik = self.df[self.df['data_source'] == "Established"].copy()
        if df_wik.empty: return

        df_wik['origin'] = df_wik['type'].apply(lambda t: t.split('_')[-1] if '_' in t else "unknown")
        
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_wik, x="lang", hue="origin", palette="magma")
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

# --- Integration with pipeline.py ---

def generate_plots(clean_path: str = "data/corpus/processed/mined_sentences.clean.jsonl", 
                   output_dir: str = "results/plots/v1"):
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