# plot.py
# ----------------------------------------------------------------
# analyzes and plots lexical borrowing data
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class BorrowingPlots:
    """Plots for lexical borrowing statistics, depending on source, language, degree of integration..."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        sns.set_theme(style="whitegrid")
        
        self.df['data_source'] = self.df['type'].apply(
            lambda t: "Established (Wiktionary)" if "wiktionary" in str(t) else "Tech neologism (Synthetic)"
        )

    def plot_pos_distribution(self, output_path):
        """Visualizes Noun vs Verb dominance (Tech Neologisms only)."""
        
        df_synth = self.df[self.df['data_source'] == "Tech neologism (Synthetic)"].copy()
        
        if df_synth.empty:
            print("(!) Warning: No synthetic data for PoS plot.")
            return

        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_synth, x="lang", hue="pos", palette="viridis")
        plt.title("Part-of-Speech distribution [tech neologisms]")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_integration_strategies(self, output_path):
        """Visualizes Wichmann's Integration Scale."""
        df_synth = self.df[self.df['data_source'] == "Tech neologism (Synthetic)"].copy()
        
        df_synth['level'] = df_synth['type'].apply(self._map_integration)
        
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(
            data=df_synth, 
            x="lang", 
            hue="level", 
            palette="rocket",
            hue_order=["1. Unintegrated", "2. Accommodated (light verb)", "3. Highly integrated"]
        )
        ax.set_yscale("log")
        
        plt.title("Integration strategies [tech neologisms]")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_spelling_adaptation(self, output_path):
        """Plots foreignization vs. nativization of spelling."""
        
        df_synth = self.df[self.df['data_source'] == "Tech neologism (Synthetic)"].copy()
        
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

    def plot_data_amounts(self, output_path):
        """Compares synthetic vs Wiktionary data volume."""
        
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x="lang", hue="data_source", palette="mako")
        plt.title("Dataset size comparison")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_origin_languages(self, output_path):
        """Analyzes origin of established loans."""
        
        df_wik = self.df[self.df['data_source'] == "Established (Wiktionary)"].copy()
        if df_wik.empty: return

        df_wik['origin'] = df_wik['type'].apply(lambda t: t.split('_')[-1] if '_' in t else "unknown")
        
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_wik, x="lang", hue="origin", palette="magma")
        plt.title("Origin of established loans")
        plt.ylabel("Sentences found")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    # -----------------------------------------------------------------------------------------

    def _map_integration(self, t):
        t = str(t)
        
        # UNINTEGRATED (raw and codeswitches)
        if t in ["noun_raw", "noun_plural_english", "cs_latin_raw"]:
            return "1. Unintegrated"
            
        # LIGHT VERBS with native grammar
        if t in ["verb_light_construction", "verb_light_latin", "verb_light_greek"]:
            return "2. Accommodated (light verb)"
            
        # INTEGRATED with morphological adaptation
        if t in ["noun_plural_native", "noun_integrated_sg", "noun_integrated_pl", 
                 "noun_transliterated", "verb_morph_prescriptive", "verb_morph_descriptive",
                 "verb_participle_prescriptive", "verb_participle_descriptive",
                 "verb_morph_integrated", "verb_habitual", "verb_morph_aro", "verb_participle"]:
            return "3. Highly integrated"
            
        return "Other"

    def _map_spelling(self, t):
        t = str(t)
        
        # RETAINED: English spelling is preserved 100% (foreignization)
        if t in ["noun_raw", "noun_plural_english", "cs_latin_raw", 
                 "verb_light_construction", "verb_light_latin", "verb_light_greek"]:
            return "Retained (foreignization)"
            
        # MODIFIED: English spelling is adapted (nativization)
        # e.g. native plurals ("clickes") and transliterations ("κλικ")
        return "Modified (nativization)"
