# visualizer.py
# ----------------------------------------------------------------
# analyzes and visualizes mined borrowing data
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class LoanwordVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        sns.set_theme(style="whitegrid")

    def plot_pos_distribution(self, output_path):
        """
        Visualizes the count of borrowed (Nouns vs Verbs) per language.
        >> Haspelmath's borrowability hierarchy.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x="lang", hue="pos", palette="viridis")
        plt.title("Borrowing by Part-of-Speech\n(Noun dominance vs Verbal resistance)")
        plt.ylabel("Number of sentences")
        plt.xlabel("Language")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_integration_strategies(self, output_path):
        """
        Visualizes the depth of integration (Wichmann's typology). 
        Classifies borrowings into `[Unintegrated, partially-integrated (Light Verb), or highly-integrated]`.
        """
        self.df['level'] = self.df['type'].apply(self._get_level)
        
        plt.figure(figsize=(12, 6))
        sns.countplot(
            data=self.df, 
            x="lang", 
            hue="level", 
            palette="rocket",
            hue_order=["1. Unintegrated", "2. Partially integrated", "3. Highly integrated"]
        )
        plt.title("Integration Strategies\n(How deeply does the loanword enter the grammar?)")
        plt.ylabel("Number of sentences")
        plt.xlabel("Language")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_visual_fidelity(self, output_path):
        """Visualizes how often the original English root string is preserved visible (foreignization vs. nativization).
        """
        self.df['fidelity'] = self.df.apply(self._check_substr, axis=1)
        
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=self.df, 
            x="lang", 
            hue="fidelity", 
            palette="Set2",
            hue_order=["Retained (English)", "Modified (Adapted)"]
        )
        plt.title("Visual fidelity\n(Is the English root spelling preserved?)")
        plt.ylabel("Number of sentences")
        plt.xlabel("Language")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _get_level(self, t):
        """Maps specific types to macro-integration levels."""
        t = str(t).lower()
        if "raw" in t or "english" in t or "latin" in t: 
            return "1. Unintegrated"
        if "light" in t: 
            return "2. Partially integrated"
        if "native" in t or "integrated" in t or "morph" in t or "transliterated" in t:
            return "3. Highly integrated"
        return "Other"

    def _check_substr(self, row):
        """Checks if the English lemma is a substring of the mined term."""
        root = str(row['lemma'])
        term = str(row['term'])
        
        # if the root is literally inside the term (e.g. 'click' in 'clickiar'), it is retained.
        if root in term:
            return "Retained (English)"
        return "Modified (Adapted)"