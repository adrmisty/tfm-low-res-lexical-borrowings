# Automated Identification and Adaptation Prediction of Loanwords in Low-Resource Languages

This repository contains the codebase for a Master's Thesis focused on the automated extraction and morphological classification of lexical borrowings. The project evaluates multiple NLP architectures on three morphologically diverse, lower-resource European languages: **Asturian (ast)**, **Basque (eu)**, and **Greek (el)**.

This Master's Thesis is for the Erasmus Mundus+ Master's degree in Language and Communication Technologies, supervised by Mgr. Magda Ševčíková, Ph.D (Charles University, CZ) and Mgr. Jeremy Barnes, Ph.D (University of the Basque Country, ES). 
---

## Introduction

Lexical borrowing is a primary mechanism of language contact and evolution. When a target language (recipient) adopts a word from a source language (donor), the loanword typically undergoes varying degrees of phonological, orthographic, and morphological adaptation to fit the recipient's grammatical rules (Haspelmath, 2009).

In morphologically rich and lower-resource languages like Asturian, Basque, and Greek, the adaptation strategies can be highly complex. For instance, verbs are rarely borrowed as raw stems; they frequently require native light verbs or specific morphological integration markers (Moravcsik, 1975; Wichmann & Wohlgemuth, 2008).

To capture this spectrum of adaptation computationally, this project utilizes a **5-tag classification tagset** (alongside two noise-filtering tags for evaluation):

* **`Raw`**: Unassimilated borrowings that retain the source language's orthography and morphology (e.g., *bug*, *link*, *click*).
* **`Adapted_Orthogra`**: Borrowings exhibiting phonological/orthographic shifts or script transliterations, but lacking native inflectional markers (e.g., *pantaila*, *tuit*, *clic*, or Latin-to-Greek alphabet transliterations).
* **`Adapted_Morph`**: Fully integrated borrowings that participate in the target language's inflectional and derivational systems, such as nominal declensions or verbal conjugations (e.g., *hackeatu*, *resetiamos*, *τρολάρει*).
* **`LightVerb_Unintegrated`**: A multi-word borrowing strategy utilizing a native light verb paired with a raw, unassimilated foreign noun/stem (e.g., *reset egin*).
* **`LightVerb_Integrated`**: A native light verb paired with an orthographically/morphologically adapted foreign noun (e.g., *κάνει κλικ*).
* **`Invalid_NE` / `Invalid_FalsePos**`: Adversarial tags used to filter out Named Entities and metalinguistic mentions during evaluation.

*(Note: Legacy tags such as `Adapted_Translit`/`Internationalism` have been conceptually merged into `Adapted_Orthogra` for cross-lingual and distribution consistency, achieving robust baseline generalizations).*

---

## Installation

Make sure you are using **Python 3.11+**! Execution for the modules below is handled via the root `main.py` script.

```bash
git clone https://github.com/adrmisty/tfm-low-res-lexical-borrowings
cd tfm-low-res-lexical-borrowings
pip install -r requirements.txt

```

*Hardware Note: For LLM inference (e.g., Qwen 9B), a multi-GPU setup is recommended. Adjust `tensor_parallel_size` in `src/model/baseline/llm.py` accordingly. Smaller models and encoders can run on a single standard GPU.*

---

## 1. Data pipeline (`src/data/`)

Models require vast amounts of annotated data, which is largely unavailable for lexical borrowings in low-resource languages. The `src/data/` module solves this by generating a massive, synthetically mined **silver standard dataset**. A sample of sentences per language has been carefully and iteratively annotated to achieve a balanced **gold standard test set**.

The legacy intermediate data from this pipeline can be found in `data/annotation` and `data/corpus`. However, the final, unified datasets and all evaluation artifacts used for the thesis document are consolidated in the `thesis/data/` directory.

### 1.1. Seed synthesis and scraping (`src/data/domain/`)

Instead of relying on random manual searches, the pipeline generates target loanwords computationally:

* **Synthetic Generators:** Language-specific rule engines (`asturian.py`, `basque.py`, `greek.py`) apply prescriptive and descriptive morphological rules to foreign stems. For example, a generator might take the English word *hack*, apply Basque morphological rules, and output synthetic permutations like *hackeatu*, *hackeatzen*...
* **Scrapers:** Scripts dynamically pull known borrowed terms from structured source categories in Wiktionary.

#### Usage

```bash
# Scrape base foreign terms (e.g., from Wiktionary)
python src/data/main.py --action scrape

# Apply language-specific prescriptive/descriptive morphological rules
python src/data/main.py --action generate --langs ast eu el

```

### 1.2. Corpus mining, cleaning and analysis (`src/data/mining/` & `src/data/analysis/`)

Once thousands of theoretical loanword forms (seeds) are generated, the `miner.py` script scans massive monolingual corpora (e.g., Wikipedia dumps) to find real-world contexts containing these exact tokens.

* Sentences are extracted, contextualized, and paired with the heuristic tag used to generate the seed (e.g., a sentence matched via a light-verb rule is automatically tagged as `LightVerb_Unintegrated`).
* The `cleaner.py` ensures length restrictions, filters out parsing noise, and formats the output into a clean JSONL dataset ready for transformer training.

```bash
# Scan raw monolingual text dumps for seed matches
python src/data/main.py --action mine --corpus data/corpus/raw/ --output data/corpus/mined/

# Clean, filter, and format sentences into JSONL for model training
python src/data/main.py --action clean --input data/corpus/mined/ --output thesis/data/silver_standard_set.jsonl

```

---

## 2. Modeling pipeline (`src/model/baseline/`)

This module orchestrates the inference and training logic for three distinct AI methodologies, allowing for a comprehensive comparative analysis of zero-shot, fine-tuned, and in-context learning approaches.

### 2.1. Language Identification at the Word Level (`langid.py`)

This serves as the foundational baseline. It treats borrowing detection purely as a foreign-character/n-gram anomaly detection task.

* Utilizes Hugging Face's `facebook/fasttext-language-identification` model.
* FastText evaluates the language of each word in isolation. Tokens diverging from the target language's ISO code are automatically flagged as `Raw` borrowings. This baseline does not serve for the morphological classification subtask.

```bash
python main.py --action run --type langid --langs ast eu el

```

### 2.2. Multilingual Contextual Encoders (`encoder.py`)

This methodology fine-tunes masked language models (**XLM-RoBERTa** and **mmBERT**) on the synthetically mined silver data (and evaluates the impact of external distant supervision via the **ConLoan** dataset) using a 2-step pipeline:

* **Step 1 (Borrowing span detection):** A token classification head predicts binary `[Native, Borrowing]` boundaries at the sub-word level, utilizing a custom dynamically-weighted cross-entropy loss function to penalize missed loanwords.
* **Step 2 (Morphological classification):** A sequence classification head. By passing both the isolated borrowing and the full surrounding sentence separated by the `</s></s>` token (`[SPAN] </s></s> [CONTEXT]`), the encoder learns how the target word interacts morphologically with its context.

**Checkpoints:**
The fully fine-tuned models (including the ConLoan experiments) from this pipeline have been open-sourced and are available for direct use via the Hugging Face Hub:

* `arodriguezf/xlmr-binary-borrowings[-conloan]`
* `arodriguezf/xlmr-multi-borrowings[-conloan]`
* `arodriguezf/mmbert-binary-borrowings[-conloan]`
* `arodriguezf/mmbert-multi-borrowings[-conloan]`

### 2.3. Large Language Models (`llm.py`)

Evaluates the efficacy of generative LLMs (specifically **Qwen3.5-9B**) utilizing few-shot in-context learning.

* **Dynamic prompting (`prompt.py`):** Supports both **1-step** (joint extraction and classification) and **2-step** (prompt chaining) pipelines.
* **$K$-Shot scaling:** Allows for dynamic injection of $k$ few-shot examples (0-shot, 3-shot, 4-shot) per taxonomy class to evaluate how empirical prompting scales in low-resource linguistic environments. Evaluated predictions are parsed natively from LLM-generated JSON strings containing embedded chain-of-thought logic.

---

## 3. Unified evaluation and visualization (`plot.py`)

All raw prediction outputs, fragmentation metrics, and final corpora are unified within the `thesis/` directory. The evaluation and plotting logic is entirely automated through the `plot.py` script, which enforces standardized, high-readability aesthetics for the final document.

### Directory Structure

```text
thesis/
├── data/
│   ├── gold_standard_set.json               # Manual annotations
│   ├── silver_standard_set.jsonl            # Synthetic corpus
│   ├── predictions_*.json                   # Outputs from FastText, XLM-R, mmBERT, Qwen
│   └── *_fragmentation.csv                  # Sub-word tokenization data
├── img/                                     # Auto-generated PNG figures
└── stats/                                   # Auto-generated textual metrics and CSVs

```

### Execution

Simply navigate to the `thesis/` directory and run the plotter. It will automatically detect all prediction and fragmentation files in the `data/` folder, compare them against the gold standard, and populate the `img/` and `stats/` directories.

```bash
cd thesis
python plot.py
```

### Output Artifacts Generated:

1. **Dataset distributions (`img/`)**: Bar charts mapping dataset sizes, Part-of-Speech distributions, and integration strategies across languages.
2. **Confusion matrices (`img/`)**: Heatmaps generated for **Aggregate** performance and individually for **Asturian (ast)**, **Basque (eu)**, and **Greek (el)**.
* `*_step1_cm.png`: Identification (Binary).
* `*_step2_cm.png`: Classification (Exact-Match).
* `*_joint_cm.png`: e2e pipeline evaluation.


3. **Granular performance vies (`img/`)**:
* `*_per_class_f1.png`: Horizontal bar charts displaying Macro F1-scores per taxonomy tag.
* `*_token_fragmentation.png`: Bar charts mapping the impact of sub-word fertility/splits on boundary detection success.
* `*_top_FPs_plot.png`: Distribution of the top 10 most frequent semantic hallucinations.


4. **Textual metrices (`stats/`)**:
* `*_stats.txt`: Comprehensive Precision, Recall, and F1 logs.
* `*_top_FalsePositives.csv`: Extracted lists of historical cognates and native terms hallucinated by the models.

---

## Author

**Adriana R. Flórez**
*Computational Linguist & Software Engineer*
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

---

*Built with ❤️ using Python.*

*June 2026 · Donostia-San Sebastián, EHU / September 2026 · Prague, MFF CUNI*
