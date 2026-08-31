# Automated Identification and Adaptation Prediction of Loanwords in Low-Resource Languages

This repository contains the codebase for a Master's Thesis focused on the automated extraction and morphological classification of lexical borrowings. The project evaluates multiple NLP architectures on three morphologically diverse, lower-resource European languages: **Asturian (ast)**, **Basque (eu)**, and **Greek (el)**.

This Master's Thesis is for the Erasmus Mundus+ Master's degree in Language and Communication Technologies, supervised by Mgr. Magda Ševčíková, Ph.D. (Charles University, CZ) and Mgr. Jeremy Barnes, Ph.D. (University of the Basque Country, ES).

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

Models require vast amounts of annotated data, which is largely unavailable for lexical borrowings in low-resource languages. The `src/data/` module solves this by generating a massive, synthetically mined **silver standard dataset** (~26k sentences). A sample of sentences per language has been carefully and iteratively annotated to achieve a balanced **gold standard test set** (465 sentences).

### 1.1. Seed synthesis and scraping (`src/data/domain/`)

Instead of relying on random manual searches, the pipeline generates target loanwords computationally:

* **Synthetic Generators:** Language-specific rule engines (`asturian.py`, `basque.py`, `greek.py`) apply prescriptive and descriptive morphological rules to foreign stems. For example, a generator might take the English word *hack*, apply Basque morphological rules, and output synthetic permutations like *hackeatu*, *hackeatzen*...
* **Scrapers:** Scripts dynamically pull known borrowed terms from structured source categories in Wiktionary.

#### Usage

```bash
# Scrape base foreign terms (e.g., from Wiktionary)
python -m src.data.main --action scrape

# Apply language-specific prescriptive/descriptive morphological rules
python -m src.data.main --action generate --langs ast eu el

```

### 1.2. Corpus mining, cleaning and analysis (`src/data/mining/` & `src/data/analysis/`)

Once thousands of theoretical loanword forms (seeds) are generated, the `miner.py` script scans massive monolingual corpora (e.g., Wikipedia dumps) to find real-world contexts containing these exact tokens.

* **Exact-boundary matching:** Spans are mapped case-insensitively using precise character offsets (`start`, `end`), ensuring that natural capitalization is preserved for the model without breaking sequence labels.
* The `cleaner.py` enforces length restrictions, resolves `wiktionary_*` tag mappings to retain maximum data, and strictly filters out any sentences present in the gold standard test set to guarantee zero data leakage.

```bash
# Scan raw monolingual text dumps for seed matches
python -m src.data.main --action mine --corpus data/corpus/raw/ --output data/corpus/mined/

# Clean, filter, and format sentences into JSONL for model training
python -m src.data.main --action clean --input data/corpus/mined/mined_sentences.jsonl --output data/corpus/processed/mined_sentences.clean.jsonl

```

---

## 2. Modeling pipeline (`src/model/baseline/`)

This module orchestrates the inference and training logic for three distinct AI methodologies, allowing for a comprehensive comparative analysis of zero-shot, fine-tuned, and in-context learning approaches.

### 2.1. Language Identification at the Word Level (`langid.py`)

This serves as the foundational baseline. It treats borrowing detection purely as a foreign-character/n-gram anomaly detection task.

* Utilizes Hugging Face's `facebook/fasttext-language-identification` model.
* FastText evaluates the language of each word in isolation. Tokens diverging from the target language's ISO code are automatically flagged as `Raw` borrowings. This baseline only identifies spans and does not perform morphological classification.

```bash
python main.py --action run --type langid --langs ast eu el

```
### 2.2. Multilingual Contextual Encoders (`encoder.py`)

This methodology fine-tunes masked language models (**XLM-RoBERTa** and **mmBERT**) using a 2-step pipeline. The framework supports dynamic caching (`run_name`) to seamlessly toggle between training on the native synthetic dataset (`standard`) and external transfer learning datasets (`conloan`).

* **Step 1 (Borrowing span detection):** A token classification head predicts binary `[Native, Borrowing]` boundaries at the sub-word level, utilizing a custom dynamically-weighted cross-entropy loss function to penalize missed loanwords.
* **Step 2 (Morphological classification):** A sequence classification head. By passing both the isolated borrowing and the full surrounding sentence separated by the `</s></s>` token (`[SPAN] </s></s> [CONTEXT]`), the encoder learns how the target word interacts morphologically with its context.

**Checkpoints:**
The fully fine-tuned models (including the ConLoan hybrid experiments) from this pipeline have been open-sourced and are available for direct use via the Hugging Face Hub:

* `adrirflorez/xlmr-binary-borrowings[-conloan]`
* `adrirflorez/xlmr-multi-borrowings[-conloan]`
* `adrirflorez/mmbert-binary-borrowings[-conloan]`
* `adrirflorez/mmbert-multi-borrowings[-conloan]`

### 2.3. Large Language Models (`llm.py`)

Evaluates the efficacy of generative LLMs (specifically **Qwen3.5-9B**) utilizing few-shot in-context learning.

* **Dynamic prompting (`prompt.py`):** Supports both **1-step** (joint extraction and classification) and **2-step** (prompt chaining) pipelines.
* **$K$-Shot scaling:** Allows for dynamic injection of $k$ few-shot examples (0-shot, 3-shot, 4-shot) per taxonomy class to evaluate how empirical prompting scales in low-resource linguistic environments. Evaluated predictions are parsed natively from LLM-generated JSON strings, mapping explicit character boundaries to resolve identical span collisions.

---

## 3. Unified evaluation and visualization (`eval.py`)

The evaluation framework has been completely overhauled during the review phase to resolve critical scoring anomalies, alignment failures, and data leakage. All evaluations now leverage **Micro-F1** scoring and **Exact-Boundary Matching** to ensure robust, statistically sound cross-lingual comparisons.

### 3.1. Evaluation Methodology

*   **Scoring metric: Macro-F1 $\rightarrow$ Micro-F1**
    *   *Issue:* The pipeline originally used Macro-F1, which averages the F1 score of each class equally regardless of support. Because morphological adaptation strategies vary heavily by language (e.g., Asturian and Basque have highly sparse or completely empty occurrences of specific light-verb constructs), these empty classes artificially tanked the overall classification scores to near zero.
    *   *Fix:* Switched to Micro-F1 for classification evaluation. 
*   **Span identification: String Matching $\rightarrow$ Exact-Boundary Offsets**
    *   *Issue:* Evaluation relied on simple string matching. If a borrowed word appeared twice in the same sentence, or if the model's casing differed from the ground truth, the evaluation dictionary keys overwrote each other or failed to match. This caused false negatives and inaccurate token mapping.
    *   *Fix:* Spans are now mapped utilizing a 4-item tuple tracking precise character offsets: `(case_id, start, end, txt)`. This perfectly resolves identical span collisions, seamlessly handles capitalization variances, and ensures exact token-to-label alignment.

### Directory Structure

Evaluation outputs are dynamically routed to `results/{exp_name}/model/` based on the active experiment (e.g. `exp_name=post_review`):

```text
results/post_review/model/
├── {model_name}/ 
│   ├── predictions_{model}_{timestamp}.json  # Raw inference outputs with start/end boundaries
│   ├── img/                                  # Auto-generated figures (confusion matrices, bar charts)
│   └── stats/                                # Auto-generated stats

```

### Execution

The evaluation logic is fully integrated into the root CLI. Simply point the script to any generated prediction JSON:

```bash
# Evaluate a specific model run
python main.py --action eval --pred_file results/post_review/model/encoder/predictions_encoder_2step_20260831_160604.json --title XLMR_STANDARD
```

### Evaluation output:

1. **Dataset distributions (`img/`)**: Bar charts mapping dataset sizes, Part-of-Speech distributions, and integration strategies across languages.

2. **Confusion matrices (`img/`)**: Heatmaps generated for **Aggregate** performance and individually for **Asturian (ast)**, **Basque (eu)**, and **Greek (el)**.
* `*_step1_cm.png`: Identification (Binary).
* `*_step2_cm.png`: Classification (Exact-Match).
* `*_joint_cm.png`: e2e pipeline evaluation.


3. **Granular performance views (`img/`)**:
* `*_per_class_f1.png`: Horizontal bar charts displaying exact-match F1-scores per taxonomy tag.
* `*_token_fragmentation.png`: Bar charts mapping the impact of sub-word fertility/splits on boundary detection success.
* `*_top_FPs_plot.png`: Distribution of the top 10 most frequent semantic hallucinations.


4. **Textual metrics (`stats/`)**:
* `*_stats.txt`: Comprehensive Precision, Recall, and Micro-F1 logs.
* `*_top_FalsePositives.csv`: Extracted lists of historical cognates and native terms hallucinated by the models.


---

## Author

**Adriana Rodríguez**
*Computational Linguist & Software Engineer*
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

---

*Built with ❤️ using Python.*

*June 2026 · Donostia-San Sebastián, EHU / September 2026 · Prague, MFF CUNI*