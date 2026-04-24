# Automated Identification and Adaptation Prediction of Loanwords in Low-Resource Languages

This repository contains the codebase for a Master's Thesis focused on the automated extraction and morphological classification of lexical borrowings. The project evaluates multiple NLP architectures on three morphologically diverse, lower-resource European languages: **Asturian (ast)**, **Basque (eu)**, and **Greek (el)**.

This Master's Thesis is for the Erasmus Mundus+ Master's degree in Language and Communication Technologies, supervised by Mgr. Magda Ševčíková, Ph.D (Charles University, CZ) and Mgr. Jeremy Barnes, Ph.D (University of the Basque Country, ES). The thesis document (*in progress*) can be found on [Overleaf](https://www.overleaf.com/project/692da51703e8fd24ec17d2e9).

***

## Introduction

Lexical borrowing is a primary mechanism of language contact and evolution. When a target language (recipient) adopts a word from a source language (donor), the loanword typically undergoes varying degrees of phonological, orthographic, and morphological adaptation to fit the recipient's grammatical rules (Haspelmath, 2009). 

In morphologically rich and lower-resource languages like Asturian, Basque, and Greek, the adaptation strategies can be highly complex. For instance, verbs are rarely borrowed as raw stems; they frequently require native light verbs or specific morphological integration markers (Moravcsik, 1975; Wichmann & Wohlgemuth, 2008). 

To capture this spectrum of adaptation computationally, this project utilizes a **5-tag classification tagset** (alongside two noise-filtering tags for evaluation):

* **`Raw`**: Unassimilated borrowings that retain the source language's orthography and morphology (e.g., *bug*, *link*, *click*).
* **`Adapted_Orthogra`**: Borrowings exhibiting phonological/orthographic shifts or script transliterations, but lacking native inflectional markers (e.g., *pantaila*, *tuit*, *clic*, or Latin-to-Greek alphabet transliterations).
* **`Adapted_Morph`**: Fully integrated borrowings that participate in the target language's inflectional and derivational systems, such as nominal declensions or verbal conjugations (e.g., *hackeatu*, *resetiamos*, *τρολάρει*).
* **`LightVerb_Unintegrated`**: A multi-word borrowing strategy utilizing a native light verb paired with a raw, unassimilated foreign noun/stem (e.g., *reset egin*).
* **`LightVerb_Integrated`**: A native light verb paired with an orthographically/morphologically adapted foreign noun (e.g., *κάνει κλικ*).
* **`Invalid_NE` / `Invalid_FalsePos`**: Adversarial tags used to filter out Named Entities and metalinguistic mentions during evaluation.

*(Note: Legacy tags such as `Adapted_Translit`/`Internationalism` have been conceptually merged into `Adapted_Orthogra` for cross-lingual and distribution consistency, and to achieve better baseline results!)*

---

## Installation

Make sure you are using **Python 3.11+**! Execution for the modules below is handled via the root `main.py` script.

```bash
git clone https://github.com/adrmisty/tfm-low-res-lexical-borrowings
cd tfm-low-res-lexical-borrowings
pip install -r requirements.txt
```

*Hardware Note: For LLM inference (e.g., Qwen 27B), a multi-GPU setup is recommended. Adjust `tensor_parallel_size` in `src/model/baseline/llm.py` accordingly. Smaller models and encoders can run on a single standard GPU.*

---

## 1. Data Pipeline (`src/data/`)

Models require vast amounts of annotated data, which is largely unavailable for lexical borrowings in low-resource languages. The `src/data/` module solves this by generating a massive, synthetically mined **silver standard dataset**. A sample of 200 examples per language has been carefully and iteratively annotated (with each change in the desired tagset) to achieve a test **gold standard sample**. 

All the resulting data from this pipeline and subsequent annotations can be found in `data/annotation` and `data/corpus`. The plots and statistics generated with the `src/data/analysis` submodule for these preliminary datasets can be found in `results/plots/`.

### 1. Seed Synthesis & Scraping (`src/data/domain/`)
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

### 2. Corpus Mining, Cleaning & Analysis (`src/data/mining/` & `src/data/analysis/`)
Once thousands of theoretical loanword forms (seeds) are generated, the `miner.py` script scans massive monolingual corpora (e.g., Wikipedia dumps) to find real-world contexts containing these exact tokens. 
* Sentences are extracted, contextualized, and paired with the heuristic tag used to generate the seed (e.g., a sentence matched via a light-verb rule is automatically tagged as `LightVerb_Unintegrated`).
* The `cleaner.py` ensures length restrictions, filters out parsing noise, and formats the output into a clean JSONL dataset ready for transformer training.

```bash
# Scan raw monolingual text dumps for seed matches
python src/data/main.py --action mine --corpus data/corpus/raw/ --output data/corpus/mined/

# Clean, filter, and format sentences into JSONL for model training
python src/data/main.py --action clean --input data/corpus/mined/ --output data/corpus/processed/mined_sentences.clean.jsonl

# Generate dataset statistics and taxonomy distribution plots
python src/data/main.py --action stats
```

---

## 2. Modeling & Evaluation Pipeline (`src/model/baseline/`)

This module orchestrates the inference and training logic for three distinct AI methodologies, allowing for a comprehensive comparative analysis of zero-shot, fine-tuned, and in-context learning approaches.

### 1. Language Identification at the Word Level (`langid.py`)
This serves as the foundational baseline. It treats borrowing detection purely as a foreign-character/n-gram anomaly detection task. 
* Utilizes Hugging Face's `facebook/fasttext-language-identification` model.
* The text is tokenized, and FastText evaluates the language of each word in isolation. Tokens diverging from the target language's ISO code (e.g., an `eng_Latn` word in an `eus_Latn` sentence) are automatically flagged as `Raw` borrowings. This baseline does not serve for the morphological classification subtask.

```bash
python main.py --action run --type langid --langs ast eu el
```

### 2. Multilingual Contextual Encoders (`encoder.py`)
This methodology fine-tunes masked language models (**XLM-RoBERTa** and **mmBERT**) on the synthetically mined silver data using a 2-step pipeline:
* **Step 1 (Borrowing Span Detection):** A token classification head (`AutoModelForTokenClassification`) predicts binary `[Native, Borrowing]` boundaries at the sub-word level, utilizing a custom dynamically-weighted cross-entropy loss function to penalize missed loanwords.
* **Step 2 (Morphological Classification):** A sequence classification head (`AutoModelForSequenceClassification`). By passing both the isolated borrowing and the full surrounding sentence separated by the `</s></s>` token (`[SPAN] </s></s> [CONTEXT]`), the encoder learns how the target word interacts morphologically with its context.

```bash
# XLM-RoBERTa
python main.py --action run --type encoder --model xlm-roberta-base --pipeline 2step --langs ast eu el

# mmBERT
python main.py --action run --type encoder --model jhu-clsp/mmBERT-base --pipeline 2step --langs ast eu el
```

### 3. Large Language Models (`llm.py`)
This evaluates the efficacy of generative LLMs (specifically the **Qwen** architecture) utilizing few-shot in-context learning. The few-shot examples have been manually crafted per-language and can be found in `data/icl/few_shot_examples.json`. This baseline features:
* **vLLM Integration:** Inference is highly optimized using `vLLM` to process thousands of prompts simultaneously on the GPU.
* **Dynamic Prompting (`prompt.py`):** Supports both **1-step** (joint extraction and classification) and **2-step** (prompt chaining) pipelines.
* **$K$-Shot Scaling:** Allows for dynamic injection of $k$ few-shot examples per taxonomy class to evaluate how empirical prompting scales in low-resource linguistic environments. Evaluated predictions are parsed natively from LLM-generated JSON strings.

```bash
# 1-step (one prompt) or 2-step (prompt chain) pipeline with dynamic k-shot injection
python main.py --action run --type vllm --model Qwen/Qwen3.5-9B --pipeline 2step --k 2 --langs ast eu el
```

### 4. Evaluation (`eval.py`)
Evaluate any model's predictions against the manually annotated gold standard.

```bash
# Point to the specific timestamped prediction file
python main.py --action eval --pred_file results/model/Qwen3.5-9B/2step/predictions_Qwen3.5-9B_2step_2shot_TIMESTAMP.json --title QWEN-2STEP --langs ast eu el
```

#### Output Artifacts
The evaluation script generates the following assets in the `results/model/<model_name>/<pipeline>/` directory:
* **`predictions_*.json`**: The raw inference output containing the sentence `id`, `lang`, the executed prompt, and the extracted spans/labels.
* **Confusion Matrices (`.png`)**: Heatmaps generated via Seaborn for visual error analysis.
    * `*_step1_cm.png`: Binary span detection performance (`Native` vs. `Borrowing`).
    * `*_step2_cm.png`: Sequence classification performance across the 5-tag morphological taxonomy.
    * `*_joint_cm.png`: Cross-lingual, end-to-end pipeline evaluation.
* **`*_stats.txt`**: Standard output logs capturing Precision, Recall, and F1 scores (macro-averaged) for both individual language splits and the joint evaluation.

---

## Author

**Adriana R. Flórez**
*Computational Linguist & Software Engineer*
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

---

*Built with ❤️ using Python.*