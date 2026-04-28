#!/bin/bash
# ================================================================
# run_experiments.sh
# Script to reproduce baseline inference, training, and 
# evaluation for the Lexical Borrowing NLP pipeline.
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# apr-2026
# ================================================================

# Exit immediately if any command fails
set -e

LANGUAGES="ast eu el"
GPU_ID=5
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "================================================================"
echo " [TFM] LEXICAL BORROWING EXPERIMENTS"
echo " Languages: $LANGUAGES"
echo " GPU Device: $GPU_ID"
echo "================================================================"

evaluate_latest() {
    """Automatically finds and evaluates the most recent prediction file in a given directory."""
    local DIR=$1
    local TITLE=$2
    
    echo " > Looking for latest predictions in: $DIR"
    # Grab the most recently modified JSON file in the directory
    LATEST_FILE=$(ls -t "$DIR"/predictions_*.json 2>/dev/null | head -n 1)
    
    if [ -z "$LATEST_FILE" ]; then
        echo " [!] No prediction file found in $DIR. Skipping evaluation."
    else
        echo " [+] Found: $LATEST_FILE"
        python main.py --action eval --pred_file "$LATEST_FILE" --title "$TITLE" --langs $LANGUAGES
    fi
    echo "----------------------------------------------------------------"
}

# ----------------------------------------------------------------
# 1. FastText (Word-Level Language ID Baseline)
# ----------------------------------------------------------------
echo ">>> [1/5] Running FastText..."
python main.py --action run --type langid --langs $LANGUAGES
evaluate_latest "results/model/FastText" "FASTTEXT"

# ----------------------------------------------------------------
# 2.1 XLM-RoBERTa (Contextual Encoder)
# ----------------------------------------------------------------
echo ">>> [2.1/3] Running XLM-RoBERTa (2-step)..."
python main.py --action run --type xlmr --langs ast eu el --langs $LANGUAGES
evaluate_latest "results/model/xlmr" "XLM-RoBERTa"

# ----------------------------------------------------------------
# 2.2 mmBERT (Contextual Encoder)
# ----------------------------------------------------------------
echo ">>> [2.2/5] Running mmBERT (2-step)..."
python main.py --action run --type mmbert --langs $LANGUAGES
evaluate_latest "results/model/mmbert" "MMBERT"

# ----------------------------------------------------------------
# 4. Qwen3.5-9B (LLM with vLLM - 2-Step Pipeline)
# ----------------------------------------------------------------
echo ">>> [3.1/3] Running Qwen3.5-9B (2-step, 3-shot)..."
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --action run --type llm --pipeline 2step --k 3 --langs $LANGUAGES
evaluate_latest "results/model/Qwen3.5-9B/2step" "QWEN-2STEP-2SHOT"

# ----------------------------------------------------------------
# 5. Qwen3.5-9B (LLM with vLLM - 1-Step Pipeline)
# ----------------------------------------------------------------
echo ">>> [3.2/3] Running Qwen3.5-9B (1-step, 3-shot)..."
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --action run --type llm --pipeline 1step --k 3 --langs $LANGUAGES
evaluate_latest "results/model/Qwen3.5-9B/1step" "QWEN-1STEP-2SHOT"

echo "================================================================"
echo " ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo " >>> Check results/model/ for output JSONs and evaluation plots!"
echo "================================================================"