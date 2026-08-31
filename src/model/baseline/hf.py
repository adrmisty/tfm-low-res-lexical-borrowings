# hf.py
# ----------------------------------------------------------------------------------------
# hugging face hub for saving models
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# may-2026

from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer
import logging
logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
from huggingface_hub import login
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent

login(token="REPLACE_WITH_TOKEN")

models_to_push = {
    #"results/post_review/model/mmbert/standard_binary": "adrirflorez/mmbert-binary-borrowings",
    #"results/post_review/model/mmbert/standard_multi":  "adrirflorez/mmbert-multi-borrowings",
    "results/post_review/model/xlmr/standard_binary":   "adrirflorez/xlmr-binary-borrowings",
    "results/post_review/model/xlmr/standard_multi":    "adrirflorez/xlmr-multi-borrowings",
    "results/post_review/model/xlmr/conloan_binary":   "adrirflorez/xlmr-conloan-binary-borrowings",
    "results/post_review/model/xlmr/conloan_multi":    "adrirflorez/xlmr-conloan-multi-borrowings"
}

def push_models(specific_path: str = None):
    """Pushes models to HF Hub. If specific_path is provided, pushes only that model."""
    targets = {specific_path: models_to_push[specific_path]} if specific_path else models_to_push

    for relative_path, hub_repo_id in targets.items():
        local_path = ROOT_DIR / relative_path
        
        if not local_path.exists():
            print(f"\t> (!): Cannot find the model '{local_path}'!")
            continue

        logging.info(f"\n> Uploading: {local_path} to HuggingFace -> {hub_repo_id}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
            
            # Dynamically load based on whether it's binary (token) or multi (sequence)
            if "binary" in relative_path:
                model = AutoModelForTokenClassification.from_pretrained(str(local_path), local_files_only=True)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(str(local_path), local_files_only=True)
            
            print(f"\t>> Pushing to {hub_repo_id}...")
            model.push_to_hub(hub_repo_id)
            tokenizer.push_to_hub(hub_repo_id)
            
            print(f">>> https://huggingface.co/{hub_repo_id}")
            
        except Exception as e:
            print(f"> (!) Error uploading {local_path}: {e}")

if __name__ == "__main__":
    push_models()