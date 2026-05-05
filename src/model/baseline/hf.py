# hf.py
# ----------------------------------------------------------------------------------------
# hugging face hub for saving models
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty)
# may-2026

from transformers import AutoModelForTokenClassification, AutoTokenizer
import logging
logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
models_to_push = {
    "results/model/mmBert/mmbert/mmbert_binary":      "arodriguezf/mmbert-binary-borrowings",
    "results/model/mmBert/mmbert/mmbert_multi":       "arodriguezf/mmbert-multi-borrowings",
    "results/model/XLM-RoBERTa/xlmr/xlmr_binary":   "arodriguezf/xlmr-binary-borrowings",
    "results/model/XLM-RoBERTa/xlmr/xlmr_multi":    "arodriguezf/xlmr-multi-borrowings"
}

def push_models():
    for relative_path, hub_repo_id in models_to_push.items():
        
        local_path = ROOT_DIR / relative_path
        
        if not local_path.exists():
            print(f"\t> (!): Cannot find the model '{local_path}'!")
            continue

        logging.info(f"\n> Uploading: {local_path} to HuggingFace")

        try:
            model = AutoModelForTokenClassification.from_pretrained(str(local_path), local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
            
            print(f"\t>> {hub_repo_id}")
            model.push_to_hub(hub_repo_id)
            tokenizer.push_to_hub(hub_repo_id)
            
            print(f"✅ >>> https://huggingface.co/{hub_repo_id}")
            
        except Exception as e:
            print(f"> (!) Error uploading {local_path}: {e}")