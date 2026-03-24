# baseline.py
# ----------------------------------------------------------------
# baseline modeling for [(step 1) LEXICAL BORROWING IDENTIFICATION]
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# mar-2026

from llm import BorrowingLLM
from eval import eval_borrowings


def run_fewzeroshot_baseline(langs: list[str], model_id="Qwen/Qwen2.5-7B-Instruct", gt="data/annotation/final/test_gold_annotations.json"):
    """Few-shot/Zero-shot prompting on LLM for (step 1) LEXICAL BORROWING IDENTIFICATION."""
    llm = BorrowingLLM(model_id, gt)

    all_predictions = []
    all_ground_truth = []

    for lang in langs:
        print(f"\n>>> Running [zero-shot/few-shot LEXICAL BORROWING IDENTIFICATION] baseline for [{lang.upper()}]...")

        few_shot_items, test_items = llm.data_splits[lang]
        shots = [
            {"text": ex["text"], "output": ex["gold_output"]} 
            for ex in few_shot_items
        ]
        all_ground_truth.extend(
            {"id": item["id"], "annotations": item["raw_annotations"]}
            for item in test_items
        )

        predictions = llm.get_borrowings(
            test_data=test_items, 
            language=lang, 
            examples=shots
        )
        all_predictions.extend(predictions)

    print("\n" + "="*50)
    print("ZERO-SHOT/FEW-SHOT EVALUATION RESULTS")
    print("="*50)
    eval_borrowings(all_predictions, all_ground_truth)