# Ben Kabongo
# October 2025


import argparse
import json
import os
import torch
from typing import List, Dict, Any
from tqdm import tqdm

from questeval.questeval_metric import QuestEval

from evaluation.utils import (
    load_results, read_data,
    safe_mean_std, save_results, simple_tokenize
)


def main(args):
    df = read_data(args.data_path)
    predictions: List[str] = df["prediction"].tolist()
    references: List[str] = df["reference"].tolist()

    if args.output_path is None:
        base, _ = os.path.splitext(args.data_path)
        args.output_path = base + "questeval.json"

    results = {}
    if os.path.exists(args.output_path):
        results = load_results(args.output_path)
        print(f"Loaded existing results from {args.output_path}")

    model = QuestEval(use_cache=False)
    if "score" in results:
        print(f"Results already exist in {args.output_path}, skipping evaluation.")
        #print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    scores: List[float] = []
    for start in tqdm(range(0, len(predictions), args.batch_size), desc="Evaluating"):
        end = min(start + args.batch_size, len(predictions))
        batch_preds = predictions[start:end]
        batch_refs = references[start:end]

        batch_scores = model.corpus_questeval(
            hypothesis=batch_preds, 
            sources=batch_refs
        )["ex_level_scores"]

        scores.extend(batch_scores)

    results: Dict[str, Any] = {}
    results["score"] = safe_mean_std(scores)
    save_results(args.output_path, results)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuestEval Evaluation Script")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with 'prediction' and 'reference' columns.")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save JSON results (overwritten at each step).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model-based metrics.")
    
    args = parser.parse_args()
    main(args)