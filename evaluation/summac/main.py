# Ben Kabongo
# October 2025


import argparse
import json
import os
import torch
from typing import List, Dict, Any

from evaluation.summac.model_summac import SummaCZS, SummaCConv

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
        args.output_path = base + "summac.json"

    results = {}
    if os.path.exists(args.output_path):
        results = load_results(args.output_path)
        print(f"Loaded existing results from {args.output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_zs = SummaCZS(
        granularity="sentence", 
        model_name="mnli-base", 
        device=device
    )

    model_conv = SummaCConv(
        granularity="sentence", 
        models=["mnli-base"], 
        bins='percentile', 
        nli_labels="e", 
        device=device, 
        start_file="default", 
        agg="mean"
    )

    if "SummaC_ZS" in results and "SummaC_Conv" in results:
        print(f"Results already exist in {args.output_path}, skipping evaluation.")
        #print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    zs_scores: List[float] = []
    conv_scores: List[float] = []

    for start in range(0, len(predictions), args.batch_size):
        end = min(start + args.batch_size, len(predictions))
        batch_preds = predictions[start:end]
        batch_refs = references[start:end]

        batch_zs_scores = model_zs.score(batch_preds, batch_refs)["scores"]
        batch_conv_scores = model_conv.score(batch_preds, batch_refs)["scores"]

        zs_scores.extend(batch_zs_scores)
        conv_scores.extend(batch_conv_scores)

    results: Dict[str, Any] = {}
    results["SummaC_ZS"] = safe_mean_std(zs_scores)
    results["SummaC_Conv"] = safe_mean_std(conv_scores)
    save_results(args.output_path, results)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SummaC Evaluation Script")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with 'prediction' and 'reference' columns.")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save JSON results (overwritten at each step).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model-based metrics.")
    
    args = parser.parse_args()
    main(args)