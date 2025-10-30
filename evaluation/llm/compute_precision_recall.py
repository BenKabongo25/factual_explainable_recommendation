# Ben Kabongo
# October 2025

import argparse
import json
import os
import pandas as pd
from typing import Dict, Any, List
from evaluation.utils import safe_mean_std, save_results


def compute_factuality_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute factuality statistics aggregated by example index.
    
    Args:
        df: DataFrame with columns 'index', 'label'
    
    Returns:
        Dictionary containing overall statistics and per-example scores
    """
    example_scores = df.groupby('index')['label'].apply(list).to_dict()
    
    example_means = []
    example_details = []
    
    for idx in sorted(example_scores.keys()):
        labels = example_scores[idx]
        mean_score = sum(labels) / len(labels) if labels else 0.0
        example_means.append(mean_score)
        
        example_details.append({
            'index': int(idx),
            'num_statements': len(labels),
            'num_factual': sum(labels),
            'mean_score': mean_score
        })
    
    results = {
        'overall_score': safe_mean_std(example_means),
        'num_examples': len(example_means),
        'total_statements': len(df),
        'example_details': example_details
    }
    
    return results


def main(args):
    df = pd.read_csv(args.data_path)
    
    required_columns = ['index', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if args.output_path is None:
        base, _ = os.path.splitext(args.data_path)
        args.output_path = base + "_factuality_stats.json"
    
    if os.path.exists(args.output_path):
        print(f"Results already exist at {args.output_path}.")
        with open(args.output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    
    print(f"Computing factuality statistics for {len(df)} statements...")
    results = compute_factuality_stats(df)
    
    save_results(args.output_path, results)
    print(f"\nResults saved to {args.output_path}")
    
    print(json.dumps({
        'overall_score': results['overall_score'],
        'num_examples': results['num_examples'],
        'total_statements': results['total_statements']
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute factuality statistics from statement-level labels"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to CSV file with statement-level factuality labels"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None, 
        help="Path to save JSON results (default: input_path + '_factuality_stats.json')"
    )
    
    args = parser.parse_args()
    main(args)