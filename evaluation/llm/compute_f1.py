# Ben Kabongo
# October 2025

import argparse
import json
import os
import pandas as pd
from typing import Dict, Any, List, Tuple
from evaluation.utils import safe_mean_std, save_results


def compute_precision_recall_f1(
    precision_labels: List[int],
    recall_labels: List[int]
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for a single example.
    
    Args:
        precision_labels: Binary labels for precision computation
        recall_labels: Binary labels for recall computation
    
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    # Precision: proportion of factual statements in precision set
    precision = sum(precision_labels) / len(precision_labels) if precision_labels else 0.0
    
    # Recall: proportion of factual statements in recall set
    recall = sum(recall_labels) / len(recall_labels) if recall_labels else 0.0
    
    # F1 score: harmonic mean of precision and recall
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_combined_factuality_stats(
    precision_df: pd.DataFrame,
    recall_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute precision, recall, and F1 statistics aggregated by example index.
    
    Args:
        precision_df: DataFrame with columns 'index', 'label' for precision
        recall_df: DataFrame with columns 'index', 'label' for recall
    
    Returns:
        Dictionary containing overall statistics and per-example scores
    """
    precision_scores = precision_df.groupby('index')['label'].apply(list).to_dict()
    recall_scores = recall_df.groupby('index')['label'].apply(list).to_dict()
    
    common_indices = sorted(set(precision_scores.keys()) & set(recall_scores.keys()))
    
    if not common_indices:
        raise ValueError("No common indices found between precision and recall dataframes")
    common_ratio = len(common_indices) / max(len(precision_scores), len(recall_scores))
    print(f"Found {len(common_indices)} common examples ({common_ratio:.2%} of the larger set)")
    print(f"Precision examples: {len(precision_scores)}, Recall examples: {len(recall_scores)}")

    precision_values = []
    recall_values = []
    f1_values = []
    example_details = []
    
    for idx in common_indices:
        prec_labels = precision_scores[idx]
        rec_labels = recall_scores[idx]
        
        metrics = compute_precision_recall_f1(prec_labels, rec_labels)
        
        precision_values.append(metrics['precision'])
        recall_values.append(metrics['recall'])
        f1_values.append(metrics['f1'])
        
        example_details.append({
            'index': int(idx),
            'precision': {
                'num_statements': len(prec_labels),
                'num_factual': sum(prec_labels),
                'score': metrics['precision']
            },
            'recall': {
                'num_statements': len(rec_labels),
                'num_factual': sum(rec_labels),
                'score': metrics['recall']
            },
            'f1_score': metrics['f1']
        })
    
    results = {
        'precision': safe_mean_std(precision_values),
        'recall': safe_mean_std(recall_values),
        'f1': safe_mean_std(f1_values),
        'num_examples': len(common_indices),
        'precision_total_statements': sum(len(precision_scores[idx]) for idx in common_indices),
        'recall_total_statements': sum(len(recall_scores[idx]) for idx in common_indices),
        'example_details': example_details
    }
    
    return results


def main(args):
    precision_df = pd.read_csv(args.precision_path)
    recall_df = pd.read_csv(args.recall_path)
    
    required_columns = ['index', 'label']
    for name, df in [('precision', precision_df), ('recall', recall_df)]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {name} dataframe: {missing_columns}")
    
    if args.output_path is None:
        base_prec, _ = os.path.splitext(args.precision_path)
        args.output_path = base_prec + "_precision_recall_f1.json"
    
    if os.path.exists(args.output_path):
        print(f"Results already exist at {args.output_path}.")
        with open(args.output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    
    print(f"Computing precision/recall/F1 statistics...")
    print(f"Precision dataframe: {len(precision_df)} statements")
    print(f"Recall dataframe: {len(recall_df)} statements")
    
    results = compute_combined_factuality_stats(precision_df, recall_df)
    
    save_results(args.output_path, results)
    print(f"\nResults saved to {args.output_path}")
    
    summary = {
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'num_examples': results['num_examples'],
        'precision_total_statements': results['precision_total_statements'],
        'recall_total_statements': results['recall_total_statements']
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute precision, recall, and F1 from two factuality datasets"
    )
    parser.add_argument(
        "--precision_path",
        type=str,
        required=True,
        help="Path to CSV file for precision computation"
    )
    parser.add_argument(
        "--recall_path",
        type=str,
        required=True,
        help="Path to CSV file for recall computation"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save JSON results (default: precision_path + '_precision_recall_f1.json')"
    )
    
    args = parser.parse_args()
    main(args)