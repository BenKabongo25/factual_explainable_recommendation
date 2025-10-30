# Ben Kabongo
# October 2025

import argparse
import json
import os
import pandas as pd
from typing import Dict, Any, List
from evaluation.utils import safe_mean_std, save_results


def compute_statement_entailment_scores(
    df: pd.DataFrame,
    row_index: int
) -> Dict[str, float]:
    """
    Compute statement-level entailment metrics for a single example.
    
    Args:
        df: DataFrame containing all predictions and references for one example
        row_index: The example index
    
    Returns:
        Dictionary with all entailment metrics
    """
    example_df = df[df['row_index'] == row_index].copy()
    
    # Get unique predicted and reference statement IDs
    pred_sids = example_df['pred_sid'].unique()
    ref_sids = example_df['ref_sid'].unique()
    
    num_pred = len(pred_sids)
    num_ref = len(ref_sids)
    
    scores = {}
    
    # === StEnt-cont-P: r2p_entail aggregated by pred_sid ===
    if num_pred > 0:
        max_r2p_entail_per_pred = []
        for pred_sid in pred_sids:
            pred_rows = example_df[example_df['pred_sid'] == pred_sid]
            max_entail = pred_rows['r2p_entail'].max()
            max_r2p_entail_per_pred.append(max_entail)
        scores['StEnt_cont_P'] = sum(max_r2p_entail_per_pred) / num_pred
    else:
        scores['StEnt_cont_P'] = 0.0
    
    # === StEnt-cont-R: p2r_entail aggregated by ref_sid ===
    if num_ref > 0:
        max_p2r_entail_per_ref = []
        for ref_sid in ref_sids:
            ref_rows = example_df[example_df['ref_sid'] == ref_sid]
            max_entail = ref_rows['p2r_entail'].max()
            max_p2r_entail_per_ref.append(max_entail)
        scores['StEnt_cont_R'] = sum(max_p2r_entail_per_ref) / num_ref
    else:
        scores['StEnt_cont_R'] = 0.0
    
    # === StEnt-cont-F1: Harmonic mean of P and R ===
    p = scores['StEnt_cont_P']
    r = scores['StEnt_cont_R']
    if p + r > 0:
        scores['StEnt_cont_F1'] = 2 * (p * r) / (p + r)
    else:
        scores['StEnt_cont_F1'] = 0.0
    
    # === StEnt-bin-P: Binary version using r2p_label or score comparison ===
    if num_pred > 0:
        binary_r2p_per_pred = []
        for pred_sid in pred_sids:
            pred_rows = example_df[example_df['pred_sid'] == pred_sid]
            # Check if any label is 'entailment' or entail > max(neutral, contradiction)
            is_entail = (
                (pred_rows['r2p_label'] == 'entailment').any() or
                (pred_rows['r2p_entail'] > pred_rows[['r2p_neutral', 'r2p_contradiction']].max(axis=1)).any()
            )
            binary_r2p_per_pred.append(1.0 if is_entail else 0.0)
        scores['StEnt_bin_P'] = sum(binary_r2p_per_pred) / num_pred
    else:
        scores['StEnt_bin_P'] = 0.0
    
    # === StEnt-bin-R: Binary version using p2r_label or score comparison ===
    if num_ref > 0:
        binary_p2r_per_ref = []
        for ref_sid in ref_sids:
            ref_rows = example_df[example_df['ref_sid'] == ref_sid]
            # Check if any label is 'entailment' or entail > max(neutral, contradiction)
            is_entail = (
                (ref_rows['p2r_label'] == 'entailment').any() or
                (ref_rows['p2r_entail'] > ref_rows[['p2r_neutral', 'p2r_contradiction']].max(axis=1)).any()
            )
            binary_p2r_per_ref.append(1.0 if is_entail else 0.0)
        scores['StEnt_bin_R'] = sum(binary_p2r_per_ref) / num_ref
    else:
        scores['StEnt_bin_R'] = 0.0
    
    # === StEnt-bin-F1: Harmonic mean of binary P and R ===
    p_bin = scores['StEnt_bin_P']
    r_bin = scores['StEnt_bin_R']
    if p_bin + r_bin > 0:
        scores['StEnt_bin_F1'] = 2 * (p_bin * r_bin) / (p_bin + r_bin)
    else:
        scores['StEnt_bin_F1'] = 0.0
    
    # === StCoh-cont-P: (r2p_entail - r2p_contradiction) aggregated by pred_sid ===
    if num_pred > 0:
        max_coh_r2p_per_pred = []
        for pred_sid in pred_sids:
            pred_rows = example_df[example_df['pred_sid'] == pred_sid]
            coh_scores = pred_rows['r2p_entail'] - pred_rows['r2p_contradiction']
            max_coh = coh_scores.max()
            max_coh_r2p_per_pred.append(max_coh)
        scores['StCoh_cont_P'] = sum(max_coh_r2p_per_pred) / num_pred
    else:
        scores['StCoh_cont_P'] = 0.0
    
    # === StCoh-cont-R: (p2r_entail - p2r_contradiction) aggregated by ref_sid ===
    if num_ref > 0:
        max_coh_p2r_per_ref = []
        for ref_sid in ref_sids:
            ref_rows = example_df[example_df['ref_sid'] == ref_sid]
            coh_scores = ref_rows['p2r_entail'] - ref_rows['p2r_contradiction']
            max_coh = coh_scores.max()
            max_coh_p2r_per_ref.append(max_coh)
        scores['StCoh_cont_R'] = sum(max_coh_p2r_per_ref) / num_ref
    else:
        scores['StCoh_cont_R'] = 0.0
    
    ## === StCoh-cont-F1: Harmonic mean of coherence P and R ===
    #p_coh = scores['StCoh_cont_P']
    #r_coh = scores['StCoh_cont_R']
    #if p_coh + r_coh > 0:
    #    scores['StCoh_cont_F1'] = 2 * (p_coh * r_coh) / (p_coh + r_coh)
    #else:
    #    scores['StCoh_cont_F1'] = 0.0
    
    # Add metadata
    scores['num_pred_statements'] = num_pred
    scores['num_ref_statements'] = num_ref
    
    return scores


def compute_aggregated_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute entailment metrics aggregated across all examples.
    
    Args:
        df: DataFrame with entailment scores
    
    Returns:
        Dictionary containing aggregated statistics
    """
    row_indices = df['row_index'].unique()
    
    print(f"Computing metrics for {len(row_indices)} examples...")
    
    # Store per-example scores
    all_scores = {
        'StEnt_cont_P': [],
        'StEnt_cont_R': [],
        'StEnt_cont_F1': [],
        'StEnt_bin_P': [],
        'StEnt_bin_R': [],
        'StEnt_bin_F1': [],
        'StCoh_cont_P': [],
        'StCoh_cont_R': [],
        'StCoh_cont_F1': []
    }
    
    example_details = []
    
    for idx in row_indices:
        example_scores = compute_statement_entailment_scores(df, idx)
        
        # Collect scores for aggregation
        for metric in all_scores.keys():
            all_scores[metric].append(example_scores[metric])
        
        # Store detailed information
        example_details.append({
            'row_index': int(idx),
            'num_pred_statements': example_scores['num_pred_statements'],
            'num_ref_statements': example_scores['num_ref_statements'],
            'StEnt_cont_P': example_scores['StEnt_cont_P'],
            'StEnt_cont_R': example_scores['StEnt_cont_R'],
            'StEnt_cont_F1': example_scores['StEnt_cont_F1'],
            'StEnt_bin_P': example_scores['StEnt_bin_P'],
            'StEnt_bin_R': example_scores['StEnt_bin_R'],
            'StEnt_bin_F1': example_scores['StEnt_bin_F1'],
            'StCoh_cont_P': example_scores['StCoh_cont_P'],
            'StCoh_cont_R': example_scores['StCoh_cont_R'],
            'StCoh_cont_F1': example_scores['StCoh_cont_F1']
        })
    
    # Compute mean and std for each metric
    results = {
        'num_examples': len(row_indices),
        'total_pairs': len(df)
    }
    
    for metric, values in all_scores.items():
        results[metric] = safe_mean_std(values)
    
    results['example_details'] = example_details
    
    return results


def main(args):
    df = pd.read_csv(args.input_path)
    
    # Verify required columns
    required_columns = [
        'row_index', 'pred_sid', 'ref_sid',
        'p2r_entail', 'p2r_neutral', 'p2r_contradiction', 'p2r_label',
        'r2p_entail', 'r2p_neutral', 'r2p_contradiction', 'r2p_label'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if args.output_path is None:
        base, _ = os.path.splitext(args.input_path)
        args.output_path = base + "_entailment_metrics.json"
    
    if os.path.exists(args.output_path):
        print(f"Results already exist at {args.output_path}.")
        with open(args.output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    
    print(f"Input dataframe: {len(df)} pairs")
    print(f"Number of examples: {df['row_index'].nunique()}")
    
    results = compute_aggregated_metrics(df)
    
    save_results(args.output_path, results)
    print(f"\nResults saved to {args.output_path}")
    
    # Print summary
    summary = {
        'num_examples': results['num_examples'],
        'total_pairs': results['total_pairs'],
        'StEnt_cont_P': results['StEnt_cont_P'],
        'StEnt_cont_R': results['StEnt_cont_R'],
        'StEnt_cont_F1': results['StEnt_cont_F1'],
        'StEnt_bin_P': results['StEnt_bin_P'],
        'StEnt_bin_R': results['StEnt_bin_R'],
        'StEnt_bin_F1': results['StEnt_bin_F1'],
        'StCoh_cont_P': results['StCoh_cont_P'],
        'StCoh_cont_R': results['StCoh_cont_R'],
        'StCoh_cont_F1': results['StCoh_cont_F1']
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute statement-level entailment and coherence metrics"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to CSV file with entailment scores"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save JSON results (default: input_path + '_entailment_metrics.json')"
    )
    
    args = parser.parse_args()
    main(args)