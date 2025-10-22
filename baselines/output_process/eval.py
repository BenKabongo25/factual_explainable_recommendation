# Ben Kabongo
# October 2025


import argparse
import ast
import json
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from tqdm import tqdm


def _parse_list(s):
    if isinstance(s, list):
        return s
    if pd.isna(s):
        return []
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return []


def _stage_cols(df: pd.DataFrame) -> List[str]:
    """Return stage_* columns sorted by numeric suffix (stage_0, stage_1, ...)."""
    cols = [c for c in df.columns if c.startswith("stage_")]
    # sort by int suffix
    cols.sort(key=lambda x: int(x.split("_")[1]))
    return cols


def _code_key(row: pd.Series, L: int, stage_cols: List[str]) -> Optional[Tuple]:
    """Build RVQ code key (topic, sentiment, stage_0..stage_{L-1}) from a row."""
    try:
        topic = str(row["topic"]).strip().lower()
        sentiment = str(row["sentiment"]).strip().lower()
        if len(stage_cols) < L:
            return None
        stages = tuple(int(row[c]) for c in stage_cols[:L])
        return (topic, sentiment) + stages
    except Exception:
        return None


def precision_at_k(hits_binary: np.ndarray, k: int) -> float:
    k = int(k)
    top = hits_binary[:k].sum() if len(hits_binary) >= k else hits_binary.sum()
    # If predictions shorter than k, missing tails are zeros by definition.
    return float(top) / float(k)


def recall_at_k(hits_binary: np.ndarray, k: int, num_positives: int) -> float:
    if num_positives <= 0:
        return 0.0
    top = hits_binary[:k].sum() if len(hits_binary) >= k else hits_binary.sum()
    return float(top) / float(num_positives)


def hitrate_at_k(hits_binary: np.ndarray, k: int) -> float:
    k = int(k)
    return float(1.0 if hits_binary[:k].sum() > 0 else 0.0)


def ndcg_at_k(hits_binary: np.ndarray, k: int, num_positives: int) -> float:
    """Binary relevance NDCG@k with ideal DCG computed from num_positives."""
    k = int(k)

    # DCG
    dcg = 0.0
    for i in range(min(k, len(hits_binary))):
        if hits_binary[i]:
            # rank positions are 1-based in DCG
            rank = i + 1
            dcg += 1.0 / np.log2(rank + 1.0)

    # IDCG
    ideal_hits = min(k, max(0, num_positives))
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def mean_and_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return 0.0, 0.0
    arr = np.array(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))



def load_topics(dataset_dir: str) -> List[str]:
    topics_path = os.path.join(dataset_dir, "topics.json")
    with open(topics_path, "r") as f:
        topics = json.load(f)
    # Ensure canonical lowercase strings
    topics = [str(t).strip().lower() for t in topics]
    return topics


def load_dataset_rvq(dataset_dir: str, L: int, allowed_topics: Set[str]) -> Dict[int, Tuple]:
    """
    Load dataset-wide RVQ mapping: global_index -> code_key.
    """
    rvq_path = os.path.join(dataset_dir, "rvq", "quantized_sts.csv")
    rvq_df = pd.read_csv(rvq_path)
    stages = _stage_cols(rvq_df)

    mapping: Dict[int, Tuple] = {}
    for _, row in tqdm(rvq_df.iterrows(), total=len(rvq_df), desc="Building dataset_code_map", unit="rows"):
        gi = int(row["global_index"])
        topic = str(row["topic"]).strip().lower()
        if topic not in allowed_topics:
            continue
        key = _code_key(row, L, stages)
        if key is not None:
            mapping[gi] = key
    return mapping


def load_splits(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["statement_ids"] = df["statement_ids"].apply(_parse_list)
        return df

    train_df = _prep(pd.read_csv(os.path.join(dataset_dir, "train_data.csv")))
    eval_df  = _prep(pd.read_csv(os.path.join(dataset_dir, "eval_data.csv")))
    test_df  = _prep(pd.read_csv(os.path.join(dataset_dir, "test_data.csv")))
    return train_df, eval_df, test_df


def build_item_candidate_codes(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ds_code_map: Dict[int, Tuple]
) -> Dict[str, Set[Tuple]]:
    """
    For each item, collect all code_keys across train+eval+test (dedup).
    """
    item2codes: Dict[str, Set[Tuple]] = {}

    def _add_split(df: pd.DataFrame):
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building item2codes", unit="rows"):
            item = str(row["item_id"])
            sids = row["statement_ids"]
            bucket = item2codes.setdefault(item, set())
            for sid in sids:
                key = ds_code_map.get(int(sid))
                if key is not None:
                    bucket.add(key)

    _add_split(train_df)
    _add_split(eval_df)
    _add_split(test_df)
    return item2codes


def load_baseline_predictions(
    baseline_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """
    Returns:
        processed_df: DataFrame aligned with test rows; must have 'statements_ids'
        rvq_df:      Baseline quantized_sts.csv (for mapping local ids -> code_key)
        id2text:     Optional mapping local_id -> statement text (from sts.csv) for residuals
    """
    # Prefer processed_statement.csv (it contains 'statements_ids')
    processed_path = os.path.join(baseline_dir, "processed_statement.csv")
    if os.path.exists(processed_path):
        processed_df = pd.read_csv(processed_path)
    else:
        # Fallback: use statements.csv (list of dicts) and resolve to ids via sts.csv
        stmts_path = os.path.join(baseline_dir, "statement.csv")
        sts_path = os.path.join(baseline_dir, "sts.csv")
        if not (os.path.exists(stmts_path) and os.path.exists(sts_path)):
            raise FileNotFoundError(
                "Neither processed_statement.csv nor (statement.csv + sts.csv) found in baseline_dir."
            )
        stmts_df = pd.read_csv(stmts_path)
        sts_df = pd.read_csv(sts_path)
        # Build text->id
        text2id = {str(row["statement"]).strip(): int(row["id"]) for _, row in sts_df.iterrows()}
        # Build processed-like DataFrame with 'statements_ids'
        ids = []
        for _, row in stmts_df.iterrows():
            triple_list = _parse_list(row.get("statements", "[]"))
            loc_ids = []
            for trip in triple_list:
                try:
                    st_txt = str(trip.get("statement", "")).strip()
                    if st_txt in text2id:
                        loc_ids.append(text2id[st_txt])
                except Exception:
                    continue
            ids.append(loc_ids)
        processed_df = pd.DataFrame({"statements_ids": ids})

    # Ensure list form
    if "statements_ids" not in processed_df.columns:
        raise ValueError("processed_statement.csv must contain a 'statements_ids' column (list of local ids).")
    processed_df = processed_df.copy()
    processed_df["statements_ids"] = processed_df["statements_ids"].apply(_parse_list)

    # Baseline RVQ codes for its local unique statements
    rvq_path = os.path.join(baseline_dir, "quantized_sts.csv")
    if not os.path.exists(rvq_path):
        raise FileNotFoundError("Baseline quantized_sts.csv not found.")
    rvq_df = pd.read_csv(rvq_path)

    # Optional: local id -> text for residuals (if sts.csv exists)
    id2text: Dict[int, str] = {}
    sts_path2 = os.path.join(baseline_dir, "sts.csv")
    if os.path.exists(sts_path2):
        sts_df2 = pd.read_csv(sts_path2)
        for _, row in sts_df2.iterrows():
            id2text[int(row["id"])] = str(row["statement"]).strip()

    return processed_df, rvq_df, id2text


def build_local_code_map(rvq_df: pd.DataFrame, L: int, allowed_topics: Set[str]) -> Dict[int, Tuple]:
    """
    Map baseline-local 'global_index' -> code_key (topic, sentiment, stages[:L]).
    """
    stages = _stage_cols(rvq_df)
    mapping: Dict[int, Tuple] = {}
    for _, row in tqdm(rvq_df.iterrows(), total=len(rvq_df), desc="Building local_code_map", unit="rows"):
        lid = int(row["global_index"])
        topic = str(row["topic"]).strip().lower()
        if topic not in allowed_topics:
            continue
        key = _code_key(row, L, stages)
        if key is not None:
            mapping[lid] = key
    return mapping



def evaluate(
    dataset_dir: str,
    baseline_dir: str,
    level: int,
    ks: List[int],
) -> None:
    # Topics
    topics = load_topics(dataset_dir)
    allowed_topics = set(topics)

    # Dataset RVQ mapping and splits
    ds_code_map = load_dataset_rvq(dataset_dir, level, allowed_topics)
    train_df, eval_df, test_df = load_splits(dataset_dir)

    # Item candidate codes
    item2codes = build_item_candidate_codes(train_df, eval_df, test_df, ds_code_map)

    # Baseline predictions and local RVQ map
    processed_df, base_rvq_df, id2text = load_baseline_predictions(baseline_dir)
    local_code_map = build_local_code_map(base_rvq_df, level, allowed_topics)

    # Align lengths: processed rows must match test rows
    if len(processed_df) != len(test_df):
        raise ValueError(
            f"Row count mismatch: processed ({len(processed_df)}) vs test_data ({len(test_df)})."
        )

    # Accumulators
    prec_logs = {k: [] for k in ks}
    rec_logs  = {k: [] for k in ks}
    hr_logs   = {k: [] for k in ks}
    ndcg_logs = {k: [] for k in ks}

    coverage_counts: List[float] = []
    valid_rate_logs: List[float] = []

    # Residuals (hallucinations) collection
    residual_rows = []

    # Helper: compute GT positive code set for an interaction
    def gt_codes_for_row(sids: List[int]) -> Set[Tuple]:
        pos: Set[Tuple] = set()
        for sid in sids:
            key = ds_code_map.get(int(sid))
            if key is not None:
                pos.add(key)
        return pos

    # Iterate aligned rows
    for idx in tqdm(range(len(test_df)), desc="Evaluating", unit="rows"):
        item_id = str(test_df.iloc[idx]["item_id"])
        gt_sids = _parse_list(test_df.iloc[idx]["statement_ids"])
        gt_pos_codes = gt_codes_for_row(gt_sids)

        # Coverage (unique ref codes at level L)
        coverage_counts.append(float(len(gt_pos_codes)))

        # Candidate universe for the item
        cand_codes = item2codes.get(item_id, set())

        # Predicted local ids -> code keys (dedup by first occurrence)
        pred_loc_ids = processed_df.iloc[idx]["statements_ids"]
        pred_codes_ordered: List[Tuple] = []
        seen: Set[Tuple] = set()

        hallucinated_loc_ids: List[int] = []
        hallucinated_codes: List[Tuple] = []
        hallucinated_texts: List[str] = []

        valid_count = 0

        for lid in pred_loc_ids:
            key = local_code_map.get(int(lid))
            if key is None:
                # Unmappable -> treat as hallucinated FP (no code or topic invalid)
                hallucinated_loc_ids.append(int(lid))
                hallucinated_codes.append(("__invalid__", "__invalid__"))
                hallucinated_texts.append(id2text.get(int(lid), ""))
                continue

            if key not in seen:
                pred_codes_ordered.append(key)
                seen.add(key)

            if key in cand_codes:
                valid_count += 1
            else:
                hallucinated_loc_ids.append(int(lid))
                hallucinated_codes.append(key)
                hallucinated_texts.append(id2text.get(int(lid), ""))

        # Valid prediction rate for this row
        total_preds = len(pred_loc_ids) if len(pred_loc_ids) > 0 else 1  # avoid div by 0
        valid_rate_logs.append(float(valid_count) / float(total_preds))

        # Build binary hit vector over the prediction order only
        # (we do NOT append any additional positives after predictions).
        hits = np.array([1 if c in gt_pos_codes else 0 for c in pred_codes_ordered], dtype=int)

        # Metrics per k
        num_pos = len(gt_pos_codes)
        for k in ks:
            prec_logs[k].append(precision_at_k(hits, k))
            rec_logs[k].append(recall_at_k(hits, k, num_pos))
            hr_logs[k].append(hitrate_at_k(hits, k))
            ndcg_logs[k].append(ndcg_at_k(hits, k, num_pos))

        # Residual row
        residual_rows.append({
            "row_index": idx,
            "user_id": test_df.iloc[idx]["user_id"],
            "item_id": item_id,
            "num_predictions": len(pred_loc_ids),
            "num_valid_predictions": int(valid_count),
            "num_hallucinated": len(hallucinated_loc_ids),
            "coverage_count_L": len(gt_pos_codes),
            "hallucinated_statement_ids": json.dumps(hallucinated_loc_ids) if hallucinated_loc_ids else None,
            "hallucinated_codes": json.dumps(hallucinated_codes) if hallucinated_codes else None,
            "hallucinated_texts": json.dumps(hallucinated_texts) if hallucinated_texts else None,
        })

    # Aggregate (mean, std)
    results = {
        "level": level,
        "k": ks,
        "metrics": {
            "precision": {f"P@{k}": {"mean": 0.0, "std": 0.0} for k in ks},
            "recall":    {f"R@{k}": {"mean": 0.0, "std": 0.0} for k in ks},
            "hit_rate":  {f"HR@{k}": {"mean": 0.0, "std": 0.0} for k in ks},
            "ndcg":      {f"NDCG@{k}": {"mean": 0.0, "std": 0.0} for k in ks},
        },
        "coverage_count": {"mean": {"mean": 0.0, "std": 0.0}},
        "valid_prediction_rate": {"mean": 0.0, "std": 0.0},
    }

    # Fill metric means/stds properly
    # (unpackers above need explicit assignment to keep clarity)
    cov_mean, cov_std = mean_and_std(coverage_counts)
    v_mean, v_std = mean_and_std(valid_rate_logs)
    results["coverage_count"] = {"mean": cov_mean, "std": cov_std}
    results["valid_prediction_rate"] = {"mean": v_mean, "std": v_std}

    for k in ks:
        p_mean, p_std = mean_and_std(prec_logs[k])
        r_mean, r_std = mean_and_std(rec_logs[k])
        h_mean, h_std = mean_and_std(hr_logs[k])
        n_mean, n_std = mean_and_std(ndcg_logs[k])
        results["metrics"]["precision"][f"P@{k}"] = {"mean": p_mean, "std": p_std}
        results["metrics"]["recall"][f"R@{k}"] = {"mean": r_mean, "std": r_std}
        results["metrics"]["hit_rate"][f"HR@{k}"] = {"mean": h_mean, "std": h_std}
        results["metrics"]["ndcg"][f"NDCG@{k}"] = {"mean": n_mean, "std": n_std}

    # Write outputs
    os.makedirs(baseline_dir, exist_ok=True)
    out_json = os.path.join(baseline_dir, f"ranking_eval_L{level}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved evaluation to: {out_json}")

    residual_df = pd.DataFrame(residual_rows)
    out_residual = os.path.join(baseline_dir, f"hallucinations_L{level}.csv")
    residual_df.to_csv(out_residual, index=False)
    print(f"[OK] Saved residuals to: {out_residual}")


def parse_args():
    p = argparse.ArgumentParser(description="Ranking evaluation with RVQ codes")
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="Path to dataset directory (train/eval/test_data.csv, rvq/quantized_sts.csv, topics.json).")
    p.add_argument("--baseline_dir", type=str, required=True,
                   help="Path to baseline outputs directory (processed_statement.csv, quantized_sts.csv).")
    p.add_argument("--level", type=int, default=3, choices=[1, 2, 3],
                   help="RVQ code prefix length (1=stage_0, 2=stage_0..1, 3=stage_0..2).")
    p.add_argument("--k", type=str, default="1,3,5,10,20",
                   help="Comma-separated list of K for @K metrics.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ks = [int(x) for x in str(args.k).split(",") if str(x).strip().isdigit()]
    ks = sorted(set(ks))
    evaluate(
        dataset_dir=args.dataset_dir,
        baseline_dir=args.baseline_dir,
        level=args.level,
        ks=ks,
    )