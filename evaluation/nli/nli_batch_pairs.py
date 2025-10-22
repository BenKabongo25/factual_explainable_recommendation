# Ben Kabongo
# October 2025


import argparse
import ast
import json
import os
import pandas as pd
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Tuple, Optional


try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")  # pip install spacy && python -m spacy download en_core_web_sm
except Exception:
    _NLP = None


_POV_RULES = [
    (r"\bI'm\b", "the user is"),
    (r"\bI've\b", "the user has"),
    (r"\bI'd\b", "the user would"),
    (r"\bI\b", "the user"),
    (r"\bme\b", "the user"),
    (r"\bmine\b", "the user's"),
    (r"\bmy\b", "their"),

    (r"\bwe're\b", "the user is"),
    (r"\bwe've\b", "the user has"),
    (r"\bwe'd\b", "the user would"),
    (r"\bwe\b", "the user"),
    (r"\bus\b", "the user"),
    (r"\bours\b", "the user's"),
    (r"\bour\b", "the user's"),
]

def normalize_pov(text: str) -> str:
    out = text
    for pat, rep in _POV_RULES:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[.\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


_FALLBACK_VERB_START = re.compile(
    r"^(?:has|have|is|are|was|were|be|being|been|"
    r"comes|holds|folds|works|fits|breaks|lasts|improves|keeps|provides|allows|"
    r"supports|includes|offers|requires|lacks|feels|looks|sounds|seems)\b",
    flags=re.IGNORECASE
)
_SUBJECT_PRESENT = re.compile(r"^(it|its|this|that|they|the product|the item|the board)\b", flags=re.IGNORECASE)

def starts_with_verb_spacy(text: str) -> bool:
    if _NLP is None:
        return False
    doc = _NLP(text)
    for token in doc:
        if token.is_space:
            continue
        return token.pos_ in ("VERB", "AUX")
    return False

def needs_subject(text: str) -> bool:
    if _SUBJECT_PRESENT.match(text):
        return False
    if _NLP is not None:
        return starts_with_verb_spacy(text)
    return bool(_FALLBACK_VERB_START.match(text))

def add_subject_if_needed(text: str) -> str:
    return text if not needs_subject(text) else f"it {text}"


def canonicalize_clause(raw: str) -> str:
    s = normalize_whitespace(raw)
    s = normalize_pov(s)
    s = add_subject_if_needed(s)
    return s

def as_explanation_sentence(statement: str, sentiment: str) -> str:
    """Wrap an atomic statement clause into an explanation sentence, per sentiment."""
    clause = canonicalize_clause(statement)
    if sentiment == "positive":
        return f"The user would appreciate this product because {clause}."
    elif sentiment == "negative":
        return f"The user may dislike that {clause}."
    else:  # neutral
        return f"They seem indifferent to {clause}."


def read_sts(path: str) -> pd.DataFrame:
    """
    Reads a statements CSV with columns: statement, topic, sentiment, frequency,
    and a leading index column (SID).
    """
    df = pd.read_csv(path)
    # If first column is an unnamed index, treat as SID
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "sid"})
    # If not present, create 'sid' from row index
    if "sid" not in df.columns:
        df.insert(0, "sid", df.index.astype(int))
    # Normalize
    for col in ("topic", "sentiment"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    df["statement"] = df["statement"].astype(str)
    return df[["sid", "statement", "topic", "sentiment"]].copy()

def read_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def parse_list_field(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        # Try JSON as fallback
        try:
            return json.loads(x)
        except Exception:
            return []


def normalize_sentiment(x: str) -> str:
    x = (x or "").strip().lower()
    if x in {"pos", "positive", "1", "+", "plus"}:
        return "positive"
    if x in {"neg", "negative", "-1", "-", "minus"}:
        return "negative"
    return "neutral"

def build_ref_items_for_row(row, ref_lookup: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list of reference statements present in this review row."""
    sids = parse_list_field(row.get("statement_ids", "[]"))
    out = []
    for sid in sids:
        try:
            sid_int = int(sid)
        except Exception:
            continue
        ref_obj = ref_lookup.get(sid_int)
        if not ref_obj:
            continue
        out.append({
            "ref_sid": sid_int,
            "statement": ref_obj["statement"],
            "topic": ref_obj["topic"],
            "sentiment": ref_obj["sentiment"],
        })
    return out

def build_pred_items_for_row(row, sts_pred_lookup: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list of predicted statements for this review row."""
    sids = parse_list_field(row.get("statements_ids", "[]"))
    out = []
    for sid in sids:
        try:
            sid_int = int(sid)
        except Exception:
            continue
        pred_obj = sts_pred_lookup.get(sid_int) if sts_pred_lookup else None
        if not pred_obj:
            continue
        out.append({
            "pred_sid": sid_int,
            "statement": pred_obj["statement"],
            "topic": pred_obj["topic"],
            "sentiment": pred_obj["sentiment"],
        })
    return out


def build_label_order(model) -> List[str]:
    """
    Return labels in the canonical order: ['entailment', 'neutral', 'contradiction'].
    Use model.config.id2label to map indices.
    """
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        id2label = {0: "entailment",  1: "neutral", 2: "contradiction"}
    norm = {i: str(lab).lower() for i, lab in id2label.items()}
    order = ["entailment", "neutral", "contradiction"]
    present = set(norm.values())
    missing = set(order) - present
    if missing:
        raise ValueError(f"Model labels missing expected classes: {missing}. Got {present}")
    # Create index list: for each of order, find its index id
    idx_of = {lab: i for i, lab in norm.items()}
    return [order[0], order[1], order[2]], [idx_of[order[0]], idx_of[order[1]], idx_of[order[2]]]

def run_nli_batch(
    tokenizer,
    model,
    device: str,
    pairs: List[Tuple[str, str]],
    batch_size: int,
    max_length: int
) -> List[Dict[str, float]]:
    """
    pairs: list of (premise, hypothesis)
    returns: list of dicts with keys: entailment, neutral, contradiction
    """
    model.eval()
    _, idx_order = build_label_order(model)  # desired order indices in model output
    results: List[Dict[str, float]] = []

    for start in tqdm(range(0, len(pairs), batch_size), desc="NLI batches", leave=False):
        batch = pairs[start:start + batch_size]
        enc = tokenizer(
            [p for p, _ in batch],
            [h for _, h in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits  # [B, 3]
            probs = torch.softmax(logits, dim=-1)  # [B, 3]
        probs = probs[:, idx_order]  # reorder to [entailment, neutral, contradiction]
        probs = probs.detach().cpu().tolist()
        for e, n, c in probs:
            results.append({
                "entailment": float(e),
                "neutral": float(n),
                "contradiction": float(c),
                "argmax": ["entailment", "neutral", "contradiction"][int(max(range(3), key=[e, n, c].__getitem__))]
            })
    return results

# ----------------------------- Main pipeline -----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load statements
    print("[INFO] Loading statement repositories...")
    sts_ref = read_sts(args.sts_ref_path)
    sts_pred = read_sts(args.sts_pred_path)

    ref_lookup = {int(r.sid): {"statement": r.statement, "topic": r.topic, "sentiment": normalize_sentiment(r.sentiment)}
                  for r in sts_ref.itertuples(index=False)}
    pred_lookup = {int(r.sid): {"statement": r.statement, "topic": r.topic, "sentiment": normalize_sentiment(r.sentiment)}
                   for r in sts_pred.itertuples(index=False)}

    # Load per-review dataframes
    print("[INFO] Loading per-review datasets...")
    ref_df = read_df(args.ref_data_path)
    pred_df = read_df(args.pred_data_path)

    if len(ref_df) != len(pred_df):
        print(f"[WARN] ref_data rows ({len(ref_df)}) != pred_data rows ({len(pred_df)}). "
              f"Will align by position up to min length.")
    n_rows = min(len(ref_df), len(pred_df))

    # Prepare model & tokenizer
    print(f"[INFO] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)
    _ = build_label_order(model)  # validate labels

    # Build all (topic, sentiment)-matched pairs across rows
    # We'll accumulate meta + text for NLI in two directions.
    print("[INFO] Building matched pairs (complete bipartite per row, same topic & sentiment)...")
    pair_meta: List[Dict[str, Any]] = []
    pairs_pred_to_ref: List[Tuple[str, str]] = []  # (premise=p, hypothesis=r)
    pairs_ref_to_pred: List[Tuple[str, str]] = []  # (premise=r, hypothesis=p)

    for i in tqdm(range(n_rows), desc="Scanning rows"):
        ref_row = ref_df.iloc[i]
        pred_row = pred_df.iloc[i]

        # Reference statements present in this review
        ref_items = build_ref_items_for_row(ref_row, ref_lookup)

        # Predicted statements for this review
        pred_items = build_pred_items_for_row(pred_row, pred_lookup)

        if not ref_items or not pred_items:
            continue

        # Form all pairs where topic & sentiment match
        for p in pred_items:
            for r in ref_items:
                if (p["topic"] == r["topic"]) and (p["sentiment"] == r["sentiment"]):
                    p_sent = as_explanation_sentence(p["statement"], p["sentiment"])
                    r_sent = as_explanation_sentence(r["statement"], r["sentiment"])

                    # Save meta
                    meta = {
                        "row_index": i,
                        "user_id": ref_row.get("user_id", None),
                        "item_id": ref_row.get("item_id", None),
                        "topic": r["topic"],
                        "sentiment": r["sentiment"],
                        "ref_sid": r["ref_sid"],
                        "pred_sid": p.get("pred_sid", None),
                        "pred_local_id": p.get("pred_local_id", None),
                        "pred_statement": p["statement"],
                        "ref_statement": r["statement"],
                        "pred_sentence": p_sent,
                        "ref_sentence": r_sent,
                    }
                    pair_meta.append(meta)
                    pairs_pred_to_ref.append((p_sent, r_sent))
                    pairs_ref_to_pred.append((r_sent, p_sent))

    if not pair_meta:
        print("[INFO] No (topic, sentiment)-matched pairs were found. Exiting.")
        return

    print(f"[INFO] Total matched pairs: {len(pair_meta)}")

    # Run NLI in batches for both directions
    print("[INFO] Running NLI (pred → ref)...")
    scores_p2r = run_nli_batch(
        tokenizer, model, device,
        pairs=pairs_pred_to_ref,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    print("[INFO] Running NLI (ref → pred)...")
    scores_r2p = run_nli_batch(
        tokenizer, model, device,
        pairs=pairs_ref_to_pred,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    assert len(scores_p2r) == len(pair_meta) == len(scores_r2p)

    # Assemble output dataframe
    out_rows = []
    for meta, s1, s2 in zip(pair_meta, scores_p2r, scores_r2p):
        out_rows.append({
            # identification
            "row_index": meta["row_index"],
            "user_id": meta["user_id"],
            "item_id": meta["item_id"],
            "topic": meta["topic"],
            "sentiment": meta["sentiment"],

            # ids
            "ref_sid": meta["ref_sid"],
            "pred_sid": meta["pred_sid"],
            "pred_local_id": meta["pred_local_id"],

            # raw atomic statements
            "pred_statement": meta["pred_statement"],
            "ref_statement": meta["ref_statement"],

            # canonicalized sentences used for NLI
            "pred_sentence": meta["pred_sentence"],
            "ref_sentence": meta["ref_sentence"],

            # pred -> ref
            "p2r_entail": s1["entailment"],
            "p2r_neutral": s1["neutral"],
            "p2r_contradiction": s1["contradiction"],
            "p2r_label": s1["argmax"],

            # ref -> pred
            "r2p_entail": s2["entailment"],
            "r2p_neutral": s2["neutral"],
            "r2p_contradiction": s2["contradiction"],
            "r2p_label": s2["argmax"],
        })

    out_df = pd.DataFrame(out_rows)

    # Save next to pred_data
    pred_dir = os.path.dirname(os.path.abspath(args.pred_data_path))
    model_tag = os.path.basename(args.model_name).replace("/", "_")
    out_path = os.path.join(pred_dir, f"nli_scores_{model_tag}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved NLI scores to: {out_path}")
    print(f"[INFO] Rows: {len(out_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch NLI scoring for (topic, sentiment)-matched statement pairs.")
    parser.add_argument("--sts_ref_path", type=str, required=True, help="Path to reference statements CSV")
    parser.add_argument("--sts_pred_path", type=str, required=True, help="Path to predicted statements CSV")
    parser.add_argument("--ref_data_path", type=str, required=True, help="Path to ref_data CSV (per-review; contains reference SIDs)")
    parser.add_argument("--pred_data_path", type=str, required=True, help="Path to pred_data CSV (per-review; contains predicted statements or SIDs)")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large-mnli", help="HF model name or path for NLI")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for NLI inference")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for tokenizer truncation")
    parser.add_argument("--device", type=str, default=None, help="Force device: 'cuda' or 'cpu' (default: auto)")

    args = parser.parse_args()
    main(args)
