# Ben Kabongo
# October 2025


import argparse
import ast
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional


try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")  # pip install spacy && python -m spacy download en_core_web_sm
except Exception:
    _NLP = None

# ----------------------------- POV normalizer -----------------------------
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

# ----------------------------- Clause cleaning -----------------------------
def normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[.\s]+$", "", text)   # strip trailing spaces and periods
    text = re.sub(r"\s+", " ", text)
    return text

# ----------------------------- Verb-first detection -----------------------------
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

# ----------------------------- List join -----------------------------
def join_clauses(clauses: List[str]) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + ", and " + clauses[-1]

# ----------------------------- Statement canonicalization -----------------------------
def canonicalize_clause(raw: str) -> str:
    s = normalize_whitespace(raw)
    s = normalize_pov(s)
    s = add_subject_if_needed(s)
    return s

# ----------------------------- Paragraph generator (reader-style) -----------------------------
def make_reader_paragraph(
    stmts: List[Tuple[str, str, str]],  # (statement, topic, sentiment_str)
) -> str:
    """
    Build a 'reader-style' paragraph from atomic statements with strict, rule-based rendering.
    - Keep original appearance order within each polarity group (input order is assumed to reflect appearance).
    - First sentence uses "The user ..." and no "Overall,".
    - Subsequent blocks use connectors.
    - No paraphrasing beyond POV normalization and subject injection.
    """
    # Group by sentiment and deduplicate by canonical text (preserving order)
    buckets: Dict[str, List[str]] = defaultdict(list)
    seen = set()

    for statement, topic, sentiment in stmts:
        can = canonicalize_clause(statement)
        key = (can.lower(), sentiment)
        if key in seen:
            continue
        seen.add(key)
        buckets[sentiment].append(can)

    pos = buckets.get("positive", [])
    neg = buckets.get("negative", [])
    neu = buckets.get("neutral",  [])

    order = []
    if pos: order.append(("pos", pos))
    if neg: order.append(("neg", neg))
    if neu: order.append(("neu", neu))
    if not order:
        return ""

    # First sentence: always "The user ..."
    first_kind, first_clauses = order[0]
    if first_kind == "pos":
        first_sent = f"The user would appreciate this product because {join_clauses(first_clauses)}."
    elif first_kind == "neg":
        first_sent = f"The user may dislike that {join_clauses(first_clauses)}."
    else:
        first_sent = f"The user seems indifferent to {join_clauses(first_clauses)}."
    sentences = [first_sent]

    # Subsequent sentences (connectors; “they” allowed)
    for kind, clauses in order[1:]:
        if kind == "pos":
            sentences.append(f"Additionally, they would appreciate this product because {join_clauses(clauses)}.")
        elif kind == "neg":
            sentences.append(f"However, they may dislike that {join_clauses(clauses)}.")
        else:
            sentences.append(f"They seem indifferent to {join_clauses(clauses)}.")

    return " ".join(sentences)

# ----------------------------- Helpers -----------------------------
def safe_parse_list(cell: Optional[str]) -> List:
    """
    Parse a stringified Python list like "[1, 2]" or "['a','b']" safely.
    Returns [] on empty/invalid.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or str(cell).strip() == "":
        return []
    try:
        val = ast.literal_eval(cell)
        if isinstance(val, list):
            return val
        return []
    except Exception:
        return []

SENT_MAP = {1: "positive", -1: "negative", 0: "neutral"}

# ----------------------------- Main pipeline -----------------------------
def build_explanations(
    reviews_df: pd.DataFrame,
    sts_df: pd.DataFrame,
    col_stmt_ids: str = "statement_ids",
    col_topic_ids: str = "topic_ids",
    col_sent_ids: str = "sentiments",
) -> pd.DataFrame:
    explanations: List[str] = []

    for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df), desc="Building explanations"):
        stmt_ids = safe_parse_list(row.get(col_stmt_ids))
        sent_ids = safe_parse_list(row.get(col_sent_ids))
        # topics can be taken from STS; topic_ids in reviews are optional for generation
        tuples: List[Tuple[str, str, str]] = []

        # Fallback: if sentiments length mismatches, we’ll take from STS row sentiment
        use_sent_from_reviews = len(stmt_ids) == len(sent_ids) and len(stmt_ids) > 0

        for i, sid in enumerate(stmt_ids):
            if sid not in sts_df.index:
                continue
            st_row = sts_df.loc[sid]
            statement_text = str(st_row["statement"])
            topic_text = str(st_row["topic"])

            if use_sent_from_reviews:
                sent_str = SENT_MAP.get(int(sent_ids[i]), "neutral")
            else:
                sent_str = str(st_row["sentiment"])

            tuples.append((statement_text, topic_text, sent_str))

        explanation = make_reader_paragraph(tuples) if tuples else ""
        explanations.append(explanation)

    out_df = pd.DataFrame({"explanation": explanations})
    return out_df


def main(args):
    reviews_df = pd.read_csv(args.data_csv)
    
    sts_df = pd.read_csv(args.sts_csv, index_col=0)
    required_cols = {"statement", "topic", "sentiment"}
    missing = required_cols - set(sts_df.columns)
    if missing:
        raise ValueError(f"STS dataframe is missing required columns: {missing}")

    out_df = build_explanations(
        reviews_df,
        sts_df,
        col_stmt_ids=args.col_statement_ids,
        col_topic_ids=args.col_topic_ids,
        col_sent_ids=args.col_sentiments,
    )
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out_df)} rows to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reader-style explanations from atomic statements.")
    parser.add_argument("--data_csv", type=str, required=True,
        help="Path to the reviews CSV with columns: statement_ids,topic_ids,sentiments")
    parser.add_argument("--sts_csv", type=str, required=True,
        help="Path to the STS CSV indexed by statement id with columns: statement,topic,sentiment")
    parser.add_argument("--output_csv", type=str, required=True, 
        help="Path to write the CSV with an added 'explanation' column")
    
    parser.add_argument("--col_statement_ids", type=str, default="statement_ids",
        help="Column name for the list of statement ids in the reviews CSV")
    parser.add_argument("--col_topic_ids", type=str, default="topic_ids",
        help="Column name for the list of topic ids in the reviews CSV (not required for generation)")
    parser.add_argument("--col_sentiments", type=str, default="sentiments",
        help="Column name for the list of sentiment ids in the reviews CSV (1 / -1 / 0)")
    
    args = parser.parse_args()
    main(args)
