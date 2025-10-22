# Ben Kabongo
# October 2025


import ast
import json
import pandas as pd
import re

from typing import Any, List


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
