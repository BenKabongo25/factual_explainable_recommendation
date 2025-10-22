# Ben Kabongo
# October 2025

import argparse
import pandas as pd
import re
import unicodedata
from typing import List, Set


ARTIFACT_TOKENS = [
    '[/INST]', '[INST]',
    '<s>', '</s>',
    '<|endoftext|>', '<|assistant|>', '<|user|>', '<|system|>',
]

INVISIBLE_CHARS = [
    '\ufeff',               # BOM
    '\u200b', '\u200c', '\u200d',  # zero-width (ZWSP, ZWNJ, ZWJ)
    '\u2060',               # word-joiner
    '\u00ad',               # soft hyphen
]

# Regex helpers
RE_MULTI_SPACE   = re.compile(r'[ \t]{2,}')
RE_MULTI_PUNCT   = re.compile(r'([,.!?])\1{1,}')      # "!!" -> "!"
RE_SPACE_PUNCT   = re.compile(r'\s+([,.!?;:])')       # "word ," -> "word,"
RE_PUNCT_SPACE   = re.compile(r'([,.!?;:])([^\s])')   # "word,word" -> "word, word"
RE_SENT_SPLIT    = re.compile(r'(?<=[\.\?!])\s+|\n+') # split on end punctuation or newlines
RE_ORPHAN_BRACKS = re.compile(r'(?:(?<=\s)|^)[\[\]]+(?=\s|$)')

# Clause separators (used only for exact-duplicate collapse, not deletion)
CLAUSE_SEPS = [', and ', '; and ', ' and ',
               ', but ', '; but ', ' but ',
               ', however, ', '; however, ']

def nfkc(s: str) -> str:
    return unicodedata.normalize('NFKC', s)

def strip_controls(s: str) -> str:
    # Keep \n and \r; drop other control/format characters
    return ''.join(ch for ch in s if ch in ('\n', '\r') or unicodedata.category(ch) not in ('Cc', 'Cf'))

def normalize_whitespace_and_punct(s: str) -> str:
    s = RE_MULTI_PUNCT.sub(r'\1', s)
    s = RE_SPACE_PUNCT.sub(r'\1', s)
    s = RE_PUNCT_SPACE.sub(r'\1 \2', s)
    # Trim right spaces per line, then collapse multi-spaces
    s = '\n'.join(line.rstrip() for line in s.splitlines())
    s = RE_MULTI_SPACE.sub(' ', s)
    return s.strip()

def remove_artifacts(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = nfkc(text)
    for tok in ARTIFACT_TOKENS:
        text = text.replace(tok, '')
    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, '')
    text = strip_controls(text)
    text = RE_ORPHAN_BRACKS.sub(' ', text)
    return normalize_whitespace_and_punct(text)

def sentence_key(s: str) -> str:
    # Key for exact-duplicate detection: lowercase, collapse spaces, strip trivial punct
    k = nfkc(s).lower().strip()
    k = re.sub(r'\s+', ' ', k)
    # keep punctuation minimal in key to avoid false diffs due to commas/periods
    k = re.sub(r'[,.!?;:]+', '', k)
    return k

def dedupe_exact(items: List[str]) -> List[str]:
    """Keep first occurrence of each exact-normalized item; preserve order."""
    seen: Set[str] = set()
    out: List[str] = []
    for it in items:
        key = sentence_key(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def split_sentences(text: str) -> List[str]:
    # Split on sentence boundaries OR line breaks; keep order
    parts = [p.strip() for p in RE_SENT_SPLIT.split(text)]
    return [p for p in parts if p]  # do not drop short ones; just remove pure empties

def split_clauses_once(sentence: str):
    # Find the first matching separator appearing in the sentence, prefer longer tokens
    for sep in sorted(CLAUSE_SEPS, key=len, reverse=True):
        if sep in sentence:
            return [p.strip() for p in sentence.split(sep)], sep
    # fallback: split on comma only for dedupe; we will re-join with ", "
    return [p.strip() for p in sentence.split(',')], ', '

def dedupe_clauses_in_sentence(sentence: str) -> str:
    clauses, sep = split_clauses_once(sentence)
    if len(clauses) <= 1:
        return sentence  # nothing to dedupe
    clauses = [c for c in clauses if c]  # keep all non-empty clauses
    clauses = dedupe_exact(clauses)      # collapse exact duplicates only
    rebuilt = sep.join(clauses)
    return normalize_whitespace_and_punct(rebuilt)

def clean_prediction(text: str) -> str:
    """
    Safe clean:
      1) remove artifacts/invisible chars
      2) split into sentences (no deletion)
      3) exact-duplicate collapse at sentence level
      4) per-sentence exact-duplicate clause collapse
      5) re-join sentences with a space
    No content filtering; preserves first occurrence of every piece of content.
    """
    text = remove_artifacts(text)
    if not text:
        return ""

    sentences = split_sentences(text)

    # Collapse exact duplicate sentences (keep first)
    sentences = dedupe_exact(sentences)

    # For each sentence, collapse exact duplicate clauses (keep first)
    cleaned_sents: List[str] = []
    for s in sentences:
        s2 = dedupe_clauses_in_sentence(s)
        cleaned_sents.append(s2)

    # Final tidy (spacing/punct), but no forced trailing punctuation
    out = ' '.join(cleaned_sents)
    out = normalize_whitespace_and_punct(out)
    return out

def process_file(infile: str, outfile: str, col: str = "prediction"):
    df = pd.read_csv(infile)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {infile}. Available columns: {list(df.columns)}")
    # Map clean function WITHOUT dropping any row
    df[col] = df[col].astype(str).map(clean_prediction)
    df.to_csv(outfile, index=False)
    print(f"[OK] Cleaned {len(df)} rows -> {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean XRec predictions safely (no row/meaningful content removal).")
    parser.add_argument("--infile", required=True, help="Input CSV path")
    parser.add_argument("--outfile", required=True, help="Output CSV path")
    parser.add_argument("--col", default="prediction", help="Column to clean (default: prediction)")
    args = parser.parse_args()
    process_file(args.infile, args.outfile, args.col)
