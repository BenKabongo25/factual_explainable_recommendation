# [Data](https://huggingface.co/datasets/benkabongo25/amazon-reviews-statement-v0)

We provide **five augmented datasets** derived from the Amazon Reviews 2014 collection: **Toys**, **Clothes**, **Beauty**, **Sports**, and **Cellphones**.
Each user–item interaction (rating + review) is paired with:

* **atomic statement–topic–sentiment (STS) triplets** extracted from the review,
* a **ground-truth explanation** constructed by aggregating *all* extracted statements in a rule-based manner (no LLM generation at this step), and
* domain **topics** (10 topics per domain).

Our augmented dataset is available here: [https://huggingface.co/datasets/benkabongo25/amazon-reviews-statement-v0](https://huggingface.co/datasets/benkabongo25/amazon-reviews-statement-v0)

---

## Repository structure

Each domain lives in its own subfolder at the root:

```
.
├─ Beauty/
├─ Cell/
├─ Clothes/
├─ Sports/
└─ Toys/
```

Inside each domain (example: `Toys/`):

```
Toys/
├─ data.csv                      # processed dataset (one row per interaction)
├─ topics.json                   # list[str] of 10 domain topics
├─ sts.csv                       # statement vocabulary & frequency
├─ metadata.json                 # Amazon item metadata (JSON per line)
├─ train_data.csv                # train split
├─ eval_data.csv                 # validation split
├─ test_data.csv                 # test split
├─ train_explanations.csv        # GT explanation per train example
├─ eval_explanations.csv         # GT explanation per val example
└─ test_explanations.csv         # GT explanation per test example
```

---

## Files & schemas

### `data.csv` (processed dataset)

Typical columns (some may be absent depending on the domain):

* `helpful` *(str list → list[int])* – e.g., `"[0, 0]"`
* `reviewTime` *(str)* – e.g., `"01 29, 2014"`
* `rating` *(float)*
* `timestamp` *(int)* – Unix epoch seconds
* `review` *(str)* – full review text
* `user_name` *(str)*, `user_id` *(str)*, `item_id` *(str)*
* `review_title` *(str)*
* `statements` *(str list of dicts → list[ {statement, topic, sentiment} ])*
* `statement_ids` *(str list → list[int])* – aligned with `statements`
* `topic_ids` *(str list → list[int])* – aligned with `statements`
* `sentiments` *(str list → list[int])* – aligned with `statements`

### Split files: `train_data.csv`, `eval_data.csv`, `test_data.csv`

Columns:

* `user_id` *(str)*, `item_id` *(str)*, `timestamp` *(int)*, `rating` *(float)*
* `statement_ids` *(str list → list[int])*
* `topic_ids` *(str list → list[int])*
* `sentiments` *(str list → list[int])*
* `review` *(str)*

### Ground-truth explanations: `*_explanations.csv`

Single column:

* `explanation` *(str)* – paragraph built by aggregating all extracted statements by sentiment:

  * positive: “The user would appreciate this product because …”
  * negative: “However, they may dislike that …”
  * neutral: “They seem indifferent to …”

### Topics: `topics.json`

A JSON array of 10 domain-specific topics, e.g. for Toys: `["age", "assembly", ..., "safety"]`.

### Statement vocabulary: `sts.csv`

Columns:

* `statement` *(str)* – canonical statement text
* `topic` *(str)* – one of the domain topics
* `sentiment` *(str)* – `positive|negative|neutral`
* `frequency` *(int)* – count in the domain

### Item metadata: `metadata.json`

**JSON Lines**: one JSON object per line (e.g., `{ "asin": "...", "title": "...", ... }`).

---

## Splits & construction

* **Temporal split per user**: retain users with ≥ 5 interactions. For each user, sort interactions by time and split **80% / 10% / 10%** into **train / validation / test**.
* **Cold-start filtering**: remove from validation/test items unseen during training (for methods requiring seen items).
* **Ground-truth generation**: from the review’s extracted STS triplets, compose the explanation with a **rule-based aggregator** so **all statements** are preserved (no truncation losses).

---

## Dataset statistics

### Interaction-level statistics

| Metric       | Toys    | Clothes | Beauty  | Sports  | Cellphones |
| ------------ | ------- | ------- | ------- | ------- | ---------- |
| Users        | 19,398  | 39,385  | 22,362  | 35,596  | 27,873     |
| Items        | 11,924  | 23,033  | 12,101  | 18,357  | 10,429     |
| Interactions | 163,711 | 274,774 | 197,621 | 293,244 | 190,194    |
| Train        | 121,751 | 203,574 | 149,569 | 219,913 | 139,889    |
| Validation   | 14,805  | 24,396  | 18,506  | 27,394  | 16,099     |
| Test         | 22,441  | 41,995  | 27,862  | 42,675  | 28,901     |

### Statement-level statistics

| Metric              | Toys    | Clothes   | Beauty    | Sports    | Cellphones |
| ------------------- | ------- | --------- | --------- | --------- | ---------- |
| Avg per interaction | 5.03    | 4.42      | 5.45      | 4.93      | 4.54       |
| Avg per user        | 41.76   | 30.12     | 46.99     | 40.24     | 30.65      |
| Avg per item        | 67.49   | 50.70     | 84.79     | 76.90     | 81.42      |
| Unique              | 587,114 | 619,917   | 622,276   | 1,055,145 | 662,466    |
| Total               | 823,932 | 1,215,270 | 1,076,769 | 1,447,240 | 863,036    |

---

## How to open the data

Below are **robust** loading snippets for both **pandas** and **🤗 Datasets**, including **pre-processing**.

### 1) Pandas

```python
import ast
import json
import pandas as pd
from pathlib import Path

DOMAIN = "Toys"  # or "Clothes", "Beauty", "Sports", "Cell"
root = Path(".")  # repository root
D = root / DOMAIN

# --- Utilities ---
parse_list = lambda s: [] if pd.isna(s) or s == "" else (ast.literal_eval(s) if isinstance(s, str) else list(s))

def parse_statements_cell(x):
    """Turn a stringified list of dicts into a list of dicts with stable keys."""
    if pd.isna(x) or x == "":
        return []
    val = ast.literal_eval(x) if isinstance(x, str) else x
    out = []
    for d in val:
        out.append({
            "statement": str(d.get("statement", "")),
            "topic": str(d.get("topic", "")),
            "sentiment": str(d.get("sentiment", "")),
        })
    return out

# --- data.csv (processed) ---
proc = pd.read_csv(D / "data.csv")
# drop leading index column if present
if proc.columns[0].startswith("Unnamed") or proc.columns[0] == "":
    proc = proc.drop(columns=[proc.columns[0]])

# cast types / parse stringified lists
for col in ["statement_ids", "topic_ids", "sentiments", "helpful"]:
    if col in proc.columns:
        proc[col] = proc[col].apply(parse_list)
if "statements" in proc.columns:
    proc["statements"] = proc["statements"].apply(parse_statements_cell)

# --- splits ---
train = pd.read_csv(D / "train_data.csv")
eval_ = pd.read_csv(D / "eval_data.csv")
test = pd.read_csv(D / "test_data.csv")
for df in (train, eval_, test):
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df.drop(columns=[df.columns[0]], inplace=True)
    for col in ["statement_ids", "topic_ids", "sentiments"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)

# --- ground-truth explanations ---
train_exp = pd.read_csv(D / "train_explanations.csv")
eval_exp = pd.read_csv(D / "eval_explanations.csv")
test_exp = pd.read_csv(D / "test_explanations.csv")

# --- topics ---
with open(D / "topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

# --- statement vocabulary ---
sts = pd.read_csv(D / "sts.csv")  # [statement, topic, sentiment, frequency]

# Quick peek
print(proc.head(1).to_dict())
print(train.head(1).to_dict())
print(topics[:3])
print(sts.head(3).to_dict(orient="records"))
```

### 2) 🤗 Datasets (CSV backend + post-processing)

```python
from datasets import load_dataset
import ast, json

REPO = "benkabongo25/amazon-reviews-statement-v0"
DOMAIN = "Toys"  # "Clothes" | "Beauty" | "Sports" | "Cell"

# Load split CSVs
files = {
    "train":      f"hf://datasets/{REPO}/{DOMAIN}/train_data.csv",
    "validation": f"hf://datasets/{REPO}/{DOMAIN}/eval_data.csv",
    "test":       f"hf://datasets/{REPO}/{DOMAIN}/test_data.csv",
}
raw = load_dataset("csv", data_files=files)

# Post-process stringified lists to real lists

def parse_list_str(example, cols=("statement_ids", "topic_ids", "sentiments")):
    for c in cols:
        if c in example and isinstance(example[c], str):
            example[c] = ast.literal_eval(example[c]) if example[c] else []
    return example

raw = raw.map(parse_list_str)
print(raw)

# Load ground-truth explanations
gte = load_dataset("csv", data_files={"test": f"hf://datasets/{REPO}/{DOMAIN}/test_explanations.csv"})
print(gte)
```

### 3) Optional: structured Features (advanced)

If you need strict schemas, you can **cast** features after loading:

```python
from datasets import Features, Value, Sequence

features = Features({
    "user_id": Value("string"),
    "item_id": Value("string"),
    "timestamp": Value("int64"),
    "rating": Value("float32"),
    "statement_ids": Sequence(Value("int64")),
    "topic_ids": Sequence(Value("int64")),
    "sentiments": Sequence(Value("int32")),
    "review": Value("string"),
})

raw = raw.cast(features)
```