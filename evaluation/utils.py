# Ben Kabongo
# October 2025


import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prediction" not in df.columns or "reference" not in df.columns:
        raise ValueError("Input file must contain 'prediction' and 'reference' columns.")
    df["prediction"] = df["prediction"].astype(str)
    df["reference"] = df["reference"].astype(str)
    return df


def simple_tokenize(text: str) -> List[str]:
    return text.strip().split()


def safe_mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def save_results(path: str, results: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)