# Ben Kabongo
# October 2025


import argparse
import evaluate
import json
import numpy as np
import os
import pandas as pd
import re
import sys
import torch
from typing import List, Dict, Any, Tuple

from evaluation.bart_scorer import BARTScorer
from evaluation.utils import (
    load_results, read_data,
    safe_mean_std, save_results, simple_tokenize
)


_ABBR = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.",
    "no.", "fig.", "al.", "inc.", "ltd."
}

def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter.

    Strategy:
      1) Normalize whitespace.
      2) Split on end punctuation followed by whitespace: (?<=[.!?])\\s+
         (fixed-width look-behind; safe in Python).
      3) Recombine if the previous segment ends with a known abbreviation
         (e.g., "Dr.", "e.g.", "etc.") or with a single-letter initial like "A.".
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r'(?<=[.!?])\s+', text)
    out: List[str] = []
    for seg in parts:
        if not out:
            out.append(seg)
            continue
        prev = out[-1].rstrip()
        # last word of previous segment
        prev_last = prev.split()[-1].lower() if prev.split() else ""
        is_initial = bool(re.fullmatch(r"[A-Za-z]\.", prev_last))   # e.g., "A."
        if prev_last in _ABBR or is_initial:
            # merge with current segment (it was a false split)
            out[-1] = prev + " " + seg
        else:
            out.append(seg)

    # final cleanup
    out = [s.lower().strip() for s in out if s and s.strip()]
    return out


def compute_usr_per_example(prediction: str) -> float:
    """
    USR within a single prediction:
      unique_sentences / total_sentences
    """
    sents = split_into_sentences(prediction)
    if len(sents) == 0:
        return 0.0
    unique_count = len(set(sents))
    return unique_count / float(len(sents))


def main(args):
    df = read_data(args.data_path)
    predictions: List[str] = df["prediction"].tolist()
    references: List[str] = df["reference"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: Dict[str, Any] = {
        "device": device,
        "counts": {"n_examples": len(predictions)}
    }

    if args.output_path is None:
        base, _ = os.path.splitext(args.data_path)
        args.output_path = base + "text_metrics.json"

    results = {}
    if os.path.exists(args.output_path):
        results = load_results(args.output_path)
        print(f"Loaded existing results from {args.output_path}")

    # BERTScore (per-example P/R/F1 → mean/std)
    bert_score_ok = False
    if "BERTScore" in results:
        keys = ["precision", "recall", "f1"]
        if all(k in results["BERTScore"] for k in keys):
            print("Skipping BERTScore (already computed)")
            print(json.dumps(results["BERTScore"], indent=2, ensure_ascii=False))
            bert_score_ok = True
    
    if not bert_score_ok:
        try:
            bertscore = evaluate.load("bertscore")
            bs = bertscore.compute(
                predictions=predictions,
                references=references,
                model_type=args.bertscore_model,
                device=device,
                batch_size=args.batch_size,
            )
            results["BERTScore"] = {
                "precision": safe_mean_std(bs["precision"]),
                "recall": safe_mean_std(bs["recall"]),
                "f1": safe_mean_std(bs["f1"]),
                "model": args.bertscore_model,
            }
            del bertscore, bs
        except Exception as e:
            results["BERTScore"] = {"error": str(e)}
        save_results(args.output_path, results)

    # BARTScore (per-example → mean/std)
    bart_score_ok = False
    if "BARTScore" in results:
        if "score" in results["BARTScore"]:
            print("Skipping BARTScore (already computed)")
            print(json.dumps(results["BARTScore"], indent=2, ensure_ascii=False))
            bart_score_ok = True

    if not bart_score_ok:
        try:
            bart_scorer = BARTScorer(device=device, max_length=512)
            scores = bart_scorer.score(srcs=predictions, tgts=references, batch_size=args.batch_size)
            results["BARTScore"] = {
                "score": safe_mean_std(scores),
                "model": "facebook/bart-large-cnn",
            }
            del bart_scorer, scores
        except Exception as e:
            results["BARTScore"] = {"error": str(e)}
        save_results(args.output_path, results)

    # Mauve (corpus-level)
    mauve_ok = False
    if "MAUVE" in results:
        if "mauve" in results["MAUVE"]:
            print("Skipping MAUVE (already computed)")
            print(json.dumps(results["MAUVE"], indent=2, ensure_ascii=False))
            mauve_ok = True

    if not mauve_ok:
        try:
            mauve = evaluate.load("mauve")
            mv = mauve.compute(
                predictions=predictions,
                references=references,
                device_id=0 if device == "cuda" else -1,
            )
            results["MAUVE"] = mv.mauve
            del mauve, mv
        except Exception as e:
            results["MAUVE"] = {"error": str(e)}
        save_results(args.output_path, results)

    # BLEURT (per-example → mean/std)
    bleurt_ok = False
    if "BLEURT" in results:
        if "score" in results["BLEURT"]:
            print("Skipping BLEURT (already computed)")
            print(json.dumps(results["BLEURT"], indent=2, ensure_ascii=False))
            bleurt_ok = True

    if not bleurt_ok:
        try:
            bleurt = evaluate.load("bleurt")
            bl = bleurt.compute(
                predictions=predictions,
                references=references,
            )
            scores = bl["scores"] if isinstance(bl, dict) and "scores" in bl else bl
            results["BLEURT"] = {"score": safe_mean_std(scores)}
            del bleurt, bl
        except Exception as e:
            results["BLEURT"] = {"error": str(e)}
        save_results(args.output_path, results)

    # ROUGE (corpus-level)
    rouge_ok = False
    if "ROUGE" in results:
        keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        if all(k in results["ROUGE"] for k in keys):
            print("Skipping ROUGE (already computed)")
            print(json.dumps(results["ROUGE"], indent=2, ensure_ascii=False))
            rouge_ok = True

    if not rouge_ok:
        try:
            rouge = evaluate.load("rouge")
            rg = rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True,
                rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            )
            results["ROUGE"] = {
                "rouge1": float(rg.get("rouge1")),
                "rouge2": float(rg.get("rouge2")),
                "rougeL": float(rg.get("rougeL")),
                "rougeLsum": float(rg.get("rougeLsum")),
            }
            del rouge, rg
        except Exception as e:
            results["ROUGE"] = {"error": str(e)}
        save_results(args.output_path, results)

    # BLEU-1 & BLEU-4 (corpus-level)
    bleu_ok = False
    if "BLEU" in results:
        if "bleu1" in results["BLEU"] and "bleu4" in results["BLEU"]:
            print("Skipping BLEU (already computed)")
            print(json.dumps(results["BLEU"], indent=2, ensure_ascii=False))
            bleu_ok = True

    if not bleu_ok:
        try:
            bleu = evaluate.load("bleu")
            refs = [[r] for r in references]  # one reference per example
            bleu1 = bleu.compute(predictions=predictions, references=refs, max_order=1, smooth=True)
            bleu4 = bleu.compute(predictions=predictions, references=refs, max_order=4, smooth=True)
            results["BLEU"] = {
                "bleu1": float(bleu1.get("bleu")),
                "bleu4": float(bleu4.get("bleu")),
            }
            del bleu, bleu1, bleu4
        except Exception as e:
            results["BLEU"] = {"error": str(e)}
        save_results(args.output_path, results)

    # METEOR (corpus-level)
    meteor_ok = False
    if "METEOR" in results:
        if "meteor" in results["METEOR"]:
            print("Skipping METEOR (already computed)")
            print(json.dumps(results["METEOR"], indent=2, ensure_ascii=False))
            meteor_ok = True
    
    if not meteor_ok:
        try:
            meteor = evaluate.load("meteor")
            mt = meteor.compute(predictions=predictions, references=references)
            results["METEOR"] = {"meteor": mt.get("meteor", None)}
            del meteor, mt
        except Exception as e:
            results["METEOR"] = {"error": str(e)}
        save_results(args.output_path, results)

    # USR (per-example, within each prediction)
    usr_ok = False
    if "USR_unique_sentence_ratio" in results:
        if "mean" in results["USR_unique_sentence_ratio"] and "std" in results["USR_unique_sentence_ratio"]:
            print("Skipping USR (already computed)")
            print(json.dumps(results["USR_unique_sentence_ratio"], indent=2, ensure_ascii=False))
            usr_ok = True

    if not usr_ok:
        try:
            usr_values = [compute_usr_per_example(p) for p in predictions]
            results["USR_unique_sentence_ratio"] = safe_mean_std(usr_values)
        except Exception as e:
            results["USR_unique_sentence_ratio"] = {"error": str(e)}
        save_results(args.output_path, results)

    # STS (SBERT cosine similarity on L2-normalized embeddings)
    sts_ok = False
    if "STS_SBERT_cosine" in results:
        if "score" in results["STS_SBERT_cosine"]:
            print("Skipping STS_SBERT (already computed)")
            print(json.dumps(results["STS_SBERT_cosine"], indent=2, ensure_ascii=False))
            sts_ok = True
    
    if not sts_ok:
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer(args.sts_model, device=device)
            # encode with normalization
            pred_emb = sbert.encode(predictions, batch_size=args.batch_size, convert_to_tensor=True,
                                    normalize_embeddings=True, show_progress_bar=True)
            ref_emb = sbert.encode(references, batch_size=args.batch_size, convert_to_tensor=True,
                                normalize_embeddings=True, show_progress_bar=True)
            sims = (pred_emb * ref_emb).sum(dim=1).detach().cpu().numpy().tolist()
            results["STS_SBERT_cosine"] = {
                "score": safe_mean_std(sims),
                "model": args.sts_model,
            }
            del sbert, pred_emb, ref_emb, sims
        except Exception as e:
            results["STS_SBERT_cosine"] = {"error": str(e)}
        save_results(args.output_path, results)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute text-generation metrics with Hugging Face `evaluate` + SBERT STS.")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with 'prediction' and 'reference' columns.")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save JSON results (overwritten at each step).")
    
    parser.add_argument("--bertscore_model", type=str, default="roberta-large",
                        help="Backbone for BERTScore (e.g., roberta-large, microsoft/deberta-large-mnli).")
    parser.add_argument("--sts_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SBERT model for STS cosine similarity.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model-based metrics.")
    
    args = parser.parse_args()
    main(args)