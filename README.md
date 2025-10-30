# Factual Explainable Recommendation

This repository provides a comprehensive framework for evaluating the **factual consistency** of text-based explainable recommendation models. It includes statement-level evaluation metrics, augmented benchmark datasets, and baseline implementations.

---

## 📄 Paper

**[On the Factual Consistency of Text-based Explainable Recommendation Models](KABONGO_GUIGUE_factual_explainable_recommendation.pdf)**  
*Ben Kabongo, Vincent Guigue*

Text-based explainable recommendation aims to generate natural-language explanations that justify item recommendations. While recent models produce fluent outputs, this work reveals a critical gap: **high surface-level quality doesn't guarantee factual accuracy**. We introduce a framework to evaluate factual consistency at the statement level and show that state-of-the-art models exhibit substantial hallucination, with precision ranging from 4.38% to 32.88%.

---

## 🗂️ Repository Structure

```
.
├── baselines/              # Reference implementations & output processing
├── data/                   # Augmented datasets (5 Amazon domains)
├── evaluation/             # Evaluation metrics (LLM, NLI, QG-QA, text similarity)
├── statement_topic_sentiment/  # Statement-Topic-Sentiment extraction pipeline
└── README.md
```

---

## 🎯 Key Contributions

1. **Statement-Level Ground-truth Construction**: LLM-based pipeline to extract atomic explanatory statements with domain-specific topics and sentiment labels from reviews
2. **Augmented Benchmark Datasets**: Five Amazon Reviews categories (Toys, Clothes, Beauty, Sports, Cellphones) with statement-level annotations
3. **Factuality Metrics**: Comprehensive suite combining LLM-based and NLI-based approaches to assess precision (factual consistency) and recall (coverage)

---

## 📊 Datasets

We provide **five augmented datasets** from Amazon Reviews 2014: **Toys**, **Clothes**, **Beauty**, **Sports**, and **Cellphones**.

Each interaction includes:
- **Atomic statement–topic–sentiment (STS) triplets** extracted from reviews
- **Ground-truth explanations** constructed by rule-based aggregation (no LLM generation)
- **Domain-specific topics** (10 topics per domain)

**Dataset:** [https://huggingface.co/datasets/benkabongo25/amazon-reviews-statement-v0](https://huggingface.co/datasets/benkabongo25/amazon-reviews-statement-v0)

**Statistics:**

| Dataset    | Users  | Items  | Interactions | Avg Statements/Interaction |
|------------|--------|--------|--------------|---------------------------|
| Toys       | 19,398 | 11,924 | 163,711      | 5.03                      |
| Clothes    | 39,385 | 23,033 | 274,774      | 4.42                      |
| Beauty     | 22,362 | 12,101 | 197,621      | 5.45                      |
| Sports     | 35,596 | 18,357 | 293,244      | 4.93                      |
| Cellphones | 27,873 | 10,429 | 190,194      | 4.54                      |

For detailed dataset documentation, see [`data/README.md`](data/README.md).

---

## 🔧 Baselines

We evaluate **six state-of-the-art models** spanning three architectural families:

- **RNN-based**: Att2Seq, NRT
- **Transformer-based**: PETER, CER, PEPLER
- **LLM-enhanced**: XRec (LightGCN + LLM)

For training details and hyperparameters, see [`baselines/README.md`](baselines/README.md).

---

## 📈 Evaluation

Our framework includes multiple evaluation approaches:

1. **LLM-based Statement Metrics**: St2Exp-P/R/F1 (precision, recall, F1)
2. **NLI-based Statement Metrics**: StEnt-*, StCoh-* (entailment, coherence)
3. **Standard NLI Metrics**: SummaC, AlignScore
4. **QG-QA Metrics**: QuestEval
5. **Text Similarity**: BERTScore, STS, BARTScore, BLEURT

For complete evaluation protocols, see [`evaluation/README.md`](evaluation/README.md).

---

## 🚀 Quick Start

### 1. Extract Statement-Topic-Sentiment Triplets

```bash
cd statement_topic_sentiment

PYTHONPATH=. python sts_extraction.py \
  --model /path/to/Meta-Llama-3-8B-Instruct \
  --prompt_text_file prompts/Toys/toys.txt \
  --dataset_path /path/to/reviews.json \
  --output_dir /path/to/output/ \
  --batch_size 32 \
  --max_new_tokens 512
```

See [`statement_topic_sentiment/README.md`](statement_topic_sentiment/README.md) for adapting to new domains.

### 2. Train a Baseline Model

Example with PETER:

```bash
PYTHONPATH=. python3 baselines/PETER/main.py \
    --dataset_name Toys \
    --dataset_dir /path/to/Toys/ \
    --save_dir /path/to/checkpoints/ \
    --emsize 512 \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

### 3. Post-process Model Outputs

```bash
# Clean predictions
PYTHONPATH=. python baselines/output_process/clean.py \
  --infile ${OUTPUT_DIR}/output.csv \
  --outfile ${OUTPUT_DIR}/output.csv

# Extract statements from predictions
PYTHONPATH=. python baselines/output_process/sts_extraction.py \
  --model /path/to/Meta-Llama-3-8B-Instruct \
  --prompt_text_file statement_topic_sentiment/prompts/toys.txt \
  --dataset_path ${OUTPUT_DIR}/output.csv \
  --output_dir ${OUTPUT_DIR}
```

See [`baselines/output_process/README.md`](baselines/output_process/README.md) for details.

### 4. Evaluate Factual Consistency

**LLM-based evaluation:**

```bash
PYTHONPATH=. python evaluation/llm/statement2doc.py \
    --model /path/to/llama-3.1-8B-instruct \
    --baseline_dir ${BASELINE_DIR} \
    --task statement2explanation \
    --batch_size 24
```

**NLI-based evaluation:**

```bash
PYTHONPATH=. python evaluation/nli/nli_batch_pairs.py \
  --sts_ref_path ${DATASET_DIR}/sts.csv \
  --sts_pred_path ${BASELINE_DIR}/sts.csv \
  --model_name microsoft/deberta-large-mnli \
  --batch_size 64
```

---

## 🔍 Key Findings

Our experiments reveal a **dramatic disconnect** between surface-level quality and factual accuracy:

- **High fluency scores**: BERTScore F1 ranges from 0.81 to 0.90
- **Low factual precision**: Statement-level precision ranges from **4.38%** (NRT on Toys) to **32.88%** (XRec on Sports)
- **Poor recall**: Models miss 70%+ of ground-truth explanatory content
- **Standard metrics fail**: Similarity metrics don't correlate with factual consistency

**Implication**: Current models generate fluent but factually inconsistent explanations, highlighting the need for factuality-aware evaluation and model development.

---

## 📚 Citation

```bibtex
@article{kabongo2025factual,
  title={On the Factual Consistency of Text-based Explainable Recommendation Models},
  author={Kabongo, Ben and Guigue, Vincent},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📝 License

This project is released under [appropriate license]. The augmented datasets are available under the same terms as the original Amazon Reviews dataset.

---

## 🙏 Acknowledgments

We build upon several excellent open-source projects. See individual baseline README files for specific attributions.