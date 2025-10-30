# Evaluation

## LLM-based Metrics

### 1) LLM annotations: Statement (prediction) to Explanation (GT) (precision)

```bash
TASK=statement2explanation
MODEL=/data/common/llama3/Meta-Llama-3.1-8B-Instruct
MODEL_NAME=llama-3.1-8b-instruct
STS_REF_PATH=${DATASET_DIR}/sts.csv
REF_DATA_PATH=${DATASET_DIR}/test_data.csv
STS_PRED_PATH=${BASELINE_DIR}/sts.csv
PRED_DATA_PATH=${BASELINE_DIR}/processed_statement.csv

PYTHONPATH=. python evaluation/llm/statement2doc.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --data_dir ${DATASET_DIR} \
    --baseline_dir ${BASELINE_DIR} \
    --sts_pred_path ${STS_PRED_PATH} \
    --sts_ref_path ${STS_REF_PATH} \
    --pred_data_path ${PRED_DATA_PATH} \
    --ref_data_path ${REF_DATA_PATH} \
    --task ${TASK} \
    --batch_size 24 \
    --max_new_tokens 5
```

### 2) LLM annotations: Statement (GT) to Explanation (prediction) (recall)

```bash
TASK=statement_ref2explanation_gen
MODEL=/data/common/llama3/Meta-Llama-3.1-8B-Instruct
MODEL_NAME=llama-3.1-8b-instruct
STS_REF_PATH=${DATASET_DIR}/sts.csv
REF_DATA_PATH=${DATASET_DIR}/test_data.csv
STS_PRED_PATH=${BASELINE_DIR}/sts.csv
PRED_DATA_PATH=${BASELINE_DIR}/processed_statement.csv

PYTHONPATH=. python evaluation/llm/statement2doc.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --data_dir ${DATASET_DIR} \
    --baseline_dir ${BASELINE_DIR} \
    --sts_pred_path ${STS_PRED_PATH} \
    --sts_ref_path ${STS_REF_PATH} \
    --pred_data_path ${PRED_DATA_PATH} \
    --ref_data_path ${REF_DATA_PATH} \
    --task ${TASK} \
    --batch_size 24 \
    --max_new_tokens 5
```

### 3) Compute metrics: St2Exp-P/R/F1

```bash
PYTHONPATH=. python evaluation/llm/compute_mectrics.py \
    --precision_path ${BASELINE_DIR}/statement2explanation_labels.csv \
    --recall_path ${BASELINE_DIR}/statement_ref2explanation_gen_labels.csv \
    --output_path ${BASELINE_DIR}/llm_metrics.json \
```

**Statement-to-Explanation Precision** `St2Exp-P` measures the factual consistency of the generated explanation $\mathbf{e}'$ with respect to the ground-truth $\mathbf{e}$ at statement level.
Given $n$ statements $\{\mathbf{s}'_1,\ldots,\mathbf{s}'_n\}$ extracted from $\mathbf{e}'$, we score each statement against the ground-truth: $s'_k = f_{\mathrm{LLM}}(\mathbf{s}'_k, \mathbf{e})$.
The metric is the fraction of positively supported statements, given by:
$$\texttt{St2Exp-P}(\mathbf{e}', \mathbf{e}) = \frac{1}{n} \sum_{k=1}^{n} s'_k \;=\; \frac{1}{n} \sum_{k=1}^{n} f_{\mathrm{LLM}}(\mathbf{s}'_k, \mathbf{e}).$$

**Statement-to-Explanation Recall** `St2Exp-R` swaps the roles of $\mathbf{e}'$ and $\mathbf{e}$, thus measuring how well the generated explanation covers the reference’s explanatory content.
Given $m$ statements $\{\mathbf{s}_1,\ldots,\mathbf{s}_m\}$ extracted from $\mathbf{e}$, each statement is scored against the prediction: $s_l = f_{\mathrm{LLM}}(\mathbf{s}_l, \mathbf{e}')$.
The metric is given by:
$$\texttt{St2Exp-R}(\mathbf{e}', \mathbf{e}) = \frac{1}{m} \sum_{l=1}^{m} s_l \;=\; \frac{1}{m} \sum_{l=1}^{m} f_{\mathrm{LLM}}(\mathbf{s}_l, \mathbf{e}').$$

**Statement-to-Explanation F1** `St2Exp-F1` is the harmonic mean of precision and recall, computed as:
$$\texttt{St2Exp-F1} \;=\;2\; \frac{\,\texttt{St2Exp-P}\cdot \texttt{St2Exp-R}}{\texttt{St2Exp-P} + \texttt{St2Exp-R}}.$$

___

## NLI-based Metrics

### 1) NLI annotations: Statement to Statement

* Prediction to GT (precision) 
* GT to Prediction (recall)

```bash
STS_REF_PATH=${DATASET_DIR}/sts.csv
REF_DATA_PATH=${DATASET_DIR}/test_data.csv
STS_PRED_PATH=${BASELINE_DIR}/sts.csv
PRED_DATA_PATH=${BASELINE_DIR}/processed_statement.csv

NLI_MODEL_NAME=microsoft/deberta-large-mnli

python evaluation/nli/nli_batch_pairs.py \
  --sts_ref_path ${STS_REF_PATH} \
  --sts_pred_path ${STS_PRED_PATH} \
  --ref_data_path ${REF_DATA_PATH} \
  --pred_data_path ${PRED_DATA_PATH} \
  --model_name ${NLI_MODEL_NAME} \
  --batch_size 64 \
  --max_length 256
```

### 2) Compute metrics

```bash
NLI_MODEL_NAME=deberta-large-mnli

PYTHONPATH=. python evaluation/nli/compute_metrics.py \
  --input_path ${BASELINE_DIR}/{NLI_MODEL_NAME}_all.csv
  --output_path ${BASELINE_DIR}/nli_metrics.json \
```

**Precision-oriented metrics**
These metrics measure the factual consistency of the generated explanation $\mathbf{e}'$ with respect to the ground-truth $\mathbf{e}$:
$$\texttt{St*-P}(\mathbf{e}', \mathbf{e}) \;=\; \frac{1}{n} \sum_{k=1}^{n} \max_{l}\, f_{\mathrm{NLI\text{-}*}}(\mathbf{s}'_k, \mathbf{s}_l),$$
where $*$ indexes the chosen scoring function.
This yields `StEnt-P` (entailment score), `StEnt-bin-P` (binary entailment), and `StCoh-P` (entailment minus contradiction).

**Recall-oriented metrics**
As with the LLM family, recall metrics swap the roles of $\mathbf{e}'$ and $\mathbf{e}$ to assess coverage:
$$\texttt{St*-R}(\mathbf{e}', \mathbf{e}) \;=\; \frac{1}{m} \sum_{l=1}^{m} \max_{k}\, f_{\mathrm{NLI\text{-}*}}(\mathbf{s}_l, \mathbf{s}'_k).$$
This gives `StEnt-R`, `StEnt-bin-R`, and `StCoh-R`.

**Aggregated metrics**
From precision and recall we compute F1:
$$\texttt{St*-F1} \;=\; \frac{2\,\texttt{St*-P}\cdot\texttt{St*-R}}{\texttt{St*-P} + \texttt{St*-R}},$$
yielding `StEnt-F1` and `tEnt-bin-F1`.

___

## Standard NLI and QG-QA Metrics

### 2) AlignScore (NLI)

```bash
ALIGN_SCORE_MODEL=roberta-large
ALIGN_SCORE_CHECKPOINT=/data/common/AlignScore/alignscore_ckpt/AlignScore-large.ckpt

PYTHONPATH=. python evaluation/alignscore/main.py \
    --data_path ${BASELINE_DIR}/output.csv \
    --batch_size 32 \
    --model ${ALIGN_SCORE_MODEL} \
    --ckpt_path ${ALIGN_SCORE_CHECKPOINT} \
    --output_path ${BASELINE_DIR}/alignscore_results.json \
```

### 2) SummaC (NLI)

```bash
PYTHONPATH=. python evaluation/summac/main.py \
    --data_path ${BASELINE_DIR}/output.csv \
    --output_path ${BASELINE_DIR}/summac_results.json \
    --batch_size 16 \
```

### 3) QuestEval (QG-QA)

```bash
PYTHONPATH=. python evaluation/qa_questeval/main.py \
    --data_path ${BASELINE_DIR}/output.csv \
    --output_path ${BASELINE_DIR}/questeval_results.json \
    --batch_size 16 \
```
___

## Text Similarity Metrics

```bash
PYTHONPATH=. python evaluation/text_evaluation.py \
    --data_path ${BASELINE_DIR}/output.csv \
    --output_path ${BASELINE_DIR}/text_similarity_results.json \
    --batch_size 16 \
```
