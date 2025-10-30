# Baselines

This folder collects the **reference implementations** we use to benchmark text-based explainable recommendation models.

> * RNN-based models: **Att2Seq**, **NRT**
> * Transformer-based models: **PETER**, **CER**, **PEPLER**
> * LLM-enhanced model: **XRec** (LightGCN encoder + LLM explainer)

---

### Att2Seq

**GitHub**: [https://github.com/lileipisces/Att2Seq](https://github.com/lileipisces/Att2Seq)
**Paper**: *Learning to Generate Product Reviews from Attributes.*

```bash
PYTHONPATH=. python3 baselines/Att2Seq/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --embedding_dim 64 \
    --hidden_size 512 \
    --n_layers 2 \
    --dropout 0.2 \
    --review_length 128 \
    --vocab_size 20000 \
    --batch_size 64 \
    --lr 0.002 \
    --n_epochs 100 \
    --seed 42 \
    --verbose
```

---

### NRT

**GitHub**: [https://github.com/lileipisces/NRT](https://github.com/lileipisces/NRT)
**Paper**: *Neural Rating Regression with Abstractive Tips Generation for Recommendation.*

```bash
PYTHONPATH=. python3 baselines/NRT/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --emsize 512 \
    --nhid 512 \
    --nlayers 4 \
    --rating_reg 1.0 \
    --text_reg 1.0 \
    --l2_reg 0.001 \
    --words 128 \
    --vocab_size 20000 \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 100 \
    --seed 42
```

---

### CER

**GitHub**: [https://github.com/JMRaczynski/CER](https://github.com/JMRaczynski/CER)
**Paper**: *The Problem of Coherence in Natural Language Explanations of Recommendations.*

```bash
PYTHONPATH=. python3 baselines/CER/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --emsize 512 \
    --nhead 2 \
    --nhid 2048 \
    --nlayers 2 \
    --dropout 0.2 \
    --lr 1.0 \
    --clip 1.0 \
    --epochs 100 \
    --batch_size 32 \
    --seed 42 \
    --cuda \
    --vocab_size 20000 \
    --endure_times 100 \
    --rating_reg 0.1 \
    --context_reg 1.0 \
    --text_reg 1.0 \
    --peter_mask \
    --no-use_feature \
    --words 128
```

---

### PETER

**GitHub**: [https://github.com/lileipisces/PETER?tab=readme-ov-file](https://github.com/lileipisces/PETER?tab=readme-ov-file)
**Paper**: *Personalized Transformer for Explainable Recommendation.*

```bash
PYTHONPATH=. python3 baselines/PETER/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --emsize 512 \
    --nhead 2 \
    --nhid 2048 \
    --nlayers 2 \
    --dropout 0.2 \
    --lr 1.0 \
    --clip 1.0 \
    --epochs 100 \
    --batch_size 32 \
    --seed 42 \
    --cuda \
    --vocab_size 20000 \
    --endure_times 100 \
    --rating_reg 0.1 \
    --context_reg 1.0 \
    --text_reg 1.0 \
    --peter_mask \
    --no-use_feature \
    --words 128
```

---

### PEPLER

**GitHub**: [https://github.com/lileipisces/PEPLER](https://github.com/lileipisces/PEPLER)
**Paper**: *Personalized Prompt Learning for Explainable Recommendation.*

```bash
PYTHONPATH=. python3 baselines/PEPLER/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --seed 42 \
    --lr 0.001 \
    --n_epochs 100 \
    --batch_size 32 \
    --review_length 128 \
    --no-load_model
```
---

### XRec

**GitHub**: [https://github.com/HKUDS/XRec](https://github.com/HKUDS/XRec)
**Paper**: *Large Language Models for Explainable Recommendation.*

#### 1) NL Profiles (Items)

```bash
MODEL=/data/common/llama3/Meta-Llama-3.1-8B-Instruct
MODEL_NAME=llama3.1-8B-it

PYTHONPATH=. python baselines/NL_profiles/generate_item_summaries.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --prompt_text_file baselines/NL_profiles/prompts/${DATASET_NAME}/item.txt \
    --data_dir ${DATA_DIR} \
    --batch_size 16 \
    --max_new_tokens 384 \
    --output_dir ${OUTPUT_DIR} \
    --item_metadata_path ${METADATA_PATH} \
    --num_words 100
```

#### 2) NL Profiles (Users)

```bash
MODEL=/data/common/llama3/Meta-Llama-3.1-8B-Instruct
MODEL_NAME=llama3.1-8B-it

PYTHONPATH=. python baselines/NL_profiles/generate_user_summaries.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --prompt_text_file baselines/NL_profiles/prompts/${DATASET_NAME}/user.txt \
    --data_dir ${DATA_DIR} \
    --batch_size 16 \
    --max_new_tokens 384 \
    --output_dir ${OUTPUT_DIR} \
    --num_words 100
```

#### 3) Train encoder

```bash
cd baselines/XRec

MODEL="${2:-LightGCN}"
N_EPOCHS="${3:-300}"
BATCH_SIZE="${4:-1024}"
LR="${5:-1e-3}"
WEIGHT_DECAY="${6:-1e-6}"
N_LAYERS="${7:-4}"
EMBEDDING_DIM="${8:-64}"

PYTHONPATH=. python3 encoder/train_encoder.py \
  --model "${MODEL}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --n_epochs "${N_EPOCHS}" \
  --n_layers "${N_LAYERS}" \
  --embedding_dim "${EMBEDDING_DIM}"
```

#### 4) Train explainer

```bash
cd baselines/XRec

MODEL="${2:-LightGCN}"
N_EPOCHS="${3:-300}"
BATCH_SIZE="${4:-1024}"
LR="${5:-1e-3}"
WEIGHT_DECAY="${6:-1e-6}"
N_LAYERS="${7:-4}"
EMBEDDING_DIM="${8:-64}"

PYTHONPATH=. python3 encoder/train_encoder.py \
  --model "${MODEL}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --n_epochs "${N_EPOCHS}" \
  --n_layers "${N_LAYERS}" \
  --embedding_dim "${EMBEDDING_DIM}"
```

---
