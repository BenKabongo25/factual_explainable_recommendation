#!/bin/bash

#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --constraint=A6000

eval "$(conda shell.bash hook)"
conda activate genesis

DATASET_NAME=$1
MODEL_NAME=$2
EMBEDDING_MODEL=all-MiniLM-L6-v2

DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics
OUTPUT_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

STS_PATH=${OUTPUT_DIR}/sts.csv
EMB_PATH=${OUTPUT_DIR}/${EMBEDDING_MODEL}/embeddings.pt
ROOT_DIR=${DATASET_DIR}/rvq
OUT_CSV=${OUTPUT_DIR}/quantized_sts.csv
CHUNK_SIZE=1024
NUM_STAGES=3

PY_SCRIPT=baselines/output_process/quantize.py

python -u "$PY_SCRIPT" \
    --sts_path "$STS_PATH" \
    --embeddings_pt "$EMB_PATH" \
    --codebooks_root "$ROOT_DIR" \
    --output_csv "$OUT_CSV" \
    --chunk_size "$CHUNK_SIZE" \
    --default_num_stages "$NUM_STAGES"