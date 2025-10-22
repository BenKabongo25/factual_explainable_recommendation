#!/bin/bash

#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00


if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[warn] nvidia-smi fail, BATCH_SIZE=16"
    export BATCH_SIZE=16
else
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u || true)"

    if echo "$gpu_names" | grep -qi "A6000"; then
      export BATCH_SIZE=32
    elif echo "$gpu_names" | grep -qi "A5000"; then
      export BATCH_SIZE=16
    else
      echo "[warn] Unknown GPU (${gpu_names:-unknown}), BATCH_SIZE=16"
      export BATCH_SIZE=16
    fi
fi

eval "$(conda shell.bash hook)"
conda activate genesis

DATASET_NAME=$1
MODEL_NAME=$2
PROMPT_NAME=${DATASET_NAME,,}
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

PYTHONPATH=. python baselines/output_process/sts_extraction.py \
    --model /data/common/llama3/Meta-Llama-3-8B-Instruct \
    --prompt_text_file /home/kabongo/statement_topic_sentiment/prompts/${PROMPT_NAME}.txt \
    --dataset_path ${BASE_DIR}/output.csv \
    --output_dir ${BASE_DIR} \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens 512