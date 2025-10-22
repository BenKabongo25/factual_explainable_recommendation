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
PROMPT_NAME=${DATASET_NAME,,}
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

PYTHONPATH=. python baselines/output_process/sts_extraction.py \
    --model /data/common/llama3/Meta-Llama-3-8B-Instruct \
    --prompt_text_file /home/kabongo/statement_topic_sentiment/prompts/${PROMPT_NAME}.txt \
    --dataset_path ${BASE_DIR}/output.csv \
    --output_dir ${BASE_DIR} \
    --batch_size 32 \
    --max_new_tokens 512