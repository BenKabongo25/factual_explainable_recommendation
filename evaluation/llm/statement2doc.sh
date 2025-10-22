#!/bin/bash

#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --constraint=A6000

eval "$(conda shell.bash hook)"
conda activate genesis

DATASET_NAME=$1
BASELINE_NAME=$2
PROMPT_NAME=${DATASET_NAME,,}
BASE_DIR=/data/common/RecommendationDatasets/exps/${BASELINE_NAME}/${DATASET_NAME}
MODEL=/data/common/llama3/Meta-Llama-3.1-8B-Instruct
MODEL_NAME=llama-3.1-8b-instruct

PYTHONPATH=. python evaluation/llm/statement2doc.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --data_dir ${BASE_DIR} \
    --input_file processed_statement.csv \
    --batch_size 48 \
    --max_new_tokens 5