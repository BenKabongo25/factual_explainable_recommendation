#!/bin/bash

#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate genesis

DATASET_NAME=$1
BASELINE_NAME=$2
TASK="${3:-statement2explanation}"

DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/
BASELINE_DIR=/data/common/RecommendationDatasets/exps/${BASELINE_NAME}/${DATASET_NAME}
MODEL=/data/common/llama3/Meta-Llama-3.1-8B-Instruct
MODEL_NAME=llama-3.1-8b-instruct

STS_REF_PATH=${DATASET_DIR}/statement_topic_sentiment_freq.csv
REF_DATA_PATH=${DATASET_DIR}/test_data.csv

STS_PRED_PATH=${BASELINE_DIR}/sts.csv
PRED_DATA_PATH=${BASELINE_DIR}/processed_statement.csv

PYTHONPATH=. python evaluation/llm/statement2doc.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --data_dir ${BASELINE_DIR} \
    --sts_pred_path ${STS_PRED_PATH} \
    --sts_ref_path ${STS_REF_PATH} \
    --pred_data_path ${PRED_DATA_PATH} \
    --ref_data_path ${REF_DATA_PATH} \
    --task ${TASK} \
    --batch_size 24 \
    --max_new_tokens 5