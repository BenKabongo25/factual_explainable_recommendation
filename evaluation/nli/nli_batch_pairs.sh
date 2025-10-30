#!/bin/bash

#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate genesis

DATASET_NAME=$1
MODEL_NAME=$2
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/
EXP_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

STS_REF_PATH=${DATASET_DIR}/statement_topic_sentiment_freq.csv
REF_DATA_PATH=${DATASET_DIR}/test_data.csv
STS_PRED_PATH=${EXP_DIR}/sts.csv
PRED_DATA_PATH=${EXP_DIR}/processed_statement.csv

NLI_MODEL_NAME=microsoft/deberta-large-mnli

python evaluation/nli/nli_batch_pairs.py \
  --sts_ref_path ${STS_REF_PATH} \
  --sts_pred_path ${STS_PRED_PATH} \
  --ref_data_path ${REF_DATA_PATH} \
  --pred_data_path ${PRED_DATA_PATH} \
  --model_name ${NLI_MODEL_NAME} \
  --batch_size 64 \
  --max_length 256
