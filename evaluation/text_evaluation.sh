#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate genesis

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

DATASET_NAME=$1
MODEL_NAME=$2
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

PYTHONPATH=. python evaluation/text_evaluation.py \
    --data_path ${BASE_DIR}/output.csv \
    --batch_size 8 \