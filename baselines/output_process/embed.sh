#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00


DATASET_NAME=$1
MODEL=$2
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL}/${DATASET_NAME}
EMBEDDING_MODEL=all-MiniLM-L6-v2

eval "$(conda shell.bash hook)"
conda activate genesis

PYTHONPATH=. python3 baselines/output_process/embed.py \
    --sts_path ${BASE_DIR}/sts.csv \
    --output_dir ${BASE_DIR} \
    --model_name ${EMBEDDING_MODEL} \
    --batch_size 32
    