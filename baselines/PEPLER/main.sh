#!/bin/bash

#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --constraint=A6000


eval "$(conda shell.bash hook)"
conda activate genesis

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

DATASET_NAME=$1

eval "$(conda shell.bash hook)"
conda activate genesis

PYTHONPATH=. python3 baselines/PEPLER/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir /data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/ \
    --save_dir /data/common/RecommendationDatasets/exps \
    --seed 42 \
    --lr 0.001 \
    --n_epochs 100 \
    --batch_size 32 \
    --review_length 128 \
    --no-load_model