#!/bin/bash

#SBATCH --partition=funky
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

DATASET_NAME=$1

eval "$(conda shell.bash hook)"
conda activate genesis

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

PYTHONPATH=. python3 baselines/NRT/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir /data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/ \
    --save_dir /data/common/RecommendationDatasets/exps \
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