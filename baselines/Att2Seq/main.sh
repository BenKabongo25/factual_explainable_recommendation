#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

DATASET_NAME=$1

eval "$(conda shell.bash hook)"
conda activate genesis

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

PYTHONPATH=. python3 baselines/Att2Seq/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir /data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/ \
    --save_dir /data/common/RecommendationDatasets/exps \
    --embedding_dim 64 \
    --hidden_size 512 \
    --n_layers 2 \
    --dropout 0.2 \
    --review_length 128 \
    --vocab_size 20000 \
    --batch_size 64 \
    --lr 0.002 \
    --n_epochs 100 \
    --seed 42 \
    --verbose