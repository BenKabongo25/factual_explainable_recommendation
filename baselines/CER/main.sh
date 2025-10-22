#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

DATASET=$1

eval "$(conda shell.bash hook)"
conda activate genesis

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

PYTHONPATH=. python3 baselines/CER/main.py \
    --dataset_name ${DATASET} \
    --dataset_dir /data/common/RecommendationDatasets/${DATASET}_Amazon14/topics/ \
    --save_dir /data/common/RecommendationDatasets/exps \
    --emsize 512 \
    --nhead 2 \
    --nhid 2048 \
    --nlayers 2 \
    --dropout 0.2 \
    --lr 1.0 \
    --clip 1.0 \
    --epochs 100 \
    --batch_size 32 \
    --seed 42 \
    --cuda \
    --vocab_size 20000 \
    --endure_times 100 \
    --rating_reg 0.1 \
    --context_reg 1.0 \
    --text_reg 1.0 \
    --peter_mask \
    --no-use_feature \
    --words 128
