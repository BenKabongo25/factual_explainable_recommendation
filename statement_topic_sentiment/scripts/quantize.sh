#!/bin/bash

#SBATCH --partition=hard
#SBATCH --nodelist=top
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=quantize
#SBATCH --output=logs/quantize.out
#SBATCH --error=logs/quantize.err

eval "$(conda shell.bash hook)"
conda activate genesis

DATASET_NAME=$1
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIR=${DATASET_DIR}/vectors/${EMBEDDING_MODEL}

PYTHONPATH=. python quantize.py \
    --embedding_path ${EMBEDDING_DIR}/embeddings.pt \
    --sts_csv ${DATASET_DIR}/statement_topic_sentiment_freq.csv \
    --output_dir ${DATASET_DIR}/rvq \
    --epochs 10 --batch_size 10240 \
    --codebook_size 32 --num_quantizers 3
