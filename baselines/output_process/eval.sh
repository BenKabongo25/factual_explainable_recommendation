#!/bin/bash

#SBATCH --partition=jazzy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate genesis

DATASET_NAME=$1
MODEL_NAME=$2
L=$3

DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics
OUTPUT_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

PYTHONPATH=. python3 baselines/output_process/eval.py \
    --dataset_dir "$DATASET_DIR" \
    --baseline_dir "$OUTPUT_DIR" \
    --level "$L"
