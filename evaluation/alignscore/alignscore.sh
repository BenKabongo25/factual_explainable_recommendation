#!/bin/bash

#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate questeval

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

DATASET_NAME=$1
MODEL_NAME=$2
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

ALIGN_SCORE_MODEL=roberta-large
ALIGN_SCORE_CHECKPOINT=/data/common/AlignScore/alignscore_ckpt/AlignScore-large.ckpt


if [ "${MODEL_NAME}" == "XRec" ]; then
    DATA_PATH=${BASE_DIR}/clean.csv
else
    DATA_PATH=${BASE_DIR}/output.csv
fi

PYTHONPATH=. python evaluation/alignscore/main.py \
    --data_path ${DATA_PATH} \
    --batch_size 32 \
    --model ${ALIGN_SCORE_MODEL} \
    --ckpt_path ${ALIGN_SCORE_CHECKPOINT} \