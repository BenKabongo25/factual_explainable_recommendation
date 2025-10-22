#!/bin/bash

#SBATCH --partition=electronic
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

if [ "${MODEL_NAME}" == "XRec" ]; then
    DATA_PATH=${BASE_DIR}/clean.csv
else
    DATA_PATH=${BASE_DIR}/output.csv
fi

PYTHONPATH=. python evaluation/summac/main.py \
    --data_path ${DATA_PATH} \
    --batch_size 16 \