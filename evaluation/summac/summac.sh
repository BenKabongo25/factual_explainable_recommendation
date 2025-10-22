DATASET_NAME=$1
MODEL_NAME=$2
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}
DATA_PATH=${BASE_DIR}/output.csv

PYTHONPATH=. python evaluation/summac/main.py \
    --data_path ${DATA_PATH} \
    --batch_size 16 \