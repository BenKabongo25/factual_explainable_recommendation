DATASET_NAME=$1
MODEL_NAME=$2
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

PYTHONPATH=. python evaluation/text_evaluation.py \
    --data_path ${BASE_DIR}/output.csv \
    --batch_size 8 \