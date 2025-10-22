DATASET_NAME=$1
MODEL_NAME=$2
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

ALIGN_SCORE_MODEL=roberta-large
ALIGN_SCORE_CHECKPOINT=/data/common/AlignScore/alignscore_ckpt/AlignScore-large.ckpt
DATA_PATH=${BASE_DIR}/output.csv

PYTHONPATH=. python evaluation/alignscore/main.py \
    --data_path ${DATA_PATH} \
    --batch_size 32 \
    --model ${ALIGN_SCORE_MODEL} \
    --ckpt_path ${ALIGN_SCORE_CHECKPOINT} \