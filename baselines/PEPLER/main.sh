DATASET_NAME=$1
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/
SAVE_DIR=/data/common/RecommendationDatasets/exps

PYTHONPATH=. python3 baselines/PEPLER/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --seed 42 \
    --lr 0.001 \
    --n_epochs 100 \
    --batch_size 32 \
    --review_length 128 \
    --no-load_model