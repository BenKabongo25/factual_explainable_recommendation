DATASET_NAME=$1
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/
SAVE_DIR=/data/common/RecommendationDatasets/exps

PYTHONPATH=. python3 baselines/NRT/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --emsize 512 \
    --nhid 512 \
    --nlayers 4 \
    --rating_reg 1.0 \
    --text_reg 1.0 \
    --l2_reg 0.001 \
    --words 128 \
    --vocab_size 20000 \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 100 \
    --seed 42