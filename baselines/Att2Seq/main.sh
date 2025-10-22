DATASET_NAME=$1
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/
SAVE_DIR=/data/common/RecommendationDatasets/exps

PYTHONPATH=. python3 baselines/Att2Seq/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --embedding_dim 64 \
    --hidden_size 512 \
    --n_layers 2 \
    --dropout 0.2 \
    --review_length 128 \
    --vocab_size 20000 \
    --batch_size 64 \
    --lr 0.002 \
    --n_epochs 100 \
    --seed 42 \
    --verbose