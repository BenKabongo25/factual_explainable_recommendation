DATASET_NAME=$1
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/
SAVE_DIR=/data/common/RecommendationDatasets/exps

PYTHONPATH=. python3 baselines/CER/main.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --save_dir ${SAVE_DIR} \
    --emsize 512 \
    --nhead 2 \
    --nhid 2048 \
    --nlayers 2 \
    --dropout 0.2 \
    --lr 1.0 \
    --clip 1.0 \
    --epochs 100 \
    --batch_size 32 \
    --seed 42 \
    --cuda \
    --vocab_size 20000 \
    --endure_times 100 \
    --rating_reg 0.1 \
    --context_reg 1.0 \
    --text_reg 1.0 \
    --peter_mask \
    --no-use_feature \
    --words 128
