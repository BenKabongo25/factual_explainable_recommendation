DATASET_NAME=$1
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics
EMBEDDING_MODEL=all-MiniLM-L6-v2

PYTHONPATH=. python embed.py \
    --dataset_dir ${DATASET_DIR} \
    --model_name ${EMBEDDING_MODEL} \
    --batch_size 32 \