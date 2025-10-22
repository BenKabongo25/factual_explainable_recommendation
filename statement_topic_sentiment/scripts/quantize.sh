DATASET_NAME=$1
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIR=${DATASET_DIR}/vectors/${EMBEDDING_MODEL}

PYTHONPATH=. python quantize.py \
    --embedding_path ${EMBEDDING_DIR}/embeddings.pt \
    --sts_csv ${DATASET_DIR}/statement_topic_sentiment_freq.csv \
    --output_dir ${DATASET_DIR}/rvq \
    --epochs 10 --batch_size 10240 \
    --codebook_size 32 --num_quantizers 3
