DATASET_NAME=$1
SPLIT=$2
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics

python build_explanations.py \
    --data_csv ${DATASET_DIR}/${SPLIT}_data.csv \
    --sts_csv ${DATASET_DIR}/statement_topic_sentiment_freq.csv \
    --output_csv ${DATASET_DIR}/${SPLIT}_explanations.csv \