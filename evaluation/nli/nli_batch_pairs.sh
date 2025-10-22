

DATASET_NAME=$1
MODEL_NAME=$2
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}/topics/
EXP_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

STS_REF_PATH=${DATASET_DIR}/statement_topic_sentiment_freq.csv
REF_DATA_PATH=${DATASET_DIR}/test_data.csv
STS_PRED_PATH=${EXP_DIR}/sts.csv
PRED_DATA_PATH=${EXP_DIR}/processed_statement.csv

NLI_MODEL_NAME="microsoft/deberta-v3-large-mnli"

python evaluation/nli/nli_batch_pairs.py \
  --sts_ref_path /path/to/sts_ref.csv \
  --sts_pred_path /path/to/sts_pred.csv \
  --ref_data_path /path/to/ref_data.csv \
  --pred_data_path /path/to/pred_data.csv \
  --model_name ${NLI_MODEL_NAME} \
  --batch_size 32 \
  --max_length 256
