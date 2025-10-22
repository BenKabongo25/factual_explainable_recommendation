DATASET_NAME=$1
MODEL_NAME=$2
PROMPT_NAME=${DATASET_NAME,,}
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}
BATCH_SIZE=16

PYTHONPATH=. python baselines/output_process/sts_extraction.py \
    --model /data/common/llama3/Meta-Llama-3-8B-Instruct \
    --prompt_text_file statement_topic_sentiment/prompts/${PROMPT_NAME}.txt \
    --dataset_path ${BASE_DIR}/output.csv \
    --output_dir ${BASE_DIR} \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens 512