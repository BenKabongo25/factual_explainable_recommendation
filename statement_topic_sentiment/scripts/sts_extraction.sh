DATASET_NAME=$1
PROMPT_NAME=${DATASET_NAME,,}
BASE_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics/

PYTHONPATH=. python sts_extraction.py \
    --model /data/common/llama3/Meta-Llama-3-8B-Instruct \
    --prompt_text_file prompts/${PROMPT_NAME}.txt \
    --dataset_path ${BASE_DIR}/reviews.json \
    --output_dir ${BASE_DIR} \
    --batch_size 32 \
    --max_new_tokens 512