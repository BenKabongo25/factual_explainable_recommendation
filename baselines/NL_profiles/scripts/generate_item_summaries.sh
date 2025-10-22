DATASET_NAME=$1
DATA_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/
MODEL=/data/common/llama3/Meta-Llama-3.1-8B-Instruct
MODEL_NAME=llama3.1-8B-it
OUTPUT_DIR=/data/common/RecommendationDatasets/exps/NL_profiles/${DATASET_NAME}
METADATA_PATH=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/metadata.json

PYTHONPATH=. python baselines/NL_profiles/generate_item_summaries.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --prompt_text_file baselines/NL_profiles/prompts/${DATASET_NAME}/item.txt \
    --data_dir ${DATA_DIR} \
    --batch_size 16 \
    --max_new_tokens 384 \
    --output_dir ${OUTPUT_DIR} \
    --item_metadata_path ${METADATA_PATH} \
    --num_words 100