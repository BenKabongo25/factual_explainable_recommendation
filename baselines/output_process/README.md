# Baseline Output Post Process 

1. **Step 1**: Output clean
   ```bash
   PYTHONPATH=. python baselines/output_process/clean.py \
     --infile \path\to\baseline_output \
     --outfile \path\to\output_file \
     --col prediction
   ```

2. **Step 2**: Statement Topic Sentiment Triplet extraction
   ```bash
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
   ```

3. **Step 3**: Post process triplets extraction
   ```bash
    DATASET_NAME=$1
    MODEL_NAME=$2
    OUTPUT_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

    PYTHONPATH=. python3 baselines/output_process/extraction_post_process.py \
        --dataset_name ${DATASET_NAME} \
        --input_path ${OUTPUT_DIR}/statement.csv \
        --output_data_path ${OUTPUT_DIR}/processed_statement.csv \
        --output_triplets_path ${OUTPUT_DIR}/sts.csv \
   ```