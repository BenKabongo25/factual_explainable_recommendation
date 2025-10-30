# Baseline Output Post Process 

1. **Step 1**: Output clean
   ```bash
   PYTHONPATH=. python baselines/output_process/clean.py \
     --infile ${OUTPUT_DIR}/output.csv \
     --outfile ${OUTPUT_DIR}/output.csv \
     --col prediction
   ```

2. **Step 2**: Statement Topic Sentiment Triplet extraction
   ```bash
    PYTHONPATH=. python baselines/output_process/sts_extraction.py \
        --model /data/common/llama3/Meta-Llama-3-8B-Instruct \
        --prompt_text_file statement_topic_sentiment/prompts/${PROMPT_NAME}.txt \
        --dataset_path ${OUTPUT_DIR}/output.csv \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens 512
   ```

3. **Step 3**: Post process triplets extraction
   ```bash
    PYTHONPATH=. python3 baselines/output_process/extraction_post_process.py \
        --dataset_name ${DATASET_NAME} \
        --input_path ${OUTPUT_DIR}/statement.csv \
        --output_data_path ${OUTPUT_DIR}/processed_statement.csv \
        --output_triplets_path ${OUTPUT_DIR}/sts.csv \
   ```