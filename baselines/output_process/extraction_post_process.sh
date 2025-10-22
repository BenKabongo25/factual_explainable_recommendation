DATASET_NAME=$1
MODEL_NAME=$2
OUTPUT_DIR=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}

PYTHONPATH=. python3 baselines/output_process/extraction_post_process.py \
    --dataset_name ${DATASET_NAME} \
    --input_path ${OUTPUT_DIR}/statement.csv \
    --output_data_path ${OUTPUT_DIR}/processed_statement.csv \
    --output_triplets_path ${OUTPUT_DIR}/sts.csv \
