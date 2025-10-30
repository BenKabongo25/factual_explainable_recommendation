DATASET_NAME=$1
DATASET_DIR=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/topics

PYTHONPATH=. python data/splits.py \
    --dataset_dir "$DATASET_DIR" \
    --processed_data_file data.csv \
    --time_train_size 0.8 \
    --time_eval_size 0.1 \
    --time_test_size 0.1 \
    --min_interactions 5 \
    --delete_cold_start_items \
    --user_train_size 0.8 \
    --user_eval_size 0.1 \
    --user_test_size 0.1