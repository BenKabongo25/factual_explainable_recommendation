DATASET_NAME=$1
MODEL=$2
BASE_DIR=/data/common/RecommendationDatasets/exps/${MODEL}/${DATASET_NAME}
EMBEDDING_MODEL=all-MiniLM-L6-v2

PYTHONPATH=. python3 baselines/output_process/embed.py \
    --sts_path ${BASE_DIR}/sts.csv \
    --output_dir ${BASE_DIR} \
    --model_name ${EMBEDDING_MODEL} \
    --batch_size 32
    