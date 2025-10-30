DATASET_NAME=$1
MODEL_NAME=$2
NLI_MODEL_NAME=deberta-large-mnli
INPUT_PATH=/data/common/RecommendationDatasets/exps/${MODEL_NAME}/${DATASET_NAME}/nli_scores_${NLI_MODEL_NAME}_all.csv

PYTHONPATH=. python evaluation/nli/compute_metrics.py \
  --input_path ${INPUT_PATH}
