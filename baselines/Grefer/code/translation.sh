DATASET_NAME="${1:-Beauty}"         # e.g., Beauty, Toys, Sports, Clothes, Cell
SPLIT="${2:-train}"                 # "train", "eval" or "test"
TEXT_ENCODER="${3:-SentenceBert}"   # "SentenceBert" or "SimCSE"

MODEL=Grefer
PROFILE_MODEL=llama3.1-8B-it

DATA_ROOT=/data/common/RecommendationDatasets
DATASET_DIR="${DATA_ROOT}/${DATASET_NAME}_Amazon14/topics"
EXP_ROOT="${DATA_ROOT}/exps"
PROFILES_DIR="${EXP_ROOT}/NL_profiles/${DATASET_NAME}/${PROFILE_MODEL}"
OUT_DIR="${EXP_ROOT}/${MODEL}/${DATASET_NAME}"
mkdir -p "${OUT_DIR}"

eval "$(conda shell.bash hook)"
conda activate genesis

export PYTHONPATH=.

PY_SCRIPT="code/translation.py"

echo "========== RUN CONFIG =========="
echo "JOB ID          : ${SLURM_JOB_ID:-N/A}"
echo "DATASET_NAME    : ${DATASET_NAME}"
echo "SPLIT           : ${SPLIT}"
echo "DATASET_DIR     : ${DATASET_DIR}"
echo "PROFILES_DIR    : ${PROFILES_DIR}"
echo "OUTPUT_DIR      : ${OUT_DIR}"
echo "PY_SCRIPT       : ${PY_SCRIPT}"
echo "================================"

python3 "${PY_SCRIPT}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUT_DIR}" \
  --profiles_dir "${PROFILES_DIR}" \
  --split "${SPLIT}" \
  --text_encoder "${TEXT_ENCODER}"
