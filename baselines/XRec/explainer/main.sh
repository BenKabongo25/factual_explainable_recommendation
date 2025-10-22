#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

DATASET_NAME="${1:-Beauty}"      # e.g., Beauty, Toys, Sports, Clothes, Cell
MODE="${2:-finetune}"            # "finetune" or "generate"
EPOCHS="${3:-1}"
BATCH_SIZE="${4:-1}"
LR="${5:-1e-4}"
WEIGHT_DECAY="${6:-1e-6}"

MODEL=XRec
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

PY_SCRIPT="explainer/main.py"

echo "========== RUN CONFIG =========="
echo "JOB ID          : ${SLURM_JOB_ID:-N/A}"
echo "DATASET_NAME    : ${DATASET_NAME}"
echo "MODE            : ${MODE}"
echo "EPOCHS          : ${EPOCHS}"
echo "BATCH_SIZE      : ${BATCH_SIZE}"
echo "LR              : ${LR}"
echo "WEIGHT_DECAY    : ${WEIGHT_DECAY}"
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
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --epochs "${EPOCHS}" \
  --mode "${MODE}"
