#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

DATASET_NAME="${1:-Beauty}"         # e.g., Beauty, Toys, Sports, Clothes, Cell
SPLIT="${2:-train}"                 # "train", "eval" or "test"

MODEL=Grefer
DATA_ROOT=/data/common/RecommendationDatasets
DATASET_DIR="${DATA_ROOT}/${DATASET_NAME}_Amazon14/topics"
EXP_ROOT="${DATA_ROOT}/exps"
OUT_DIR="${EXP_ROOT}/${MODEL}/${DATASET_NAME}"
mkdir -p "${OUT_DIR}"

eval "$(conda shell.bash hook)"
conda activate genesis

export PYTHONPATH=.

PY_SCRIPT="code/dense_retriever.py"

echo "========== RUN CONFIG =========="
echo "JOB ID          : ${SLURM_JOB_ID:-N/A}"
echo "DATASET_NAME    : ${DATASET_NAME}"
echo "SPLIT           : ${SPLIT}"
echo "DATASET_DIR     : ${DATASET_DIR}"
echo "OUTPUT_DIR      : ${OUT_DIR}"
echo "PY_SCRIPT       : ${PY_SCRIPT}"
echo "================================"

python3 "${PY_SCRIPT}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUT_DIR}" \
  --split "${SPLIT}"
