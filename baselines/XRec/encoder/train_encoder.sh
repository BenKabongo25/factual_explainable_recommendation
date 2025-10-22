DATASET_NAME="${1:-Beauty}"           # ex: Beauty, Toys, Sports, ...
MODEL="${2:-LightGCN}"                # ex: LightGCN
N_EPOCHS="${3:-300}"
BATCH_SIZE="${4:-1024}"
LR="${5:-1e-3}"
WEIGHT_DECAY="${6:-1e-6}"
N_LAYERS="${7:-4}"
EMBEDDING_DIM="${8:-64}"

DATA_ROOT=/data/common/RecommendationDatasets
DATASET_DIR="${DATA_ROOT}/${DATASET_NAME}_Amazon14/topics"
EXP_ROOT=${DATA_ROOT}/exps
OUT_DIR="${EXP_ROOT}/XRec/${DATASET_NAME}"
mkdir -p "${OUT_DIR}"

eval "$(conda shell.bash hook)"
conda activate genesis

export PYTHONPATH=.

echo "========== RUN CONFIG =========="
echo "JOB ID         : ${SLURM_JOB_ID:-N/A}"
echo "DATASET_NAME   : ${DATASET_NAME}"
echo "MODEL          : ${MODEL}"
echo "DATASET_DIR    : ${DATASET_DIR}"
echo "OUTPUT_DIR     : ${OUT_DIR}"
echo "BATCH_SIZE     : ${BATCH_SIZE}"
echo "LR             : ${LR}"
echo "WEIGHT_DECAY   : ${WEIGHT_DECAY}"
echo "N_EPOCHS       : ${N_EPOCHS}"
echo "N_LAYERS       : ${N_LAYERS}"
echo "EMBEDDING_DIM  : ${EMBEDDING_DIM}"
echo "================================"

PY_SCRIPT="encoder/train_encoder.py"

python3 "${PY_SCRIPT}" \
  --model "${MODEL}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --n_epochs "${N_EPOCHS}" \
  --n_layers "${N_LAYERS}" \
  --embedding_dim "${EMBEDDING_DIM}"
