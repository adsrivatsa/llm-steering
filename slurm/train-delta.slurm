#!/bin/bash
#SBATCH --job-name=train_delta
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gpus-per-task=1
#SBATCH --constraint=v100
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/train_delta_%A.out
#SBATCH --error=slurm/logs/train_delta_%A.err

# ── Environment ───────────────────────────────────────────────────────────────
set -e
module purge
module load gcc/13.3.0

export SCRATCH_DIR="/scratch1/kelidari"
export TMPDIR="${SCRATCH_DIR}/tmp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONNOUSERSITE=1

mkdir -p "${TMPDIR}"
mkdir -p "slurm/logs"

cd "${SLURM_SUBMIT_DIR:-$HOME/llm-steering}"

# ── Activate Conda ────────────────────────────────────────────────────────────
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate llm_steering
echo "  ✅ Conda environment llm_steering active"

# ── Parameters ────────────────────────────────────────────────────────────────
EPOCHS=10
BATCH_SIZE=32
LR=1e-3
MARGIN=0.05
DATASET_PATH="${SCRATCH_DIR}/dataset_3d/output/dataset.pt"

# You can adjust this to save to an EXP specific folder if needed
OUTPUT_PATH="${SCRATCH_DIR}/dataset_3d/trained_delta.pt"

if [ ! -z "${EXP_ID}" ]; then
    DATASET_PATH="${SCRATCH_DIR}/dataset_3d/output/EXP${EXP_ID}/dataset.pt"
    OUTPUT_PATH="${SCRATCH_DIR}/dataset_3d/trained_delta_EXP${EXP_ID}.pt"
fi

echo "=== Training 3D DELTA Tensor: $(date) ==="
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Dataset:      ${DATASET_PATH}"
echo "Output:       ${OUTPUT_PATH}"

# ── Run ───────────────────────────────────────────────────────────────────────
python -u src/dataset_3d/train_delta.py \
    --dataset-path "${DATASET_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --margin ${MARGIN}

echo "=== Training done: $(date) ==="
