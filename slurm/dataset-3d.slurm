#!/bin/bash
#SBATCH --job-name=dataset3d
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gpus-per-task=1
#SBATCH --constraint=v100
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
# Matches exactly what the notebook (llm_steering_3d.ipynb) sets
set -e
module purge
module load gcc/13.3.0

export SCRATCH_DIR="/scratch1/kelidari"
export HF_HOME="${SCRATCH_DIR}/hf_cache"
export TMPDIR="${SCRATCH_DIR}/tmp"
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Create scratch dirs
mkdir -p "${HF_HOME}"
mkdir -p "${TMPDIR}"
mkdir -p "${SCRATCH_DIR}/dataset_3d/output"
mkdir -p "${SCRATCH_DIR}/dataset_3d/checkpoints"

cd "${SLURM_SUBMIT_DIR:-$HOME/llm-steering}"

# ── Activate Conda ────────────────────────────────────────────────────────────
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate llm_steering
echo "  ✅ Conda environment llm_steering active"

# ── Parameters (must match notebook Section 2 · Configuration) ────────────────
MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct"
DEVICE="cuda"
STORAGE_DTYPE="float16"
CHECKPOINT_INTERVAL=100
OUTPUT_DIR="${SCRATCH_DIR}/dataset_3d/output"
CHECKPOINT_DIR="${SCRATCH_DIR}/dataset_3d/checkpoints"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== Starting dataset generation: $(date) ==="
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model:      ${MODEL_NAME}"
echo "Output:     ${OUTPUT_DIR}"
echo "Output file:${OUTPUT_DIR}/dataset.pt"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Dataset row schema: S x L rows with D(hidden), y1(x1 expert probs), y2(x2 expert probs)"
echo "Storage:    ${STORAGE_DTYPE}"
echo "HF_HOME:    ${HF_HOME}"
echo "TMPDIR:     ${TMPDIR}"

echo "=== Torch/CUDA preflight ==="
python -u -c "
import os
import torch

print(f'torch={torch.__version__}')
print(f'torch.version.cuda={torch.version.cuda}')
print(f'cuda.is_available={torch.cuda.is_available()}')
print(f'cuda.device_count={torch.cuda.device_count()}')
print(f'CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\")}')
if not torch.cuda.is_available():
    raise SystemExit(
        'CUDA is unavailable in this job environment. '
        'Use a GPU allocation and install a CUDA-enabled torch build.'
    )
"

python -u -m src.dataset_3d.generate \
    --model "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --checkpoint-interval ${CHECKPOINT_INTERVAL} \
    --storage-dtype "${STORAGE_DTYPE}" \
    --split train \
    --device ${DEVICE}

echo "=== Done: $(date) ==="
