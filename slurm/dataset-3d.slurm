#!/bin/bash
#SBATCH --job-name=dataset3d
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
export TRANSFORMERS_CACHE="${SCRATCH_DIR}/hf_cache"
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
echo "Initializing Miniconda..."
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate llm_steering

# ── Parameters (must match notebook Section 2 · Configuration) ────────────────
MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct"
CHUNK_SIZE=2000
DEVICE="cuda"
OUTPUT_DIR="${SCRATCH_DIR}/dataset_3d/output"
CHECKPOINT_DIR="${SCRATCH_DIR}/dataset_3d/checkpoints"
# N_TOKENS: not passed → defaults to None (all tokens), matching notebook

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== Starting dataset generation: $(date) ==="
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model:      ${MODEL_NAME}"
echo "Output:     ${OUTPUT_DIR}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Chunk size: ${CHUNK_SIZE}"
echo "HF_HOME:    ${HF_HOME}"
echo "TMPDIR:     ${TMPDIR}"

python -u -m src.dataset_3d.generate \
    --model "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --chunk-size ${CHUNK_SIZE} \
    --device ${DEVICE}

echo "=== Done: $(date) ==="
