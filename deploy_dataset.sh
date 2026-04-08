#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# deploy_dataset.sh — Piped to CARC via SSH
#
# Usage (from local Mac):
#   ssh $USER@discovery1.usc.edu \
#       "USER=$USER WANDB_API_KEY=$WANDB_API_KEY HF_TOKEN=$HF_TOKEN bash -s" \
#       < deploy_dataset.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

# Initialize Lmod module system (required for non-interactive shells like SSH pipes)
if [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

PROJECT_DIR="/home1/${USER}/llm-steering"
SCRATCH="/scratch1/${USER}"
DATASET_DIR="${SCRATCH}/dataset_3d"

echo "═══════════════════════════════════════════"
echo "  1 · Pull latest code"
echo "═══════════════════════════════════════════"
cd "${PROJECT_DIR}"
git pull --ff-only
echo "  ✅ Code up to date"

echo ""
echo "═══════════════════════════════════════════"
echo "  2 · Setup directories & environment"
echo "═══════════════════════════════════════════"
mkdir -p "${DATASET_DIR}/output"
mkdir -p "${DATASET_DIR}/checkpoints"
mkdir -p "${SCRATCH}/hf_cache"
echo "  Output:      ${DATASET_DIR}/output"
echo "  Checkpoints: ${DATASET_DIR}/checkpoints"
echo "  HF cache:    ${SCRATCH}/hf_cache"

# --- Environment Setup ---
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment (.venv)..."
    module load python/3.12
    python -m venv .venv
fi

echo "  Activating environment and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip

# Check for the vllm wheel mentioned in pyproject.toml
VLLM_WHEEL="vllm-0.18.0+cu126-cp312-cp312-linux_x86_64.whl"
if [ ! -f "$VLLM_WHEEL" ]; then
    echo "  ⚠️  WARNING: $VLLM_WHEEL not found. Installation may fail if required by pyproject.toml."
    echo "     (Please upload the wheel or update pyproject.toml if vllm is needed)"
fi

# Install dependencies in editable mode
# Note: This might fail if the wheel above is missing and required.
pip install -e . || echo "  ⚠️  Warning: pip install -e . failed. You may need to install dependencies manually."

echo ""
echo "═══════════════════════════════════════════"
echo "  3 · Submit SLURM job"
echo "═══════════════════════════════════════════"
JOB_ID=$(sbatch --parsable \
    --export=ALL,HF_TOKEN="${HF_TOKEN}",WANDB_API_KEY="${WANDB_API_KEY}" \
    slurm/dataset-3d.slurm)

echo "  ✅ Job submitted: ${JOB_ID}"
echo ""
echo "  Monitor:"
echo "    squeue -u ${USER}"
echo "    tail -f ${PROJECT_DIR}/dataset3d_${JOB_ID}.out"
