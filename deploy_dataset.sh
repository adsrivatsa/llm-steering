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

# Initialize Lmod/Module system (required for non-interactive shells like SSH pipes)
# Try multiple standard CARC and Linux paths
if [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

# Baseline profile sourcing
[ -f /etc/profile ] && source /etc/profile

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
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

export PATH="$HOME/miniconda/bin:$PATH"

ENV_NAME="llm_steering"
if conda env list | grep -q "$ENV_NAME"; then
    echo "  $ENV_NAME payload detected! Synchronizing missing packages..."
    conda env update -f environment.yml --prune
else
    echo "  $ENV_NAME payload not found. Constructing fresh conda environment..."
    conda env create -f environment.yml
fi

echo "  Activating conda environment $ENV_NAME..."
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Check for the vllm wheel mentioned in pyproject.toml
VLLM_WHEEL="vllm-0.18.0+cu126-cp312-cp312-linux_x86_64.whl"
if [ -f "$VLLM_WHEEL" ]; then
    echo "  Installing local vLLM wheel..."
    pip install "$VLLM_WHEEL"
else
    echo "  ⚠️  WARNING: $VLLM_WHEEL not found. Installation may fail if required."
fi

# Install the project locally as well as vllm if available
pip install -e . || echo "  ⚠️  Warning: pip install -e . failed. You may need to install vLLM manually."

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
