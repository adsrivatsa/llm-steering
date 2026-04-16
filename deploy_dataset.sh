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
# --- Experiment Configuration ---
EXP_ID="${EXP_ID:-1}"
MODEL_KEY="${MODEL_KEY:-olmoe}"
EXP_LABEL="EXP${EXP_ID}"

mkdir -p "${DATASET_DIR}/output/${EXP_LABEL}"
mkdir -p "${DATASET_DIR}/shards/${EXP_LABEL}"
mkdir -p "${DATASET_DIR}/checkpoints/${EXP_LABEL}"
mkdir -p "slurm/logs/${EXP_LABEL}"
mkdir -p "${SCRATCH}/hf_cache"
echo "  Experiment:  ${EXP_LABEL}"
echo "  Model Key:   ${MODEL_KEY}"
echo "  Output:      ${DATASET_DIR}/output/${EXP_LABEL}"
echo "  Shards:      ${DATASET_DIR}/shards/${EXP_LABEL}"
echo "  Logs:        slurm/logs/${EXP_LABEL}"
echo "  HF cache:    ${SCRATCH}/hf_cache"

# --- Environment Setup ---
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"

ENV_NAME="llm_steering"
# if conda env list | grep -q "$ENV_NAME"; then
#     echo "  $ENV_NAME found. Synchronizing packages..."
#     conda env update -f environment.yml --prune
# else
#     echo "  $ENV_NAME not found. Creating fresh conda environment..."
#     conda env create -f environment.yml
# fi

# conda activate "$ENV_NAME"
# echo "  ✅ Conda environment $ENV_NAME active"

# Check for the vllm wheel mentioned in pyproject.toml
VLLM_WHEEL="vllm-0.18.0+cu126-cp312-cp312-linux_x86_64.whl"
if [ -f "$VLLM_WHEEL" ]; then
    echo "  Installing local vLLM wheel..."
    pip install "$VLLM_WHEEL"
else
    echo "  ⚠️  WARNING: $VLLM_WHEEL not found. Installation may fail if required."
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  3 · Submit SLURM job"
echo "═══════════════════════════════════════════"
JOB_ID=$(sbatch --parsable \
    --export=ALL,HF_TOKEN="${HF_TOKEN}",WANDB_API_KEY="${WANDB_API_KEY}",EXP_ID="${EXP_ID}",MODEL_KEY="${MODEL_KEY}" \
    --output="slurm/logs/${EXP_LABEL}/ds3d_shard_%A_%a.out" \
    --error="slurm/logs/${EXP_LABEL}/ds3d_shard_%A_%a.err" \
    slurm/dataset-3d-parallel.slurm)

echo "  ✅ Job submitted: ${JOB_ID}"
echo ""
echo "  Monitor:"
echo "    squeue -u ${USER}"
echo "    tail -f slurm/logs/${EXP_LABEL}/ds3d_shard_${JOB_ID}_0.out"
