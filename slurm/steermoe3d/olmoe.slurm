#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --job-name=collect-olmoe-steermoe3d-inference
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --array=0-35
#SBATCH --constraint="a100|l40s|a40"
#SBATCH --mem=24G
#SBATCH --time=08:00:00

# Top-K settings (using 0..64 for Olmoe)
ACTIVATED=(0 2 4 8 16 32)
DEACTIVATED=(0 4 8 16 32 64)
N_A=${#ACTIVATED[@]}
N_D=${#DEACTIVATED[@]}

LLM_NAME="allenai/OLMoE-1B-7B-0125-Instruct"
TASK="faithfulness"
# IMPORTANT: Adjust this if your array produced the file elsewhere
DELTA_PATH="/scratch1/${SLURM_JOB_USER}/dataset_3d/trained_delta.pt"

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/13.3.0
module load python/3.12
export TRITON_CACHE_DIR=/scratch1/${SLURM_JOB_USER}/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

# threading fix
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

export HF_HOME=/scratch1/${SLURM_JOB_USER}/.cache/huggingface

# Use conda here since we swapped to conda in train scripts, but maintain fallback
if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    conda activate llm_steering
    PY_BIN="python"
else
    PY_BIN="$(pwd)/.venv/bin/python"
fi


IA=$((SLURM_ARRAY_TASK_ID / N_D))
ID=$((SLURM_ARRAY_TASK_ID % N_D))
N_ACTIVATED="${ACTIVATED[$IA]}"
N_DEACTIVATED="${DEACTIVATED[$ID]}"

echo "Working directory: $(pwd)"
echo "Job started on $(hostname) at $(date)"
echo "Running inference with LLM: $LLM_NAME on task: $TASK (3D SteerMoE Mode)"
echo "Array task id: $SLURM_ARRAY_TASK_ID -> experts_activated=$N_ACTIVATED experts_deactivated=$N_DEACTIVATED (pass_name=steermoe3d_a${N_ACTIVATED}_d${N_DEACTIVATED})"

$PY_BIN -u -m src.steermoe3d.main \
  --llm "$LLM_NAME" \
  --task "$TASK" \
  --delta-path "$DELTA_PATH" \
  --experts-activated "$N_ACTIVATED" \
  --experts-deactivated "$N_DEACTIVATED"
