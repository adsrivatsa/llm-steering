#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --job-name=toksteermoe-qwen-inference
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --array=0-35
#SBATCH --constraint="a100|l40s|a40"
#SBATCH --mem=24G
#SBATCH --time=08:00:00

ACTIVATED=(0 1 3 6 12 24)
DEACTIVATED=(0 24 48 96 192 384)
N_A=${#ACTIVATED[@]}
N_D=${#DEACTIVATED[@]}

LLM_NAME="Qwen/Qwen3-30B-A3B"
TASK="faithfulness"

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/13.3.0
module load python/3.12
export TRITON_CACHE_DIR=/scratch1/${SLURM_JOB_USER}/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

# threading fix
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

export HF_HOME=/scratch1/${SLURM_JOB_USER}/.cache/huggingface
export PY_BIN="$(pwd)/.venv/bin/python"

IA=$((SLURM_ARRAY_TASK_ID / N_D))
ID=$((SLURM_ARRAY_TASK_ID % N_D))
N_ACTIVATED="${ACTIVATED[$IA]}"
N_DEACTIVATED="${DEACTIVATED[$ID]}"

echo "Working directory: $(pwd)"
echo "Job started on $(hostname) at $(date)"
echo "Running inference with LLM: $LLM_NAME on task: $TASK"
echo "Array task id: $SLURM_ARRAY_TASK_ID -> experts_activated=$N_ACTIVATED experts_deactivated=$N_DEACTIVATED (pass_name=toksteermoe_a${N_ACTIVATED}_d${N_DEACTIVATED})"

$PY_BIN -m src.toksteermoe.main \
  --llm "$LLM_NAME" \
  --task "$TASK" \
  --experts-activated "$N_ACTIVATED" \
  --experts-deactivated "$N_DEACTIVATED"