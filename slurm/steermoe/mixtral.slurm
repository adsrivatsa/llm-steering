#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --job-name=collect-olmoe-steermoe-inference
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=3
#SBATCH --partition=gpu
#SBATCH --array=0-35
#SBATCH --constraint="a100|l40s|a40"
#SBATCH --mem=24G
#SBATCH --time=08:00:00

ACTIVATED=(0 1 2 4 8 16)
DEACTIVATED=(0 8 16 32 64 128)
N_A=${#ACTIVATED[@]}
N_D=${#DEACTIVATED[@]}

LLM_NAME="mistralai/Mixtral-8x7B-Instruct-v0.1"
TASK="faithfulness"

cd "$SLURM_SUBMIT_DIR"

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
echo "Array task id: $SLURM_ARRAY_TASK_ID -> experts_activated=$N_ACTIVATED experts_deactivated=$N_DEACTIVATED (pass_name=a${N_ACTIVATED}_d${N_DEACTIVATED})"

$PY_BIN -m src.steermoe.main \
  --llm "$LLM_NAME" \
  --task "$TASK" \
  --experts-activated "$N_ACTIVATED" \
  --experts-deactivated "$N_DEACTIVATED"
