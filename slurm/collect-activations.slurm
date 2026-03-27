#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --job-name=collect-activations
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --constraint="a100|l40s|a40"
#SBATCH --mem=24G
#SBATCH --time=24:00:00

LLM_NAME="allenai/OLMoE-1B-7B-0125-Instruct"
TASK="faithfulness"

cd "$SLURM_SUBMIT_DIR"

# threading fix
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

export HF_HOME=/scratch1/${SLURM_JOB_USER}/.cache/huggingface
export PY_BIN="$(pwd)/.venv/bin/python"

echo "Working directory: $(pwd)"
echo "Job started on $(hostname) at $(date)"
echo "Running inference with LLM: $LLM_NAME on task: $TASK"

$PY_BIN -m src.activation.main --llm "$LLM_NAME" --task "$TASK"

