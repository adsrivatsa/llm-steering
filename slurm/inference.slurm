#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --job-name=moe_inference
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=2
#SBATCH --constraint="a100|l40s|a40"
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Single-task config: one LLM and one dataset.
LLM_NAME="openai/gpt-oss-20b"
DATASET_NAME="squad"

cd "$SLURM_SUBMIT_DIR"

# threading fix
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Adjust this to point to your Python binary if different.
export PY_BIN=/scratch1/adsrivat/llm-steering/.venv/bin/python

echo "Working directory: $(pwd)"
echo "Job started on $(hostname) at $(date)"
echo "Running inference with LLM: $LLM_NAME on dataset: $DATASET_NAME"

# Run your script; assumes main.py accepts --llm and --dataset arguments.
$PY_BIN main.py --llm "$LLM_NAME" --dataset "$DATASET_NAME"

