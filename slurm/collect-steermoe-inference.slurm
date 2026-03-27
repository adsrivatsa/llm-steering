#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --job-name=collect-steermoe-inference
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

export PY_BIN="/scr2/${SLURM_JOB_USER}/llm-steering/.venv/bin/python"

echo "Working directory: $(pwd)"
echo "Job started on $(hostname) at $(date)"
echo "Running inference with LLM: $LLM_NAME on task: $TASK"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  MODE="steered"
  EXTRA_ARGS=""
else
  MODE="unsteered"
  EXTRA_ARGS="--no-steering"
fi

echo "Array task id: $SLURM_ARRAY_TASK_ID ($MODE)"

$PY_BIN -m src.steermoe.main --llm "$LLM_NAME" --task "$TASK" $EXTRA_ARGS

