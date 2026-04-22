#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --job-name=squad-steermoe3d-inference
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --array=0-24
#SBATCH --constraint="a100|v100|a40"
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# 5x5 grid of parameters
A_VALS=(0 2 4 8 16)
D_VALS=(0 32 64 128 256)

A_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
D_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

experts_activated=${A_VALS[$A_IDX]}
experts_deactivated=${D_VALS[$D_IDX]}

# Accept env vars for overrides
MODEL_KEY=${MODEL_KEY:-"allenai/OLMoE-1B-7B-0125-Instruct"}
DELTA_PATH=${DELTA_PATH:-"/scratch1/${USER}/dataset_3d/output/EXP1/trained_delta.pt"}
INFERENCE_DIR=${INFERENCE_DIR:-"/scratch1/${USER}/dataset_3d/inference/squad"}

echo "Running inferences for SteerMoE3D on SQuAD."
echo "Model: $MODEL_KEY"
echo "Delta: $DELTA_PATH"
echo "Activated (A): $experts_activated"
echo "Deactivated (D): $experts_deactivated"

export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate llm_steering

conda run -n llm_steering python -u -m src.steermoe3d.main \
    --task squad \
    --llm "$MODEL_KEY" \
    --delta-path "$DELTA_PATH" \
    --inference-dir "$INFERENCE_DIR" \
    --experts-activated $experts_activated \
    --experts-deactivated $experts_deactivated
