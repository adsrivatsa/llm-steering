#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# run_squad_inference.sh — Submits the 25-array job for SQuAD testing
# ══════════════════════════════════════════════════════════════════════════════
set -e

PROJECT_DIR="/home1/${USER}/llm-steering"
cd "${PROJECT_DIR}"

export MODEL_KEY="allenai/OLMoE-1B-7B-0125-Instruct"
export DELTA_PATH="/scratch1/${USER}/dataset_3d/output/EXP1/trained_delta.pt"
export INFERENCE_DIR="/scratch1/${USER}/dataset_3d/inference/squad"

mkdir -p "logs"

echo "Submitting 25-job array for SQuAD inference to SLURM..."
JOB_ID=$(sbatch --parsable \
    --export=ALL,MODEL_KEY="${MODEL_KEY}",DELTA_PATH="${DELTA_PATH}",INFERENCE_DIR="${INFERENCE_DIR}" \
    slurm/steermoe3d_inference_squad.slurm)

echo "✅ Array job submitted: ${JOB_ID}"
echo "Monitor with: squeue -u ${USER}"
echo "Check outputs with: tail -f logs/squad-steermoe3d-inference-${JOB_ID}_0.out"
