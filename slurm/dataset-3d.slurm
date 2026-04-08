#!/bin/bash
#SBATCH --job-name=dataset3d
#SBATCH --account=kelidari_1006
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load gcc/13.3.0 python/3.12.5

export SCRATCH_DIR="/scratch1/kelidari"
export HF_HOME="${SCRATCH_DIR}/hf_cache"
export HF_TOKEN="${HF_TOKEN}"
export TRANSFORMERS_CACHE="${SCRATCH_DIR}/hf_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /home1/kelidari/llm-steering

# ── Activate venv ─────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== Starting dataset generation: $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model: allenai/OLMoE-1B-7B-0125-Instruct"

python -m src.dataset_3d.generate \
    --model "allenai/OLMoE-1B-7B-0125-Instruct" \
    --output-dir "${SCRATCH_DIR}/dataset_3d/output" \
    --checkpoint-dir "${SCRATCH_DIR}/dataset_3d/checkpoints" \
    --n-tokens 20000 \
    --chunk-size 2000 \
    --device cuda

echo "=== Done: $(date) ==="
