#!/bin/bash
#SBATCH --job-name=ds3d_shard
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gpus-per-task=1
#SBATCH --constraint=v100
#SBATCH --time=2:00:00
#SBATCH --array=0-31
# Logs will be placed in slurm/logs/EXP{EXP_ID} via sbatch CLI flags in deploy_dataset.sh
#SBATCH --output=ds3d_shard_%A_%a.out
#SBATCH --error=ds3d_shard_%A_%a.err

# ── Environment ───────────────────────────────────────────────────────────────
set -e
module purge
module load gcc/13.3.0

export SCRATCH_DIR="/scratch1/kelidari"
export HF_HOME="${SCRATCH_DIR}/hf_cache"
export TMPDIR="${SCRATCH_DIR}/tmp"
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONNOUSERSITE=1

mkdir -p "${HF_HOME}"
mkdir -p "${TMPDIR}"
mkdir -p "${SCRATCH_DIR}/dataset_3d/shards/EXP${EXP_ID}"
mkdir -p "${SCRATCH_DIR}/dataset_3d/output/EXP${EXP_ID}"

cd "${SLURM_SUBMIT_DIR:-$HOME/llm-steering}"

# ── Activate Conda ────────────────────────────────────────────────────────────
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate llm_steering
echo "  ✅ Conda environment llm_steering active"

# ── Parameters ────────────────────────────────────────────────────────────────
case "${MODEL_KEY}" in
    "phi35moe") MODEL_NAME="microsoft/Phi-3.5-MoE-instruct" ;;
    "qwen3moe") MODEL_NAME="Qwen/Qwen3-30B-A3B" ;;
    "gptoss")   MODEL_NAME="openai/gpt-oss-20b" ;;
    "mixtral")  MODEL_NAME="mistralai/Mixtral-8x7B-Instruct-v0.1" ;;
    "olmoe")    MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct" ;;
    *)          MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct" ;;
esac

DEVICE="cuda"
STORAGE_DTYPE="float16"
OUTPUT_DIR="${SCRATCH_DIR}/dataset_3d/shards/EXP${EXP_ID}"
NUM_SHARDS=32
SHARD_INDEX=${SLURM_ARRAY_TASK_ID}

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== Shard ${SHARD_INDEX}/${NUM_SHARDS}: $(date) ==="
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model:      ${MODEL_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Output file:${OUTPUT_DIR}/shard_$(printf '%02d' ${SHARD_INDEX}).pt"

echo "=== Torch/CUDA preflight ==="
cat > /tmp/_cuda_preflight_${SHARD_INDEX}.py << 'PYEOF'
import os, torch
print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PYEOF
python -u /tmp/_cuda_preflight_${SHARD_INDEX}.py

python -u -m src.dataset_3d.generate \
    --model "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --storage-dtype "${STORAGE_DTYPE}" \
    --split train \
    --device ${DEVICE} \
    --shard-index ${SHARD_INDEX} \
    --num-shards ${NUM_SHARDS} \
    --wandb-group "EXP${EXP_ID}"

echo "=== Shard ${SHARD_INDEX} done: $(date) ==="
