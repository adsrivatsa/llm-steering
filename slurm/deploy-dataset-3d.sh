#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# deploy-dataset-3d.sh — Run from your LOCAL Mac
#
# This script:
#   1. Commits & pushes your local changes
#   2. SSHs into CARC
#   3. Pulls the latest code
#   4. Submits the SLURM job
#
# Usage:
#   bash slurm/deploy-dataset-3d.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

CARC_USER="kelidari"
CARC_HOST="discovery.usc.edu"
CARC_PROJECT_DIR="~/llm-steering"
CARC_SCRATCH="/scratch1/kelidari"

echo "═══════════════════════════════════════════"
echo "  Step 1: Commit & Push (local)"
echo "═══════════════════════════════════════════"
cd "$(dirname "$0")/.."

git add -A
git diff --cached --quiet && echo "  Nothing to commit, skipping..." || \
    git commit -m "Add 3D delta dataset generation pipeline"
git push
echo "  ✅ Pushed to remote"

echo ""
echo "═══════════════════════════════════════════"
echo "  Step 2: Pull & Submit on CARC"
echo "═══════════════════════════════════════════"

ssh "${CARC_USER}@${CARC_HOST}" bash -s "${CARC_PROJECT_DIR}" "${CARC_SCRATCH}" << 'REMOTE_SCRIPT'
set -e
PROJECT_DIR="$1"
SCRATCH="$2"

cd "$PROJECT_DIR"

echo "  Pulling latest code..."
git pull --ff-only

echo "  Creating scratch dirs..."
mkdir -p "${SCRATCH}/dataset_3d/output"
mkdir -p "${SCRATCH}/dataset_3d/checkpoints"
mkdir -p "${SCRATCH}/hf_cache"

echo "  Submitting SLURM job..."
JOB_ID=$(sbatch --parsable slurm/dataset-3d.slurm)
echo "  ✅ Job submitted: ${JOB_ID}"
echo ""
echo "  Monitor with:"
echo "    squeue -u $(whoami)"
echo "    tail -f dataset3d_${JOB_ID}.out"
REMOTE_SCRIPT

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Done! Job submitted on CARC."
echo "═══════════════════════════════════════════"
echo ""
echo "  SSH in to monitor:"
echo "    ssh ${CARC_USER}@${CARC_HOST}"
echo "    squeue -u ${CARC_USER}"
