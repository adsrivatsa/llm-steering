# Risk-Scaled and Token-Aware MoE Steering

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C.svg)](https://pytorch.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.18.0-green.svg)](https://github.com/vllm-project/vllm)

This repository contains the implementation of risk-scaled and token-aware steering for Mixture-of-Experts (MoE) language models. The codebase supports:

- Activation collection on SQuAD prompts.
- Delta computation between context-rich and context-lean prompt modes.
- Two steering methods at inference time:
  - `steermoe`: global expert steering.
  - `toksteermoe`: token-specific steering with fallback global weights.
- Evaluation on six faithfulness benchmarks.
- Heatmap visualization over `(experts_activated, experts_deactivated)` sweeps.

## Repository Structure

- `src/activation/`: collects routing activations and saves checkpoints.
- `src/steermoe/`: runs global steering inference.
- `src/toksteermoe/`: runs token-aware steering inference.
- `src/benchmark/`: dataset loaders, prompts, inference loops, and scoring.
- `src/visualization/`: heatmap generation from saved inference outputs.
- `slurm/`: reproducible cluster job scripts used for experiments.
- `inference/`: saved JSONL outputs (example runs already included).

## Environment Setup

### 1) System Requirements

- OS: Linux (required for the pinned local vLLM wheel in `pyproject.toml`).
- Python: `3.12.x`
- CUDA GPU: required for practical runs. We used L40S/A40 on CARC.
- Disk/network: enough to cache model weights and benchmark datasets.

### 2) Create Environment

```bash
git clone https://github.com/adsrivatsa/llm-steering
cd llm-steering

uv sync
```

### 3) Required Local Wheel (only for CARC because of Rocky Linux)

This repository expects a local wheel source for vLLM:

- `./vllm-0.18.0+cu126-cp312-cp312-linux_x86_64.whl`

If this wheel is missing, installation will fail until the file is added or the source entry in `pyproject.toml` is changed.

## Device/System Used for Runs

The provided `slurm/*.slurm` scripts document the run environment used in this project:

- Cluster scheduler: Slurm.
- Module stack: `gcc/13.3.0`, `python/3.12`.
- GPU partitions/constraints: `a100`, `l40s`, `a40` (model-dependent).
- Typical resources:
  - activation collection: 1 GPU, 5 CPUs, 24 GB RAM.
  - steering sweeps: 1-2 GPUs, 5 CPUs, 24 GB RAM, array jobs.
- Cache paths:
  - `HF_HOME=/scratch1/${SLURM_JOB_USER}/.cache/huggingface`
  - `TRITON_CACHE_DIR=/scratch1/${SLURM_JOB_USER}/.triton_cache`

## How to Run the Code

All commands below assume the venv is active.

### Step 1: Collect Activations

```bash
python -m src.activation.main \
  --task faithfulness \
  --llm allenai/OLMoE-1B-7B-0125-Instruct \
  --checkpoint-dir activations
```

This produces activation checkpoints under `activations/squad/` for `x1` and `x2` passes.

### Step 2: Run Steering Inference

Global steering (`steermoe`):

```bash
python -m src.steermoe.main \
  --task faithfulness \
  --llm allenai/OLMoE-1B-7B-0125-Instruct \
  --activations-dir activations \
  --inference-dir inference \
  --experts-activated 8 \
  --experts-deactivated 64
```

Token-aware steering (`toksteermoe`):

```bash
python -m src.toksteermoe.main \
  --task faithfulness \
  --llm allenai/OLMoE-1B-7B-0125-Instruct \
  --activations-dir activations \
  --inference-dir inference \
  --experts-activated 8 \
  --experts-deactivated 64
```

Each run writes benchmark outputs to:

- `inference/<safe_model_name>/<dataset>/<pass_name>.jsonl`

where `pass_name` is either:

- `steermoe_a{A}_d{D}` or
- `toksteermoe_a{A}_d{D}`

### Step 3: Generate Heatmaps

```bash
python -m src.visualization.heatmap \
  --task faithfulness \
  --algorithm steermoe \
  --llm allenai/OLMoE-1B-7B-0125-Instruct \
  --inference-dir inference \
  --visualization-dir visualization
```

Use `--algorithm toksteermoe` for token-aware results.

### Step 4: Cluster Reproduction (Slurm)

Ready-to-run scripts are included:

- Activation collection: `slurm/collect-activations.slurm`
- Global sweep scripts: `slurm/steermoe/*.slurm`
- Token-aware sweep scripts: `slurm/toksteermoe/*.slurm`

These scripts launch array jobs over predefined `(activated, deactivated)` grids and call the same Python modules shown above.

## How Results Are Generated

This section maps the exact code path from inputs to final metrics.

1. `src.activation.main`
   - Loads SQuAD (`src/activation/dataset.py`).
   - Builds two prompt variants per example (`x1` with document + question, `x2` with question only).
   - Captures MoE routing activations from custom vLLM model registration.
   - Saves tensors `A` and `N` checkpoints via `src/checkpoint.py`.

2. `src.steermoe.main` / `src.toksteermoe.main`
   - Loads `x1` and `x2` checkpoints.
   - Computes risk delta from activation differences.
   - Converts deltas into expert manual weights.
   - Applies steering through custom model hooks in `src/*/modelling/*.py`.
   - Runs six benchmark inference modules:
     - `faitheval_counterfactual`
     - `faitheval_unanswerable`
     - `faitheval_inconsistent`
     - `cf_trivia_qa`
     - `mquake`
     - `mctest`

3. Benchmark scoring (`src/benchmark/*.py`)
   - Each benchmark writes one JSONL line per example with prompt, model output, and gold labels.
   - Each benchmark has its own `score(...)` function.
   - Reported metric is accuracy for each dataset.

4. Visualization (`src/visualization/heatmap.py`)
   - Reads every JSONL file matching `steermoe_a*_d*` or `toksteermoe_a*_d*`.
   - Recomputes benchmark accuracy via each benchmark `score(...)` function.
   - Produces per-benchmark and cumulative heatmaps in `visualization/heatmap/<model_name>/`.

## Datasets Used

Loaded dynamically at runtime by dataset classes in `src/benchmark/dataset.py` and `src/activation/dataset.py`:

- `rajpurkar/squad`
- `Salesforce/FaithEval-counterfactual-v1.0`
- `Salesforce/FaithEval-unanswerable-v1.0`
- `Salesforce/FaithEval-inconsistent-v1.0`
- CF-TriviaQA JSONL from Google Research GitHub
- MQuAKE datasets from Princeton NLP GitHub
- MCTest parquet splits from Hugging Face

## Misc Notes

- Inference is resumable: each benchmark appends to partially completed JSONL outputs.
- Activation collection is checkpointed (`checkpoint.INTERVAL = 50`).
- Model registration mode is selected using `LLM_REGISTRATION` in each main module.
- Included `inference/` files provide concrete examples of generated outputs.

## Authors

Abhinav Srivatsa, Carl Cheng, Nima Kelidari, Bhushan Shankar Halasagi, Harsh Sharma (CSCI 544)
