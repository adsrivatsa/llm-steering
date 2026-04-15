# %% [markdown]
# # 🚀 Sample-Level 3D Dataset Generation
#
# Generates one row per `(sample, layer)`:
# - `D`: mean-pooled hidden state of `x1` at layer `l`
# - `y1`: mean-pooled softmax router scores of `x1` at layer `l`
# - `y2`: mean-pooled softmax router scores of `x2` at layer `l`
#
# Prompts:
# - `x1 = "Document {context} Question {question}"`
# - `x2 = "Question {question}"`
#
# Output file: `dataset.pt` with `{"D", "y1", "y2", "metadata"}`.

# %% [markdown]
# ## 1 · Environment Setup

# %%
import os
import sys

miniconda_path = f"{os.environ.get('HOME', '')}/miniconda/bin"
os.environ["PATH"] = f"{miniconda_path}:" + os.environ.get("PATH", "")

SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch1/kelidari")
os.environ["HF_HOME"] = f"{SCRATCH}/hf_cache"
os.environ["TMPDIR"] = f"{SCRATCH}/tmp"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

print(f"Conda PATH = {miniconda_path}")
print(f"HF_HOME    = {os.environ['HF_HOME']}")
print(f"TMPDIR     = {os.environ['TMPDIR']}")

# %%
import os

env_name = "llm_steering"
print("Checking environment status...")
res = os.system(f"conda env list | grep {env_name} > /dev/null")
if res == 0:
    print(f"{env_name} exists! Synchronizing any missing packages ...")
    os.system("conda env update -f environment.yml --prune")
else:
    print(f"{env_name} does not exist. Creating fresh environment...")
    os.system("conda env create -f environment.yml")

# %% [markdown]
# ## 2 · Configuration

# %%
MODEL_NAME = "allenai/OLMoE-1B-7B-0125-Instruct"
DEVICE = "cuda"
STORAGE_DTYPE = "float16"
MAX_EXAMPLES = None  # Set to 5 for a quick manual smoke run.
CHECKPOINT_INTERVAL = 100
SPLIT = "train"

SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch1/kelidari")
OUTPUT_DIR = f"{SCRATCH}/dataset_3d/output"
CHECKPOINT_DIR = f"{SCRATCH}/dataset_3d/checkpoints"
DATASET_FILE = f"{OUTPUT_DIR}/dataset.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Model          = {MODEL_NAME}")
print(f"Split          = {SPLIT}")
print(f"Storage dtype  = {STORAGE_DTYPE}")
print(f"Output file    = {DATASET_FILE}")
print(f"Checkpoint dir = {CHECKPOINT_DIR}")
print(f"Max examples   = {MAX_EXAMPLES or 'all (~87K)'}")

# %%
# OLMoE-1B-7B-0125-Instruct defaults
L = 16
E = 64
D = 2048
S = MAX_EXAMPLES or 87599  # SQuAD train size
rows = S * L
bytes_per_element = 2 if STORAGE_DTYPE == "float16" else 4
per_row_bytes = (D + 2 * E) * bytes_per_element
total_gb = rows * per_row_bytes / 1e9

print(f"Rows (S*L):          {rows:,}")
print(f"Per-row bytes:       {per_row_bytes:,}")
print(f"Estimated dataset:   {total_gb:.2f} GB ({STORAGE_DTYPE})")
print("Formula: S * L * (D + 2E) * bytes_per_element")

# %%
import torch
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda.is_available={torch.cuda.is_available()}")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA requested but unavailable. Run this in a GPU allocation and use a CUDA-enabled torch build."
    )

# %% [markdown]
# ## 3 · Run Dataset Generation

# %%
MAX_EXAMPLES_ARG = f"--max-examples {MAX_EXAMPLES}" if MAX_EXAMPLES is not None else ""
!conda run -n llm_steering python -u -m src.dataset_3d.generate \
    --model "{MODEL_NAME}" \
    --output-dir "{OUTPUT_DIR}" \
    --checkpoint-dir "{CHECKPOINT_DIR}" \
    --checkpoint-interval {CHECKPOINT_INTERVAL} \
    --storage-dtype "{STORAGE_DTYPE}" \
    --split "{SPLIT}" \
    --device "{DEVICE}" {MAX_EXAMPLES_ARG}

# %% [markdown]
# ## 4 · Verification

# %%
# %%writefile verify_dataset.py
import subprocess, textwrap, pathlib
_verify_code = textwrap.dedent("""\
    import sys
    import torch

    dataset_path = sys.argv[1]
    obj = torch.load(dataset_path, map_location='cpu', weights_only=False)
    D, y1, y2, meta = obj['D'], obj['y1'], obj['y2'], obj['metadata']
    rows = meta['rows']
    hidden_dim = meta['hidden_dim']
    experts = meta['experts']
    print(f'Loaded: {dataset_path}')
    print(f'D shape:  {tuple(D.shape)}')
    print(f'y1 shape: {tuple(y1.shape)}')
    print(f'y2 shape: {tuple(y2.shape)}')
    print(f"Metadata rows={rows}, D={hidden_dim}, E={experts}, L={meta['layers']}, S={meta['num_samples']}")
    assert D.shape == (rows, hidden_dim)
    assert y1.shape == (rows, experts)
    assert y2.shape == (rows, experts)
    assert D.dtype == y1.dtype == y2.dtype
    assert not torch.isnan(D.float()).any()
    assert not torch.isnan(y1.float()).any()
    assert not torch.isnan(y2.float()).any()
    sum_y1 = y1.float().sum(dim=-1)
    sum_y2 = y2.float().sum(dim=-1)
    print(f'y1 row-sum mean: {sum_y1.mean().item():.6f}')
    print(f'y2 row-sum mean: {sum_y2.mean().item():.6f}')
    print('Verification checks passed.')
""")
pathlib.Path("verify_dataset.py").write_text(_verify_code)

# %%
!conda run -n llm_steering python -u verify_dataset.py "{DATASET_FILE}"

