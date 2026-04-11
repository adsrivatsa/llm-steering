# %% [markdown]
# # 🚀 Token-Aware 3D Delta — Dataset Generation
#
# Generates **X{N,L,E,D}** and **Y{N,L,E,1}** tensors from SQuAD for learning Δ(E,L,D).

# %% [markdown]
# ## 1 · Environment Setup

# %%
import os
import sys

miniconda_path = f"{os.environ.get('HOME', '')}/miniconda/bin"
os.environ["PATH"] = f"{miniconda_path}:" + os.environ.get("PATH", "")

# Cache dirs → scratch (absolute path, NOT ~/scratch1)
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch1/kelidari")
os.environ["HF_HOME"] = f"{SCRATCH}/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = f"{SCRATCH}/hf_cache"
os.environ["TMPDIR"] = f"{SCRATCH}/tmp"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

print(f"Conda PATH = {miniconda_path}")
print(f"HF_HOME    = {os.environ['HF_HOME']}")
print(f"TMPDIR     = {os.environ['TMPDIR']}")

# %%
import os

# Preemptively accept Conda TOS to prevent hanging prompts
# !conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
# !conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

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
MODEL      = "allenai/OLMoE-1B-7B-0125-Instruct"
CHUNK_SIZE = 2000
DEVICE     = "cuda"
SCRATCH    = os.environ.get("SCRATCH_DIR", "/scratch1/kelidari")
OUTPUT_DIR = f"{SCRATCH}/dataset_3d/output"
CKPT_DIR   = f"{SCRATCH}/dataset_3d/checkpoints"

# %% [markdown]
# ## 3 · Run Dataset Generation

# %%
!conda run -n llm_steering python -u -m src.dataset_3d.generate \
    --model "{MODEL}" \
    --output-dir "{OUTPUT_DIR}" \
    --checkpoint-dir "{CKPT_DIR}" \
    --chunk-size {CHUNK_SIZE} \
    --device {DEVICE}

# %% [markdown]
# ## 4 · Verification

# %%
!conda run -n llm_steering python -u -c "
import os, glob, torch
d = '{OUTPUT_DIR}'
meta = torch.load(os.path.join(d, 'metadata.pt'), map_location='cpu', weights_only=False)
chunks = sorted(glob.glob(os.path.join(d, 'chunk_*.pt')))
first = torch.load(chunks[0], map_location='cpu', weights_only=True)
X, Y = first['X'], first['Y']
print(f'N={meta[\"n_total\"]}, L={meta[\"layers\"]}, E={meta[\"experts\"]}, D={meta[\"hidden_dim\"]}')
print(f'X shape: {X.shape}')
print(f'Y shape: {Y.shape}')
assert X.dim() == 4
total = sum(os.path.getsize(f) for f in chunks)
print(f'Chunks: {len(chunks)}, Total: {total/1e9:.2f} GB')
print('All checks passed!')
"
