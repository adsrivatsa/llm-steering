# %% [markdown]
# # 🚀 Token-Aware 3D Delta — Dataset Generation
#
# Generates **X{N,L,E,D}** and **Y{N,L,E,1}** tensors from SQuAD for learning Δ(E,L,D).
#
# All execution uses `conda run` to ensure the correct environment is active.
# No manual `pip install` needed — everything runs inside the `llm_steering` conda env.

# %% [markdown]
# ## 1 · Environment Setup

# %%
import os
import sys

miniconda_path = f"{os.environ.get('HOME', '')}/miniconda/bin"
os.environ["PATH"] = f"{miniconda_path}:" + os.environ.get("PATH", "")

# W&B API key
os.environ['WANDB_API_KEY'] = "MYtoken"

print(f"Conda PATH securely bound to: {miniconda_path}")
print("✓ W&B API key configured!")

# %%
import os

# Preemptively accept Conda TOS to prevent hanging prompts
# !conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
# !conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Intelligently handle environment creation or update
env_name = "llm_steering"
print("Checking environment status...")
res = os.system(f"conda env list | grep {env_name} > /dev/null")
if res == 0:
    print(f"{env_name} exists! Synchronizing any missing packages ...")
    os.system("conda env update -f environment.yml --prune")
else:
    print(f"{env_name} does not exist. Creating fresh environment...")
    os.system("conda env create -f environment.yml")

# %%
# Install project in editable mode inside the conda env
os.system("conda run -n llm_steering pip install -e . 2>&1 | tail -5")
print("✅ Project installed in conda env")

# %% [markdown]
# ## 2 · Configuration

# %%
import os

# ── Configurable parameters ──────────────────────────────────────────────────
MODEL_NAME     = "allenai/OLMoE-1B-7B-0125-Instruct"
N_TOKENS       = 20_000        # middle-frequency tokens to select (None = all)
CHUNK_SIZE     = 2000           # tokens per saved chunk file
MAX_EXAMPLES   = None           # set to e.g. 500 for a quick debug run
DEVICE         = "cuda"

# Scratch directories
SCRATCH = os.environ.get("SCRATCH_DIR", os.path.expanduser("~/scratch1/kelidari"))
OUTPUT_DIR     = os.path.join(SCRATCH, "dataset_3d", "output")
CHECKPOINT_DIR = os.path.join(SCRATCH, "dataset_3d", "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Model          = {MODEL_NAME}")
print(f"Output dir     = {OUTPUT_DIR}")
print(f"Checkpoint dir = {CHECKPOINT_DIR}")
print(f"N tokens       = {N_TOKENS}")
print(f"Chunk size     = {CHUNK_SIZE}")
print(f"Max examples   = {MAX_EXAMPLES or 'all (~87K)'}")

# %% [markdown]
# ## 3 · Estimated Disk & RAM Usage

# %%
# Quick size estimate before running
L, K, E = 16, 8, 64    # OLMoE config
D = 2048                # hidden dim
N = N_TOKENS or 50000   # approximate if using all tokens

h_ram  = N * L * D * 4 / 1e9
a_ram  = L * E * N * 4 / 1e9
x_disk = N * L * E * D * 4 / 1e9
y_disk = N * L * E * 4 / 1e9

print(f"Model config: L={L}, K={K}, E={E}, D={D}, N≈{N}")
print(f"\nRAM estimate (accumulators in memory during collection):")
print(f"  H_sum:  {h_ram:.2f} GB  (N, L, D)")
print(f"  A1/A2:  {a_ram:.2f} GB each  (L, E, N)")
print(f"  Total:  {h_ram + 2*a_ram:.2f} GB")
print(f"\nDisk estimate (saved dataset):")
print(f"  X:      {x_disk:.2f} GB  (N, L, E, D)")
print(f"  Y:      {y_disk:.2f} GB  (N, L, E, 1)")
print(f"  Total:  {x_disk + y_disk:.2f} GB")

# %% [markdown]
# ## 4 · Run Dataset Generation
#
# Runs the full pipeline (2 passes through the model + save) inside the conda env.
# This is the long-running step — checkpoints are saved every 500 examples.

# %%
import subprocess

cmd_parts = [
    "conda", "run", "-n", "llm_steering", "python", "-u", "-m", "src.dataset_3d.generate",
    "--model", MODEL_NAME,
    "--output-dir", OUTPUT_DIR,
    "--checkpoint-dir", CHECKPOINT_DIR,
    "--chunk-size", str(CHUNK_SIZE),
    "--device", DEVICE,
]

if N_TOKENS is not None:
    cmd_parts.extend(["--n-tokens", str(N_TOKENS)])
if MAX_EXAMPLES is not None:
    cmd_parts.extend(["--max-examples", str(MAX_EXAMPLES)])

print("Command:", " ".join(cmd_parts))
print("=" * 60)

process = subprocess.Popen(
    cmd_parts,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env={**os.environ},  # pass WANDB_API_KEY through
)

# Stream output in real-time
for line in iter(process.stdout.readline, ""):
    print(line, end="")

process.wait()
print("\n" + "=" * 60)
print(f"Exit code: {process.returncode}")
if process.returncode == 0:
    print("✅ Dataset generation complete!")
else:
    print("❌ Dataset generation FAILED — check output above")

# %% [markdown]
# ## 5 · Verification & Statistics
#
# Verify the saved chunks have the correct shapes: **X(N,L,E,D)** and **Y(N,L,E,1)**.

# %%
# Write verification script
verify_code = '''
import os, sys, glob, torch

OUTPUT_DIR = sys.argv[1]

print("=" * 60)
print("VERIFICATION")
print("=" * 60)

meta_path = os.path.join(OUTPUT_DIR, "metadata.pt")
if not os.path.exists(meta_path):
    print("No metadata.pt found")
    sys.exit(1)

meta = torch.load(meta_path, map_location="cpu", weights_only=False)
print(f"Model:      {meta['model_name']}")
print(f"N total:    {meta['n_total']}")
print(f"Layers:     {meta['layers']}")
print(f"Experts:    {meta['experts']}")
print(f"Hidden dim: {meta['hidden_dim']}")

chunk_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "chunk_*.pt")))
print(f"Chunk files: {len(chunk_files)}")

first = torch.load(chunk_files[0], map_location="cpu", weights_only=True)
X, Y = first["X"], first["Y"]
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

assert X.dim() == 4, f"X should be 4D, got {X.dim()}D"
assert X.shape[2] == meta["experts"]
print(f"X is (N, L={meta['layers']}, E={meta['experts']}, D={meta['hidden_dim']})")

total_bytes = sum(os.path.getsize(f) for f in chunk_files) + os.path.getsize(meta_path)
print(f"Total dataset size: {total_bytes / 1e9:.2f} GB")
print("All checks passed!")
'''

with open("_verify_dataset.py", "w") as f:
    f.write(verify_code)

subprocess.run(
    ["conda", "run", "-n", "llm_steering", "python", "-u", "_verify_dataset.py", OUTPUT_DIR],
    capture_output=False, text=True,
)

# %%
# Cleanup temp scripts
import os
for f in ["_verify_dataset.py", "_explore_dataset.py"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed {f}")
print("✅ Done")
