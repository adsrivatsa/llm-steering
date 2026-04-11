# %% [markdown]
# # 🚀 Token-Aware 3D Delta — Dataset Generation
#
# Generates X{N,L,E,D} and Y{N,L,E,1} tensors from SQuAD for learning Δ(E,L,D).
#
# **Run on CARC** via the companion `slurm/dataset-3d.slurm`, or cell-by-cell
# in VS Code (the `# %%` markers are recognised as notebook cells).

# %% [markdown]
# ## 1 · Environment Setup

# %%
import os, sys, subprocess

# HuggingFace token — set yours here or via env
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = ""  # <-- paste token if needed

# Cache dirs (use scratch on CARC to avoid quota issues)
SCRATCH = os.environ.get("SCRATCH_DIR", os.path.expanduser("~/scratch1/kelidari"))
os.environ["HF_HOME"] = os.path.join(SCRATCH, "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(SCRATCH, "hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

print(f"HF_HOME       = {os.environ['HF_HOME']}")
print(f"SCRATCH       = {SCRATCH}")
print(f"Python        = {sys.executable}")

# %% [markdown]
# ## 2 · Configuration

# %%
# ── Configurable parameters ──────────────────────────────────────────────────
MODEL_NAME     = "allenai/OLMoE-1B-7B-0125-Instruct"
N_TOKENS       = 20_000        # middle-frequency tokens to select
CHUNK_SIZE     = 2000           # tokens per saved chunk file (each chunk is ~1.7 GB with E=64)
MAX_EXAMPLES   = None          # set to e.g. 500 for a quick debug run
CKPT_INTERVAL  = 500           # save checkpoint every N examples
DEVICE         = "cuda"

OUTPUT_DIR     = os.path.join(SCRATCH, "dataset_3d", "output")
CHECKPOINT_DIR = os.path.join(SCRATCH, "dataset_3d", "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Model          = {MODEL_NAME}")
print(f"Output dir     = {OUTPUT_DIR}")
print(f"Checkpoint dir = {CHECKPOINT_DIR}")
print(f"N tokens       = {N_TOKENS}")
print(f"Max examples   = {MAX_EXAMPLES or 'all (~87K)'}")

# %% [markdown]
# ## 3 · Imports

# %%
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
from src.dataset_3d.generate import (
    MOE_EXPERT_CONFIG,
    load_squad,
    select_middle_tokens,
    build_prompts,
    collect_pass_data,
    compute_and_save,
)

print(f"torch  = {torch.__version__}")
print(f"CUDA   = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    = {torch.cuda.get_device_name(0)}")
    print(f"VRAM   = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 4 · Load Dataset & Select Tokens

# %%
dataset = load_squad()
if MAX_EXAMPLES:
    dataset = dataset.select(range(min(MAX_EXAMPLES, len(dataset))))
print(f"SQuAD examples: {len(dataset)}")

# %%
hf_token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

selected_set, tid2idx, idx2tid, freq_info = select_middle_tokens(
    dataset, tokenizer, N_TOKENS
)
print(f"Token selection: {freq_info}")

# %%
# Build prompts
prompts_x1, prompts_x2 = build_prompts(dataset)
print(f"Prompts built: {len(prompts_x1)} x1, {len(prompts_x2)} x2")

# Estimate RAM for accumulators
layers, top_k, experts = MOE_EXPERT_CONFIG[MODEL_NAME]
hidden_dim = 2048  # OLMoE hidden size
n_sel = len(tid2idx)
h_ram = n_sel * layers * hidden_dim * 4 / 1e9
a_ram = layers * experts * n_sel * 4 / 1e9
x_disk = n_sel * layers * experts * hidden_dim * 4 / 1e9
y_disk = n_sel * layers * experts * 4 / 1e9
print(f"\nModel config: L={layers}, K={top_k}, E={experts}, D={hidden_dim}")
print(f"RAM estimate (accumulators):")
print(f"  H_sum:  {h_ram:.2f} GB  (N, L, D)")
print(f"  A1/A2:  {a_ram:.2f} GB each")
print(f"  Total:  {h_ram + 2*a_ram:.2f} GB")
print(f"Disk estimate (saved dataset):")
print(f"  X:      {x_disk:.2f} GB  (N, L, E, D)")
print(f"  Y:      {y_disk:.2f} GB  (N, L, E, 1)")
print(f"  Total:  {x_disk + y_disk:.2f} GB")

# %% [markdown]
# ## 5 · Load Model

# %%
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token,
    trust_remote_code=True,
)
model.eval()
print(f"Hidden dim: {model.config.hidden_size}")
print(f"Model loaded on: {next(model.parameters()).device}")

# %% [markdown]
# ## 6 · Pass 1 — x1 (Document + Question)
#
# Collects **hidden states** (for X) and **router activations** (for Y).
# This is the slowest step. Checkpoints save every `CKPT_INTERVAL` examples.

# %%
H1, A1, N1 = collect_pass_data(
    model, tokenizer, prompts_x1, tid2idx,
    layers, experts, top_k, DEVICE,
    pass_name="x1",
    collect_hidden_states=True,
    checkpoint_dir=CHECKPOINT_DIR,
    checkpoint_interval=CKPT_INTERVAL,
)
print(f"Pass x1 done. Tokens with ≥1 occurrence: {(N1 > 0).sum().item()}")

# %% [markdown]
# ## 7 · Pass 2 — x2 (Question Only)
#
# Collects **router activations only** (no hidden states needed).

# %%
_, A2, N2 = collect_pass_data(
    model, tokenizer, prompts_x2, tid2idx,
    layers, experts, top_k, DEVICE,
    pass_name="x2",
    collect_hidden_states=False,
    checkpoint_dir=CHECKPOINT_DIR,
    checkpoint_interval=CKPT_INTERVAL,
)
print(f"Pass x2 done. Tokens with ≥1 occurrence: {(N2 > 0).sum().item()}")

# %% [markdown]
# ## 8 · Compute X and Y & Save

# %%
X_base, Y = compute_and_save(
    H1, A1, N1, A2, N2,
    tid2idx, idx2tid,
    MODEL_NAME, OUTPUT_DIR, CHUNK_SIZE,
)

# %% [markdown]
# ## 9 · Verification & Statistics

# %%
print("=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"X_base shape: {X_base.shape}  (N, L, D) — in memory")
print(f"Y shape:      {Y.shape}  (N, L, E, 1)")
print(f"X_base dtype: {X_base.dtype}")
print(f"Y dtype:      {Y.dtype}")
print()

# Check for NaNs
print(f"X_base NaN count: {torch.isnan(X_base).sum().item()}")
print(f"Y NaN count:      {torch.isnan(Y).sum().item()}")

# Y distribution
print(f"\nY statistics:")
print(f"  mean:    {Y.mean().item():.6f}")
print(f"  std:     {Y.std().item():.6f}")
print(f"  min:     {Y.min().item():.6f}")
print(f"  max:     {Y.max().item():.6f}")
print(f"  % > 0:   {(Y > 0).float().mean().item() * 100:.1f}%")
print(f"  % < 0:   {(Y < 0).float().mean().item() * 100:.1f}%")
print(f"  % == 0:  {(Y == 0).float().mean().item() * 100:.1f}%")

# X norm per layer
print(f"\nX_base L2 norm per layer (averaged over tokens):")
for l in range(X_base.shape[1]):
    norm = X_base[:, l, :].norm(dim=-1).mean().item()
    print(f"  Layer {l:2d}: {norm:.2f}")

# %%
# Verify chunks are loadable
import glob
chunk_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "chunk_*.pt")))
meta = torch.load(os.path.join(OUTPUT_DIR, "metadata.pt"), map_location="cpu", weights_only=False)
print(f"\nChunk files: {len(chunk_files)}")
print(f"Metadata keys: {list(meta.keys())}")

# Load first chunk and verify X is (chunk, L, E, D)
first_chunk = torch.load(chunk_files[0], map_location="cpu", weights_only=True)
print(f"\nFirst chunk (on disk):")
print(f"  X: {first_chunk['X'].shape}  (chunk, L, E, D) ✅")
print(f"  Y: {first_chunk['Y'].shape}  (chunk, L, E, 1)")

# Verify X has the expert dimension
assert first_chunk['X'].dim() == 4, f"X should be 4D (N,L,E,D), got {first_chunk['X'].dim()}D"
assert first_chunk['X'].shape[2] == experts, f"X dim 2 should be E={experts}, got {first_chunk['X'].shape[2]}"
print()
print("✅ Dataset generation complete! X is (N, L, E, D).")
