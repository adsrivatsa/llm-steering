"""
Dataset generation for learning the 3D delta Δ(E, L, D).

Produces:
  X: (N, L, E, D)  — averaged hidden states per token per layer, broadcast across experts
  Y: (N, L, E, 1)  — expert activation risk difference (classification target)

N = number of selected tokens (all by default, or middle N by frequency)
L = number of transformer layers
E = number of experts per layer
D = hidden state dimension

Usage:
  python -m src.dataset_3d.generate \\
      --model allenai/OLMoE-1B-7B-0125-Instruct \\
      --output-dir dataset_3d_output
"""

import argparse
import os
import time
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Optional W&B ──────────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _wandb_log(data: dict, **kwargs):
    """Log to wandb if available and initialized."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(data, **kwargs)


# ── Model Configurations ─────────────────────────────────────────────────────
# {model_name: (layers, top_k, experts)}
MOE_EXPERT_CONFIG = {
    "allenai/OLMoE-1B-7B-0125-Instruct": (16, 8, 64),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (32, 2, 8),
    "Qwen/Qwen3-30B-A3B": (48, 8, 128),
    "microsoft/Phi-3.5-MoE-instruct": (32, 2, 16),
}


def load_squad(split: str = "train"):
    """Load SQuAD v1.1 dataset."""
    return load_dataset("rajpurkar/squad", split=split)


# ── Token Selection ───────────────────────────────────────────────────────────

def select_middle_tokens(dataset, tokenizer, n_tokens: int | None = None):
    """Select tokens with middle-range frequency from SQuAD questions.

    Returns:
        selected_set: set of selected token IDs
        token_id_to_index: token ID → contiguous index
        index_to_token_id: contiguous index → token ID
        freq_info: dict with frequency statistics
    """
    token_freq: Counter = Counter()

    for example in tqdm(dataset, desc="Counting token frequencies"):
        question = example["question"]
        for variant in [question, f" {question}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            token_freq.update(ids)

    sorted_by_freq = sorted(token_freq.items(), key=lambda x: x[1])
    total = len(sorted_by_freq)

    if n_tokens is None or total <= n_tokens:
        selected_list = [t[0] for t in sorted_by_freq]
        freq_range = (sorted_by_freq[0][1], sorted_by_freq[-1][1])
    else:
        start = (total - n_tokens) // 2
        end = start + n_tokens
        selected_list = [t[0] for t in sorted_by_freq[start:end]]
        freq_range = (sorted_by_freq[start][1], sorted_by_freq[end - 1][1])

    selected_sorted = sorted(selected_list)
    token_id_to_index = {tid: idx for idx, tid in enumerate(selected_sorted)}
    index_to_token_id = {idx: tid for tid, idx in token_id_to_index.items()}

    freq_info = {
        "total_unique_tokens": total,
        "selected_tokens": len(selected_sorted),
        "min_freq": freq_range[0],
        "max_freq": freq_range[1],
    }
    return set(selected_sorted), token_id_to_index, index_to_token_id, freq_info


# ── Question Token Range ─────────────────────────────────────────────────────

def find_question_token_range(tokenizer, content: str, question: str):
    """Return (start, end) token indices of the question within chat-formatted input.

    Uses token-subsequence matching. Adapted from src/activation/main.py.
    """
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )
    full_ids: list[int] = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    candidates: list[list[int]] = []
    for q in [question, f" {question}"]:
        ids = tokenizer(q, add_special_tokens=False)["input_ids"]
        if ids and ids not in candidates:
            candidates.append(ids)

    best_start, best_end = None, None
    for target_ids in candidates:
        t_len = len(target_ids)
        if t_len == 0 or t_len > len(full_ids):
            continue
        for i in range(len(full_ids) - t_len + 1):
            if full_ids[i : i + t_len] == target_ids:
                best_start, best_end = i, i + t_len

    if best_start is None:
        return 0, 0, []
    return best_start, best_end, full_ids


# ── Data Collection ───────────────────────────────────────────────────────────

def collect_pass_data(
    model,
    tokenizer,
    prompts: list[tuple[str, str]],
    token_id_to_index: dict[int, int],
    layers: int,
    experts: int,
    top_k: int,
    device,
    pass_name: str = "x1",
    collect_hidden_states: bool = True,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 500,
):
    """Run one pass through the model, collecting data for selected tokens.

    Args:
        prompts: list of (prompt_text, question_text) tuples
        collect_hidden_states: only needed for x1

    Returns:
        H_sum: (N, L, D) accumulated hidden states (None if not collected)
        A: (L, E, N) expert activation counts
        N_counts: (N,) token occurrence counts
    """
    n_tokens = len(token_id_to_index)
    hidden_dim = model.config.hidden_size

    H_sum = torch.zeros(n_tokens, layers, hidden_dim) if collect_hidden_states else None
    A = torch.zeros(layers, experts, n_tokens)
    N_counts = torch.zeros(n_tokens, dtype=torch.long)

    start_idx = 0
    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, f"{pass_name}_ckpt.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if ckpt["A"].shape[-1] == n_tokens:
                start_idx = int(ckpt["step"])
                A = ckpt["A"]
                N_counts = ckpt["N"]
                if collect_hidden_states and "H_sum" in ckpt:
                    H_sum = ckpt["H_sum"]
                print(f"[{pass_name}] Resuming from step {start_idx}")
            else:
                print(f"[{pass_name}] Checkpoint n_tokens ({ckpt['A'].shape[-1]}) mismatch with current ({n_tokens}). Starting from scratch.")

    for idx in tqdm(
        range(start_idx, len(prompts)),
        desc=f"Pass {pass_name}",
        initial=start_idx,
        total=len(prompts),
    ):
        prompt_text, question_text = prompts[idx]

        q_start, q_end, full_ids = find_question_token_range(
            tokenizer, prompt_text, question_text
        )
        if q_start == 0 and q_end == 0:
            continue

        # Tokenise chat-formatted text
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=collect_hidden_states,
                output_router_logits=True,
            )

        router_logits = outputs.router_logits          # tuple, one per layer
        hidden_states = outputs.hidden_states if collect_hidden_states else None

        # Process question tokens
        q_token_ids = full_ids[q_start:q_end]
        for pos_offset, tid in enumerate(q_token_ids):
            if tid not in token_id_to_index:
                continue
            t_idx = token_id_to_index[tid]
            abs_pos = q_start + pos_offset

            for layer in range(layers):
                # Router logits may be (seq_len, E) or (batch*seq_len, E)
                logits = router_logits[layer]
                if logits.dim() == 3:
                    logits = logits[0]           # remove batch dim
                expert_logit = logits[abs_pos]   # (E,)
                topk_idx = torch.topk(expert_logit, top_k).indices.cpu()
                A[layer, topk_idx, t_idx] += 1

                if collect_hidden_states:
                    h = hidden_states[layer + 1][0, abs_pos].cpu().float()
                    H_sum[t_idx, layer] += h

            N_counts[t_idx] += 1

        # Periodic checkpoint + wandb logging
        if checkpoint_dir and (idx + 1) % checkpoint_interval == 0:
            _save_ckpt(checkpoint_dir, pass_name, idx + 1, A, N_counts, H_sum)
            tokens_seen = (N_counts > 0).sum().item()
            _wandb_log({
                f"{pass_name}/step": idx + 1,
                f"{pass_name}/progress_pct": (idx + 1) / len(prompts) * 100,
                f"{pass_name}/unique_tokens_seen": tokens_seen,
                f"{pass_name}/total_token_occurrences": N_counts.sum().item(),
            })

    # Final checkpoint
    if checkpoint_dir:
        _save_ckpt(checkpoint_dir, pass_name, len(prompts), A, N_counts, H_sum)

    # Log final pass stats
    tokens_seen = (N_counts > 0).sum().item()
    _wandb_log({
        f"{pass_name}/final_unique_tokens": tokens_seen,
        f"{pass_name}/final_total_occurrences": N_counts.sum().item(),
        f"{pass_name}/completed": True,
    })

    return H_sum, A, N_counts


def _save_ckpt(directory, pass_name, step, A, N, H_sum=None):
    os.makedirs(directory, exist_ok=True)
    state = {"step": step, "A": A, "N": N}
    if H_sum is not None:
        state["H_sum"] = H_sum
    tmp = os.path.join(directory, f"{pass_name}_ckpt.pt.tmp")
    final = os.path.join(directory, f"{pass_name}_ckpt.pt")
    torch.save(state, tmp)
    os.replace(tmp, final)
    print(f"  checkpoint saved: step={step}")


# ── Dataset Computation & Saving ──────────────────────────────────────────────

def compute_and_save(
    H1_sum: torch.Tensor,
    A1: torch.Tensor,
    N1: torch.Tensor,
    A2: torch.Tensor,
    N2: torch.Tensor,
    token_id_to_index: dict,
    index_to_token_id: dict,
    model_name: str,
    output_dir: str,
    chunk_size: int = 2000,
):
    """Compute X and Y from accumulated data and save in chunks.

    X stored as (chunk, L, E, D) — hidden states broadcast across all experts.
    Y stored as (chunk, L, E, 1).

    Returns X_base (N, L, D) and Y (N, L, E, 1) in memory.
    The full (N, L, E, D) expansion is done chunk-by-chunk to avoid RAM blow-up.
    """
    # Filter to tokens present in both passes
    valid = (N1 > 0) & (N2 > 0)
    valid_indices = valid.nonzero(as_tuple=True)[0]
    print(f"Valid tokens (appear in both passes): {len(valid_indices)}")

    N1_safe = N1[valid_indices].float().clamp(min=1)
    N2_safe = N2[valid_indices].float().clamp(min=1)

    # X_base: averaged hidden states from x1 pass — shape (N_valid, L, D)
    X_base = H1_sum[valid_indices] / N1_safe.unsqueeze(1).unsqueeze(2)

    # Y: risk difference — shape (N_valid, L, E, 1)
    #   p1[l, e, n] = A1[l, e, n] / N1[n]
    #   p2[l, e, n] = A2[l, e, n] / N2[n]
    p1 = A1[:, :, valid_indices] / N1_safe.unsqueeze(0).unsqueeze(0)
    p2 = A2[:, :, valid_indices] / N2_safe.unsqueeze(0).unsqueeze(0)
    Y = (p1 - p2).permute(2, 0, 1).unsqueeze(-1)  # (N_valid, L, E, 1)

    # Save chunks — X expanded to (chunk, L, E, D) per chunk
    os.makedirs(output_dir, exist_ok=True)
    layers, experts = A1.shape[0], A1.shape[1]
    hidden_dim = X_base.shape[2]
    N_total = X_base.shape[0]
    n_chunks = 0

    for i in range(0, N_total, chunk_size):
        end = min(i + chunk_size, N_total)
        # Expand X_base (chunk, L, D) → (chunk, L, E, D)
        X_chunk = X_base[i:end].unsqueeze(2).expand(-1, -1, experts, -1).clone()
        torch.save(
            {"X": X_chunk, "Y": Y[i:end]},
            os.path.join(output_dir, f"chunk_{n_chunks:04d}.pt"),
        )
        n_chunks += 1

    # Save metadata
    meta = {
        "model_name": model_name,
        "layers": layers,
        "experts": experts,
        "hidden_dim": hidden_dim,
        "n_total": N_total,
        "n_chunks": n_chunks,
        "chunk_size": chunk_size,
        "valid_indices": valid_indices,
        "token_id_to_index": token_id_to_index,
        "index_to_token_id": index_to_token_id,
        "N1": N1,
        "N2": N2,
    }
    torch.save(meta, os.path.join(output_dir, "metadata.pt"))
    print(f"Saved {n_chunks} chunks to {output_dir}")
    print(f"  X shape per chunk: (≤{chunk_size}, {layers}, {experts}, {hidden_dim})")
    print(f"  Y shape per chunk: (≤{chunk_size}, {layers}, {experts}, 1)")

    # Disk size estimate
    x_bytes = N_total * layers * experts * hidden_dim * 4
    y_bytes = N_total * layers * experts * 4
    print(f"  Total X size: {x_bytes / 1e9:.2f} GB")
    print(f"  Total Y size: {y_bytes / 1e9:.2f} GB")
    print(f"  Total dataset: {(x_bytes + y_bytes) / 1e9:.2f} GB")

    # Log final dataset stats to wandb
    _wandb_log({
        "dataset/n_valid_tokens": N_total,
        "dataset/n_chunks": n_chunks,
        "dataset/X_shape": f"({N_total}, {layers}, {experts}, {hidden_dim})",
        "dataset/Y_shape": f"({N_total}, {layers}, {experts}, 1)",
        "dataset/X_size_gb": x_bytes / 1e9,
        "dataset/Y_size_gb": y_bytes / 1e9,
        "dataset/total_size_gb": (x_bytes + y_bytes) / 1e9,
        "dataset/X_nan_count": int(torch.isnan(X_base).sum().item()),
        "dataset/Y_nan_count": int(torch.isnan(Y).sum().item()),
        "dataset/Y_mean": Y.mean().item(),
        "dataset/Y_std": Y.std().item(),
        "dataset/Y_min": Y.min().item(),
        "dataset/Y_max": Y.max().item(),
        "dataset/Y_pct_positive": (Y > 0).float().mean().item() * 100,
        "dataset/Y_pct_negative": (Y < 0).float().mean().item() * 100,
        "dataset/Y_pct_zero": (Y == 0).float().mean().item() * 100,
    })

    return X_base, Y


# ── Main Entry ────────────────────────────────────────────────────────────────

def build_prompts(dataset):
    """Build (x1, x2) prompt pairs from SQuAD."""
    prompts_x1, prompts_x2 = [], []
    for ex in dataset:
        doc = f"Document {ex['context']}"
        q = f"Question {ex['question']}"
        prompts_x1.append((f"{doc} {q}", q))
        prompts_x2.append((q, q))
    return prompts_x1, prompts_x2


def generate(
    model_name: str,
    output_dir: str,
    n_tokens: int | None = None,
    chunk_size: int = 2000,
    max_examples: int | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 500,
    device: str = "cuda",
):
    """End-to-end dataset generation."""
    hf_token = os.environ.get("HF_TOKEN")
    t_start = time.time()

    # ── Initialize W&B ────────────────────────────────────────────────────────
    if _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            entity="VLAvengers",
            project="tokenaware-steering-moe",
            config={
                "model_name": model_name,
                "n_tokens": n_tokens,
                "chunk_size": chunk_size,
                "max_examples": max_examples,
                "checkpoint_interval": checkpoint_interval,
                "device": device,
                "output_dir": output_dir,
            },
            tags=["dataset-generation", "3d-delta"],
        )
        print("✅ W&B run initialized")
    else:
        if not _WANDB_AVAILABLE:
            print("⚠️  wandb not installed — logging disabled")
        else:
            print("⚠️  WANDB_API_KEY not set — logging disabled")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    print("Loading SQuAD...")
    dataset = load_squad()
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    print(f"  Examples: {len(dataset)}")

    print("Selecting middle-frequency tokens...")
    selected_set, tid2idx, idx2tid, freq_info = select_middle_tokens(
        dataset, tokenizer, n_tokens
    )
    print(f"  {freq_info}")

    layers, top_k, experts = MOE_EXPERT_CONFIG[model_name]
    print(f"Model config: L={layers}, K={top_k}, E={experts}")

    # Log token selection info
    _wandb_log({
        "tokens/total_unique": freq_info["total_unique_tokens"],
        "tokens/selected": freq_info["selected_tokens"],
        "tokens/min_freq": freq_info["min_freq"],
        "tokens/max_freq": freq_info["max_freq"],
        "model/layers": layers,
        "model/top_k": top_k,
        "model/experts": experts,
    })

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()
    hidden_dim = model.config.hidden_size
    print(f"  Hidden dim: {hidden_dim}")
    _wandb_log({"model/hidden_dim": hidden_dim})

    prompts_x1, prompts_x2 = build_prompts(dataset)

    # Pass 1: x1 (doc + question) — hidden states + router activations
    print("\n=== Pass 1: x1 (document + question) ===")
    t_pass1 = time.time()
    H1, A1, N1 = collect_pass_data(
        model, tokenizer, prompts_x1, tid2idx,
        layers, experts, top_k, device,
        pass_name="x1", collect_hidden_states=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )
    t_pass1_done = time.time()
    _wandb_log({"timing/pass_x1_minutes": (t_pass1_done - t_pass1) / 60})

    # Pass 2: x2 (question only) — router activations only
    print("\n=== Pass 2: x2 (question only) ===")
    t_pass2 = time.time()
    _, A2, N2 = collect_pass_data(
        model, tokenizer, prompts_x2, tid2idx,
        layers, experts, top_k, device,
        pass_name="x2", collect_hidden_states=False,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )
    t_pass2_done = time.time()
    _wandb_log({"timing/pass_x2_minutes": (t_pass2_done - t_pass2) / 60})

    # Compute and save
    print("\n=== Computing X and Y ===")
    X, Y = compute_and_save(
        H1, A1, N1, A2, N2,
        tid2idx, idx2tid,
        model_name, output_dir, chunk_size,
    )

    # Final timing
    t_total = time.time() - t_start
    _wandb_log({
        "timing/total_minutes": t_total / 60,
        "status": "completed",
    })
    print(f"\nTotal time: {t_total / 60:.1f} minutes")

    # Finish W&B run
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("✅ W&B run finished")

    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D delta training dataset")
    parser.add_argument(
        "--model", type=str,
        default="allenai/OLMoE-1B-7B-0125-Instruct",
    )
    parser.add_argument("--output-dir", type=str, default="dataset_3d_output")
    parser.add_argument("--checkpoint-dir", type=str, default="dataset_3d_ckpt")
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    generate(
        model_name=args.model,
        output_dir=args.output_dir,
        n_tokens=args.n_tokens,
        chunk_size=args.chunk_size,
        max_examples=args.max_examples,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
