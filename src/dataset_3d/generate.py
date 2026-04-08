"""
Dataset generation for learning the 3D delta Δ(E, L, D).

Produces:
  X: (N, L, D)     — averaged hidden states per token per layer
  Y: (N, L, E, 1)  — expert activation risk difference (classification target)

N = number of selected tokens (all by default, or middle N by frequency)
L = number of transformer layers
E = number of experts per layer
D = hidden state dimension

At training time, X is broadcast to E experts: X[:, :, None, :].expand(N, L, E, D)

Usage:
  python -m src.dataset_3d.generate \\
      --model allenai/OLMoE-1B-7B-0125-Instruct \\
      --output-dir dataset_3d_output
"""

import argparse
import os
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            start_idx = int(ckpt["step"])
            A = ckpt["A"]
            N_counts = ckpt["N"]
            if collect_hidden_states and "H_sum" in ckpt:
                H_sum = ckpt["H_sum"]
            print(f"[{pass_name}] Resuming from step {start_idx}")

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

        # Periodic checkpoint
        if checkpoint_dir and (idx + 1) % checkpoint_interval == 0:
            _save_ckpt(checkpoint_dir, pass_name, idx + 1, A, N_counts, H_sum)

    # Final checkpoint
    if checkpoint_dir:
        _save_ckpt(checkpoint_dir, pass_name, len(prompts), A, N_counts, H_sum)

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

    X stored as (chunk, L, D) — expand to E at training time.
    Y stored as (chunk, L, E, 1).
    """
    # Filter to tokens present in both passes
    valid = (N1 > 0) & (N2 > 0)
    valid_indices = valid.nonzero(as_tuple=True)[0]
    print(f"Valid tokens (appear in both passes): {len(valid_indices)}")

    N1_safe = N1[valid_indices].float().clamp(min=1)
    N2_safe = N2[valid_indices].float().clamp(min=1)

    # X: averaged hidden states from x1 pass  — shape (N_valid, L, D)
    X = H1_sum[valid_indices] / N1_safe.unsqueeze(1).unsqueeze(2)

    # Y: risk difference — shape (N_valid, L, E, 1)
    #   p1[l, e, n] = A1[l, e, n] / N1[n]
    #   p2[l, e, n] = A2[l, e, n] / N2[n]
    p1 = A1[:, :, valid_indices] / N1_safe.unsqueeze(0).unsqueeze(0)
    p2 = A2[:, :, valid_indices] / N2_safe.unsqueeze(0).unsqueeze(0)
    Y = (p1 - p2).permute(2, 0, 1).unsqueeze(-1)  # (N_valid, L, E, 1)

    # Save chunks
    os.makedirs(output_dir, exist_ok=True)
    N_total = X.shape[0]
    n_chunks = 0

    for i in range(0, N_total, chunk_size):
        end = min(i + chunk_size, N_total)
        torch.save(
            {"X": X[i:end], "Y": Y[i:end]},
            os.path.join(output_dir, f"chunk_{n_chunks:04d}.pt"),
        )
        n_chunks += 1

    # Save metadata
    layers, experts = A1.shape[0], A1.shape[1]
    hidden_dim = X.shape[2]
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
    print(f"  X shape per chunk: (≤{chunk_size}, {layers}, {hidden_dim})")
    print(f"  Y shape per chunk: (≤{chunk_size}, {layers}, {experts}, 1)")

    # RAM estimate
    x_bytes = N_total * layers * hidden_dim * 4
    y_bytes = N_total * layers * experts * 4
    print(f"  Total X size: {x_bytes / 1e9:.2f} GB")
    print(f"  Total Y size: {y_bytes / 1e9:.2f} GB")

    return X, Y


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

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Hidden dim: {model.config.hidden_size}")

    prompts_x1, prompts_x2 = build_prompts(dataset)

    # Pass 1: x1 (doc + question) — hidden states + router activations
    print("\n=== Pass 1: x1 (document + question) ===")
    H1, A1, N1 = collect_pass_data(
        model, tokenizer, prompts_x1, tid2idx,
        layers, experts, top_k, device,
        pass_name="x1", collect_hidden_states=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )

    # Pass 2: x2 (question only) — router activations only
    print("\n=== Pass 2: x2 (question only) ===")
    _, A2, N2 = collect_pass_data(
        model, tokenizer, prompts_x2, tid2idx,
        layers, experts, top_k, device,
        pass_name="x2", collect_hidden_states=False,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )

    # Compute and save
    print("\n=== Computing X and Y ===")
    X, Y = compute_and_save(
        H1, A1, N1, A2, N2,
        tid2idx, idx2tid,
        model_name, output_dir, chunk_size,
    )

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
