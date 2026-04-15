"""
Sample-level dataset generation for steering.

For each SQuAD sample and each layer l, produce one row:
  D  : mean pooled hidden state of x1 at layer l            -> (hidden_dim,)
  y1 : mean pooled softmax(router_logits) of x1 at layer l  -> (n_experts,)
  y2 : mean pooled softmax(router_logits) of x2 at layer l  -> (n_experts,)

x1 = "Document {context} Question {question}"
x2 = "Question {question}"

Final shapes:
  D  : (S * L, hidden_dim)
  y1 : (S * L, n_experts)
  y2 : (S * L, n_experts)

Saved as a single file: dataset.pt
"""

import argparse
import os
import time
from typing import Any

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


def _wandb_log(data: dict, **kwargs) -> None:
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(data, **kwargs)


def load_squad(split: str = "train"):
    return load_dataset("rajpurkar/squad", split=split)


def build_prompts(example: dict[str, Any]) -> tuple[str, str]:
    context = example["context"]
    question = example["question"]
    x1 = f"Document {context} Question {question}"
    x2 = f"Question {question}"
    return x1, x2


def _resolve_input_device(model, fallback: str) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device(fallback)


def _tokenize_prompt(tokenizer, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    kwargs = {
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int) and 0 < model_max_length < 100_000:
        kwargs["truncation"] = True
        kwargs["max_length"] = model_max_length

    inputs = tokenizer(prompt, **kwargs)
    return {k: v.to(device) for k, v in inputs.items()}


def _mean_pool_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
    # Hidden state can be (batch, seq, dim) or (seq, dim)
    if hidden_state.dim() == 3:
        hidden_state = hidden_state[0]
    return hidden_state.float().mean(dim=0)


def _mean_pool_router_softmax(router_logits: torch.Tensor, seq_len: int) -> torch.Tensor:
    # Router logits can be (batch, seq, E), (seq, E), or (batch*seq, E)
    if router_logits.dim() == 3:
        router_logits = router_logits[0]
    elif router_logits.dim() != 2:
        raise ValueError(f"Unexpected router logits shape: {tuple(router_logits.shape)}")

    if router_logits.shape[0] != seq_len:
        router_logits = router_logits[: min(router_logits.shape[0], seq_len)]
    if router_logits.shape[0] == 0:
        raise ValueError("Router logits contain zero tokens after alignment.")

    probs = torch.softmax(router_logits.float(), dim=-1)
    return probs.mean(dim=0)


def process_sample(
    model,
    tokenizer,
    sample: dict[str, Any],
    device: torch.device,
    expected_layers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1, x2 = build_prompts(sample)

    inputs_x1 = _tokenize_prompt(tokenizer, x1, device)
    seq_len_x1 = int(inputs_x1["input_ids"].shape[1])
    with torch.no_grad():
        out_x1 = model(
            **inputs_x1,
            output_hidden_states=True,
            output_router_logits=True,
        )

    inputs_x2 = _tokenize_prompt(tokenizer, x2, device)
    seq_len_x2 = int(inputs_x2["input_ids"].shape[1])
    with torch.no_grad():
        out_x2 = model(
            **inputs_x2,
            output_hidden_states=False,
            output_router_logits=True,
        )

    hidden_states = out_x1.hidden_states
    router_x1 = out_x1.router_logits
    router_x2 = out_x2.router_logits
    available_layers = min(len(hidden_states) - 1, len(router_x1), len(router_x2))
    if expected_layers is not None:
        if available_layers < expected_layers:
            raise RuntimeError(
                f"Expected at least {expected_layers} layers, got {available_layers}."
            )
        n_layers = expected_layers
    else:
        n_layers = available_layers

    d_rows: list[torch.Tensor] = []
    y1_rows: list[torch.Tensor] = []
    y2_rows: list[torch.Tensor] = []

    for layer_idx in range(n_layers):
        d_rows.append(_mean_pool_hidden_state(hidden_states[layer_idx + 1]).cpu())
        y1_rows.append(_mean_pool_router_softmax(router_x1[layer_idx], seq_len_x1).cpu())
        y2_rows.append(_mean_pool_router_softmax(router_x2[layer_idx], seq_len_x2).cpu())

    return torch.stack(d_rows), torch.stack(y1_rows), torch.stack(y2_rows)


def _checkpoint_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "sample_level_ckpt.pt")


def save_checkpoint(
    checkpoint_dir: str,
    sample_idx: int,
    num_samples: int,
    num_layers: int,
    hidden_dim: int,
    n_experts: int,
    D_all: torch.Tensor,
    y1_all: torch.Tensor,
    y2_all: torch.Tensor,
    storage_dtype: torch.dtype,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    rows_filled = sample_idx * num_layers
    d_partial = D_all[:rows_filled].clone()
    y1_partial = y1_all[:rows_filled].clone()
    y2_partial = y2_all[:rows_filled].clone()
    state = {
        "sample_idx": sample_idx,
        "current_sample_idx": sample_idx,
        "num_samples": num_samples,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "n_experts": n_experts,
        "rows_filled": rows_filled,
        "storage_dtype": str(storage_dtype),
        "D": d_partial,
        "y1": y1_partial,
        "y2": y2_partial,
        "D_partial": d_partial,
        "y1_partial": y1_partial,
        "y2_partial": y2_partial,
    }
    tmp = _checkpoint_path(checkpoint_dir) + ".tmp"
    final = _checkpoint_path(checkpoint_dir)
    torch.save(state, tmp)
    os.replace(tmp, final)
    print(f"  checkpoint saved: sample_idx={sample_idx}/{num_samples}")


def load_checkpoint(checkpoint_dir: str | None) -> dict[str, Any] | None:
    if not checkpoint_dir:
        return None
    path = _checkpoint_path(checkpoint_dir)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    canonical = {"sample_idx", "D", "y1", "y2", "num_layers", "hidden_dim", "n_experts"}
    partial = {
        "current_sample_idx",
        "D_partial",
        "y1_partial",
        "y2_partial",
        "num_layers",
        "hidden_dim",
        "n_experts",
    }
    keys = set(ckpt.keys())
    if canonical.issubset(keys):
        return ckpt
    if partial.issubset(keys):
        ckpt["sample_idx"] = int(ckpt["current_sample_idx"])
        ckpt["D"] = ckpt["D_partial"]
        ckpt["y1"] = ckpt["y1_partial"]
        ckpt["y2"] = ckpt["y2_partial"]
        return ckpt
    print("Checkpoint missing required keys. Ignoring checkpoint.")
    return None


def generate(
    model_name: str,
    output_dir: str,
    max_examples: int | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 100,
    split: str = "train",
    device: str = "cuda",
    storage_dtype: str = "float16",
):
    hf_token = os.environ.get("HF_TOKEN")
    t_start = time.time()
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    if storage_dtype not in dtype_map:
        raise ValueError(f"Unsupported storage dtype: {storage_dtype}")
    storage_dtype_t = dtype_map[storage_dtype]

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    if _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
        run_suffix = f"{max_examples}ex" if max_examples else "full"
        wandb.init(
            entity="VLAvengers",
            project="tokenaware-steering-moe",
            name=f"sample-level-{model_name.split('/')[-1]}-{run_suffix}",
            config={
                "model_name": model_name,
                "max_examples": max_examples,
                "checkpoint_interval": checkpoint_interval,
                "split": split,
                "device": device,
                "storage_dtype": storage_dtype,
                "output_dir": output_dir,
                "checkpoint_dir": checkpoint_dir,
            },
            tags=["dataset-generation", "sample-level", "squad"],
        )
        print("W&B run initialized")
    else:
        if not _WANDB_AVAILABLE:
            print("wandb not installed: logging disabled")
        else:
            print("WANDB_API_KEY not set: logging disabled")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    print(f"Loading SQuAD split={split}...")
    dataset = load_squad(split=split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    num_samples = len(dataset)
    if num_samples == 0:
        raise RuntimeError("No samples available after applying split/max_examples.")
    print(f"  Samples: {num_samples}")

    print(f"Loading model: {model_name}")
    model_kwargs = {
        "token": hf_token,
        "trust_remote_code": True,
    }
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    input_device = _resolve_input_device(model, device)
    print(f"  Model input device: {input_device}")

    ckpt = load_checkpoint(checkpoint_dir)

    d_all: torch.Tensor
    y1_all: torch.Tensor
    y2_all: torch.Tensor

    if ckpt is not None and ckpt.get("num_samples") not in (None, num_samples):
        print(
            "Checkpoint sample count does not match current run. "
            "Ignoring checkpoint and starting from scratch."
        )
        ckpt = None

    if ckpt is None:
        print("Bootstrapping shapes from first sample...")
        d0, y10, y20 = process_sample(
            model=model,
            tokenizer=tokenizer,
            sample=dataset[0],
            device=input_device,
            expected_layers=getattr(model.config, "num_hidden_layers", None),
        )
        num_layers = int(d0.shape[0])
        hidden_dim = int(d0.shape[1])
        n_experts = int(y10.shape[1])
        total_rows = num_samples * num_layers

        d_all = torch.zeros(total_rows, hidden_dim, dtype=storage_dtype_t)
        y1_all = torch.zeros(total_rows, n_experts, dtype=storage_dtype_t)
        y2_all = torch.zeros(total_rows, n_experts, dtype=storage_dtype_t)

        d_all[:num_layers] = d0.to(storage_dtype_t)
        y1_all[:num_layers] = y10.to(storage_dtype_t)
        y2_all[:num_layers] = y20.to(storage_dtype_t)
        start_sample = 1
        print(
            f"  Shapes resolved: L={num_layers}, D={hidden_dim}, E={n_experts}, "
            f"rows={total_rows}"
        )
    else:
        num_layers = int(ckpt["num_layers"])
        hidden_dim = int(ckpt["hidden_dim"])
        n_experts = int(ckpt["n_experts"])
        total_rows = num_samples * num_layers
        start_sample = int(ckpt["sample_idx"])
        rows_filled = int(ckpt.get("rows_filled", ckpt["D"].shape[0]))
        if rows_filled != start_sample * num_layers:
            start_sample = rows_filled // num_layers
            rows_filled = start_sample * num_layers
        start_sample = min(start_sample, num_samples)
        rows_filled = min(rows_filled, total_rows)

        d_all = torch.zeros(total_rows, hidden_dim, dtype=storage_dtype_t)
        y1_all = torch.zeros(total_rows, n_experts, dtype=storage_dtype_t)
        y2_all = torch.zeros(total_rows, n_experts, dtype=storage_dtype_t)

        d_all[:rows_filled] = ckpt["D"][:rows_filled].to(storage_dtype_t)
        y1_all[:rows_filled] = ckpt["y1"][:rows_filled].to(storage_dtype_t)
        y2_all[:rows_filled] = ckpt["y2"][:rows_filled].to(storage_dtype_t)
        print(f"Resuming from sample {start_sample}/{num_samples}")

    bytes_per_element = torch.finfo(storage_dtype_t).bits // 8
    estimated_gb = total_rows * (hidden_dim + 2 * n_experts) * bytes_per_element / 1e9
    print(
        f"Estimated dataset size ({storage_dtype}): {estimated_gb:.2f} GB "
        f"for rows={total_rows}"
    )
    _wandb_log(
        {
            "dataset/rows": total_rows,
            "dataset/layers": num_layers,
            "dataset/hidden_dim": hidden_dim,
            "dataset/experts": n_experts,
            "dataset/estimated_size_gb": estimated_gb,
        }
    )

    pbar = tqdm(range(start_sample, num_samples), desc="Generating rows (sample x layer)")
    for sample_idx in pbar:
        d_rows, y1_rows, y2_rows = process_sample(
            model=model,
            tokenizer=tokenizer,
            sample=dataset[sample_idx],
            device=input_device,
            expected_layers=num_layers,
        )

        row_start = sample_idx * num_layers
        row_end = row_start + num_layers
        d_all[row_start:row_end] = d_rows.to(storage_dtype_t)
        y1_all[row_start:row_end] = y1_rows.to(storage_dtype_t)
        y2_all[row_start:row_end] = y2_rows.to(storage_dtype_t)

        if checkpoint_dir and (sample_idx + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                sample_idx=sample_idx + 1,
                num_samples=num_samples,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                n_experts=n_experts,
                D_all=d_all,
                y1_all=y1_all,
                y2_all=y2_all,
                storage_dtype=storage_dtype_t,
            )

        if (sample_idx + 1) % 25 == 0:
            pct = (sample_idx + 1) / num_samples * 100
            _wandb_log({"progress/sample_idx": sample_idx + 1, "progress/pct": pct})

    if checkpoint_dir:
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            sample_idx=num_samples,
            num_samples=num_samples,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            D_all=d_all,
            y1_all=y1_all,
            y2_all=y2_all,
            storage_dtype=storage_dtype_t,
        )

    metadata = {
        "model_name": model_name,
        "split": split,
        "num_samples": num_samples,
        "rows": total_rows,
        "layers": num_layers,
        "hidden_dim": hidden_dim,
        "experts": n_experts,
        "storage_dtype": storage_dtype,
        "x1_template": "Document {context} Question {question}",
        "x2_template": "Question {question}",
        "y_definition": "mean_token_softmax_router_logits",
        "pooling": "mean_over_all_tokens",
    }

    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, "dataset.pt")
    tmp_path = dataset_path + ".tmp"
    torch.save({"D": d_all, "y1": y1_all, "y2": y2_all, "metadata": metadata}, tmp_path)
    os.replace(tmp_path, dataset_path)

    elapsed_min = (time.time() - t_start) / 60
    print(f"Saved dataset to: {dataset_path}")
    print(f"D shape:  {tuple(d_all.shape)}")
    print(f"y1 shape: {tuple(y1_all.shape)}")
    print(f"y2 shape: {tuple(y2_all.shape)}")
    print(f"Total time: {elapsed_min:.1f} minutes")

    _wandb_log(
        {
            "dataset/path": dataset_path,
            "dataset/D_nan_count": int(torch.isnan(d_all.float()).sum().item()),
            "dataset/y1_nan_count": int(torch.isnan(y1_all.float()).sum().item()),
            "dataset/y2_nan_count": int(torch.isnan(y2_all.float()).sum().item()),
            "timing/total_minutes": elapsed_min,
            "status": "completed",
        }
    )
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

    return d_all, y1_all, y2_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sample-level dataset: D (hidden), y1/y2 (expert scores)"
    )
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0125-Instruct")
    parser.add_argument("--output-dir", type=str, default="dataset_3d_output")
    parser.add_argument("--checkpoint-dir", type=str, default="dataset_3d_ckpt")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
    )
    args = parser.parse_args()

    generate(
        model_name=args.model,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        split=args.split,
        device=args.device,
        storage_dtype=args.storage_dtype,
    )
