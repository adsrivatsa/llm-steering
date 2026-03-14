"""
Checkpointing for SteerMoE delta computation.

- Collection checkpoints: save every N prompts; only the latest checkpoint per
  (dataset, model, pass) is kept (overwritten each time) so computation can resume.
- Final raw deltas: save unaggregated counts (expert_counts_x1/x2, token_counts_x1/x2)
  before aggregating to risk_diff.
"""

from __future__ import annotations

import os
from typing import Any

import torch


CHECKPOINT_INTERVAL = 10

COLLECTION_CHECKPOINT_PREFIX = "collection"
FINAL_RAW_FILENAME = "raw.pt"


def _model_name_safe(model_name: str) -> str:
    return model_name.replace("/", "_")


def _collection_checkpoint_path(
    checkpoint_dir: str,
    dataset_name: str,
    model_name: str,
    pass_name: str,
) -> str:
    """Single path per (dataset, model, pass); overwritten on each save."""
    model_safe = _model_name_safe(model_name)
    return os.path.join(
        checkpoint_dir,
        f"{COLLECTION_CHECKPOINT_PREFIX}_{dataset_name}_{model_safe}_{pass_name}.pt",
    )


def save_collection_checkpoint(
    checkpoint_dir: str,
    pass_name: str,
    step: int,
    expert_counts_by_token: torch.Tensor,
    token_counts: torch.Tensor,
    *,
    num_experts: int,
    num_token_indices: int,
    dataset_name: str = "",
    model_name: str = "",
) -> str:
    """
    Save collection state (overwrites existing checkpoint for this dataset/model/pass).
    Returns the path written.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = _collection_checkpoint_path(checkpoint_dir, dataset_name, model_name, pass_name)
    state = {
        "step": step,
        "expert_counts_by_token": expert_counts_by_token.cpu(),
        "token_counts": token_counts.cpu(),
        "num_experts": num_experts,
        "num_token_indices": num_token_indices,
        "dataset_name": dataset_name,
        "model_name": model_name,
    }
    # Write to a temp file then rename so we never leave a corrupted checkpoint.
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_collection_checkpoint(
    checkpoint_dir: str,
    pass_name: str,
    dataset_name: str,
    model_name: str,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]] | None:
    """
    Load the collection checkpoint for this (dataset, model, pass) if it exists.
    Returns (expert_counts_by_token, token_counts, step, metadata) or None.
    """
    path = _collection_checkpoint_path(checkpoint_dir, dataset_name, model_name, pass_name)
    if not os.path.isfile(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Corrupted checkpoint (e.g. interrupted during save); remove and start fresh.
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    expert_counts = state["expert_counts_by_token"]
    token_counts = state["token_counts"]
    if device is not None:
        expert_counts = expert_counts.to(device)
        token_counts = token_counts.to(device)
    metadata = {
        k: state[k]
        for k in ("num_experts", "num_token_indices", "dataset_name", "model_name")
        if k in state
    }
    return (
        expert_counts,
        token_counts,
        int(state["step"]),
        metadata,
    )


def save_final_raw(
    raw_dir: str,
    expert_counts_x1: torch.Tensor,
    expert_counts_x2: torch.Tensor,
    token_counts_x1: torch.Tensor,
    token_counts_x2: torch.Tensor,
    *,
    dataset_name: str,
    model_name: str,
) -> str:
    """
    Save final raw (unaggregated) counts before they are summed over tokens.
    """
    os.makedirs(raw_dir, exist_ok=True)
    model_safe = _model_name_safe(model_name)
    subdir = os.path.join(raw_dir, dataset_name)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{model_safe}_{FINAL_RAW_FILENAME}")
    torch.save(
        {
            "expert_counts_x1": expert_counts_x1.cpu(),
            "expert_counts_x2": expert_counts_x2.cpu(),
            "token_counts_x1": token_counts_x1.cpu(),
            "token_counts_x2": token_counts_x2.cpu(),
            "dataset_name": dataset_name,
            "model_name": model_name,
        },
        path,
    )
    return path


def save_final_activations(
    eps_dir: str,
    delta_dir: str,
    epsilon: torch.Tensor,
    risk_diff: torch.Tensor,
    *,
    dataset_name: str,
    model_name: str,
) -> tuple[str, str]:
    """
    Save aggregated epsilon and delta (risk_diff) matrices.
    Returns (eps_path, delta_path).
    """
    model_safe = _model_name_safe(model_name)
    for d in (eps_dir, delta_dir):
        os.makedirs(d, exist_ok=True)
    eps_sub = os.path.join(eps_dir, dataset_name)
    delta_sub = os.path.join(delta_dir, dataset_name)
    os.makedirs(eps_sub, exist_ok=True)
    os.makedirs(delta_sub, exist_ok=True)
    eps_path = os.path.join(eps_sub, f"{model_safe}.pt")
    delta_path = os.path.join(delta_sub, f"{model_safe}.pt")
    torch.save(epsilon.cpu(), eps_path)
    torch.save(risk_diff.cpu(), delta_path)
    return eps_path, delta_path
