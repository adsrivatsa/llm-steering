"""
Raw activation checkpoint utilities for SteerMoE.

This module intentionally only supports saving/loading raw per-pass activations:
- expert_counts_by_token
- token_counts
"""

from __future__ import annotations

import os
from typing import Any

import torch

DIR = "./checkpoints"
INTERVAL = 50
RAW_FILE_SUFFIX = ".pt"


def _model_name_safe(model_name: str) -> str:
    return model_name.replace("/", "_")


def _raw_path(
    raw_dir: str,
    dataset_name: str,
    model_name: str,
    pass_name: str,
) -> str:
    """Single path per (dataset, model, pass); overwritten on each save."""
    model_safe = _model_name_safe(model_name)
    return os.path.join(
        raw_dir,
        dataset_name,
        f"{model_safe}_{pass_name}{RAW_FILE_SUFFIX}",
    )


def save(
    raw_dir: str,
    pass_name: str,
    expert_counts_by_token: torch.Tensor,
    token_counts: torch.Tensor,
    *,
    dataset_name: str,
    model_name: str,
    step: int | None = None,
) -> str:
    """
    Save raw per-pass state for this (dataset, model, pass).
    Returns the path written.
    """
    path = _raw_path(raw_dir, dataset_name, model_name, pass_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "expert_counts_by_token": expert_counts_by_token.cpu(),
        "token_counts": token_counts.cpu(),
        "step": int(step) if step is not None else None,
    }
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def load(
    raw_dir: str,
    pass_name: str,
    dataset_name: str,
    model_name: str,
) -> dict[str, Any] | None:
    """
    Load the raw per-pass checkpoint for this (dataset, model, pass).
    Returns the state dict with an additional "path" key, or None if unavailable.
    """
    path = _raw_path(raw_dir, dataset_name, model_name, pass_name)
    if not os.path.isfile(path):
        return None

    state = torch.load(path, map_location="cpu", weights_only=True)
    state["path"] = path
    return state
