import os
from typing import Any

import torch

INTERVAL = 50
RAW_FILE_SUFFIX = ".pt"


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def raw_path(
    raw_dir: str,
    dataset_name: str,
    model_name: str,
    pass_name: str,
) -> str:
    """Single path per (dataset, model, pass); overwritten on each save."""
    model_safe = safe_model_name(model_name)
    return os.path.join(
        raw_dir,
        dataset_name,
        f"{model_safe}_{pass_name}{RAW_FILE_SUFFIX}",
    )


def save(
    raw_dir: str,
    pass_name: str,
    *,
    state: dict[str, Any],
    dataset_name: str,
    model_name: str,
    step: int | None = None,
) -> str:
    path = raw_path(raw_dir, dataset_name, model_name, pass_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    state["step"] = step
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def load(
    raw_dir: str,
    pass_name: str,
    dataset_name: str,
    model_name: str,
) -> dict[str, Any] | None:
    path = raw_path(raw_dir, dataset_name, model_name, pass_name)
    if not os.path.isfile(path):
        return None

    state = torch.load(path, map_location="cpu", weights_only=True)
    state["path"] = path
    return state
