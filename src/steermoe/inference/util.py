import os

from src import checkpoint


def output_jsonl_path(
    checkpoint_dir: str, dataset_name: str, model_name: str, pass_name: str
) -> str:
    model_safe = checkpoint.safe_model_name(model_name)
    return os.path.join(checkpoint_dir, model_safe, dataset_name, f"{pass_name}.jsonl")


def nonempty_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
