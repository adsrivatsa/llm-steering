from collections import Counter
import os
import argparse

import torch
from tqdm.auto import tqdm
from checkpoint import save_final_activations, save_final_raw
from faithfulness import FaithfulnessDataset, DatasetName
from llm import ModelName
from steermoe import get_steermoe_activations
from transformers import AutoTokenizer


def _collect_unique_token_ids(
    dataset_name: str, model_name: ModelName, split: str = "train"
) -> set[int]:
    """
    Collect the set of unique token IDs produced by `model_name`'s tokenizer
    over all (document, question) pairs in the specified dataset split.
    """
    dataset = FaithfulnessDataset(dataset_name=dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    unique_token_ids: set[int] = set()
    for document, question in tqdm(
        dataset, desc=f"Collecting unique tokens ({dataset_name}, {split})"
    ):
        prompt = f"Document: {document}, Question: {question}, Answer: "
        encoded = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        unique_token_ids.update(encoded)

    return unique_token_ids


def count_unique_tokens(
    dataset_name: str, model_name: ModelName, split: str = "train"
) -> int:
    """
    Count the number of unique tokens produced by `model_name`'s tokenizer
    over all (document, question) pairs in the specified dataset split.
    """
    return len(_collect_unique_token_ids(dataset_name, model_name, split))


def build_token_index_lookup(
    dataset_name: str, model_name: ModelName, split: str = "train"
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Build lookup tables between a dense local index space and the
    (sparse) encoded token IDs that actually appear in the dataset.

    Returns:
        index_to_token_id: maps local index -> token ID
        token_id_to_index: maps token ID -> local index
    """
    unique_token_ids = sorted(
        _collect_unique_token_ids(dataset_name, model_name, split)
    )

    index_to_token_id: dict[int, int] = {}
    token_id_to_index: dict[int, int] = {}

    for idx, tid in enumerate(unique_token_ids):
        index_to_token_id[idx] = tid
        token_id_to_index[tid] = idx

    return index_to_token_id, token_id_to_index


def build_token_frequency_table(
    dataset_name: str, model_name: ModelName, split: str = "train"
) -> dict[int, int]:
    """
    Build a frequency table mapping token ID -> count across the dataset
    for the given model's tokenizer.
    """
    dataset = FaithfulnessDataset(dataset_name=dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    counter: Counter[int] = Counter()
    for document, question in tqdm(
        dataset, desc=f"Building frequency table ({dataset_name}, {split})"
    ):
        prompt = f"Document: {document}, Question: {question}, Answer: "
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        counter.update(input_ids)

    # Convert Counter to a plain dict[int, int]
    return dict(counter)


def main(
    dataset_name: DatasetName,
    model_name: ModelName,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 10,
) -> None:
    # Build the token index lookup once and pass the mapping from
    # token_id -> local index into the SteerMoE computation so we can
    # track expert activations at the (expert, token) level.
    _, token_id_to_index = build_token_index_lookup(
        dataset_name=dataset_name, model_name=model_name, split="train"
    )

    # Compute risk-difference matrix with optional checkpointing every 10 prompts.
    risk_diff, raw_state = get_steermoe_activations(
        dataset_name=dataset_name,
        task="faithfulness",
        model_name=model_name,
        token_id_to_index=token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )

    # Convert risk difference (delta) matrix into epsilon matrix using a scalar eps.
    eps = 0.1  # TODO: tune this hyperparameter as needed
    epsilon = torch.zeros_like(risk_diff)
    epsilon[risk_diff > 0] = eps
    epsilon[risk_diff < 0] = -eps

    # Save aggregated epsilon and delta under activations/epsilon and activations/delta.
    eps_dir = os.path.join("activations", "epsilon")
    delta_dir = os.path.join("activations", "delta")
    eps_path, delta_path = save_final_activations(
        eps_dir,
        delta_dir,
        epsilon,
        risk_diff,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    # Save final raw deltas (unaggregated counts) under activations/raw.
    raw_dir = os.path.join("activations", "raw")
    save_final_raw(
        raw_dir,
        raw_state["expert_counts_x1"],
        raw_state["expert_counts_x2"],
        raw_state["token_counts_x1"],
        raw_state["token_counts_x2"],
        dataset_name=dataset_name,
        model_name=model_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save SteerMoE activations (epsilon and delta matrices)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        help="Dataset name (currently only 'squad' is supported).",
    )
    parser.add_argument(
        "--llm",
        "--model",
        dest="model_name",
        type=str,
        default="openai/gpt-oss-20b",
        help="Hugging Face model identifier for the MoE LLM.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for collection checkpoints (every N prompts). Enables save/resume. Use '' to disable.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save a checkpoint every this many prompts (default: 10).",
    )

    args = parser.parse_args()
    dataset_name: DatasetName = args.dataset  # type: ignore[assignment]
    model_name: ModelName = args.model_name  # type: ignore[assignment]

    # Use default "checkpoints" so resume works; pass --checkpoint-dir '' to disable.
    checkpoint_dir = args.checkpoint_dir or None

    main(
        dataset_name=dataset_name,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
