from collections import Counter
import os
import argparse

import torch
from tqdm.auto import tqdm
from faithfulness import FaithfulnessDataset, DatasetName
from llm import get_tokenizer, ModelName
from steermoe import get_steermoe_activations


def _collect_unique_token_ids(
    dataset_name: str, model_name: ModelName, split: str = "train"
) -> set[int]:
    """
    Collect the set of unique token IDs produced by `model_name`'s tokenizer
    over all (document, question) pairs in the specified dataset split.
    """
    dataset = FaithfulnessDataset(dataset_name=dataset_name, split=split)
    tokenizer = get_tokenizer(model_name)

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
    tokenizer = get_tokenizer(model_name)

    counter: Counter[int] = Counter()
    for document, question in tqdm(
        dataset, desc=f"Building frequency table ({dataset_name}, {split})"
    ):
        prompt = f"Document: {document}, Question: {question}, Answer: "
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        counter.update(input_ids)

    # Convert Counter to a plain dict[int, int]
    return dict(counter)


def main(dataset_name: DatasetName, model_name: ModelName) -> None:
    # n_unique = count_unique_tokens(
    #     dataset_name=dataset_name, model_name=model_name, split="train"
    # )
    # index_to_token_id, token_id_to_index = build_token_index_lookup(
    #     dataset_name=dataset_name, model_name=model_name, split="train"
    # )
    # freq_table = build_token_frequency_table(
    #     dataset_name=dataset_name, model_name=model_name, split="train"
    # )

    # Example: compute risk-difference matrix for the 'faithfulness' task.
    risk_diff = get_steermoe_activations(
        dataset_name=dataset_name,
        task="faithfulness",
        model_name=model_name,
    )

    # Convert risk difference (delta) matrix into epsilon matrix using a scalar eps.
    eps = 0.1  # TODO: tune this hyperparameter as needed
    epsilon = torch.zeros_like(risk_diff)
    epsilon[risk_diff > 0] = eps
    epsilon[risk_diff < 0] = -eps

    # Save epsilon and delta matrices under activations/epsilon and activations/delta.
    model_name_safe = model_name.replace("/", "_")
    eps_dir = os.path.join("activations", "epsilon", dataset_name)
    delta_dir = os.path.join("activations", "delta", dataset_name)
    os.makedirs(eps_dir, exist_ok=True)
    os.makedirs(delta_dir, exist_ok=True)

    eps_path = os.path.join(eps_dir, f"{model_name_safe}.pt")
    delta_path = os.path.join(delta_dir, f"{model_name_safe}.pt")

    torch.save(epsilon, eps_path)
    torch.save(risk_diff, delta_path)


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

    args = parser.parse_args()
    dataset_name: DatasetName = args.dataset  # type: ignore[assignment]
    model_name: ModelName = args.model_name  # type: ignore[assignment]

    main(dataset_name=dataset_name, model_name=model_name)
