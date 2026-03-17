from collections import Counter
import os
import argparse

from tqdm.auto import tqdm
import checkpoint
from faithfulness import FaithfulnessDataset, DatasetName
from llm import ModelName
from steer import faithfulness_activations
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
) -> None:
    # Build the token index lookup once and pass the mapping from
    # token_id -> local index into the SteerMoE computation so we can
    # track expert activations at the (expert, token) level.
    _, token_id_to_index = build_token_index_lookup(
        dataset_name=dataset_name, model_name=model_name, split="train"
    )

    # Compute raw expert activation counts for both x1 and x2 passes.
    faithfulness_activations(
        dataset_name=dataset_name,
        model_name=model_name,
        token_id_to_index=token_id_to_index,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save raw SteerMoE activations."
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
        default="microsoft/Phi-3.5-MoE-instruct",
        help="Hugging Face model identifier for the MoE LLM.",
    )

    args = parser.parse_args()
    dataset_name: DatasetName = args.dataset  # type: ignore[assignment]
    model_name: ModelName = args.model_name  # type: ignore[assignment]

    main(
        dataset_name=dataset_name,
        model_name=model_name,
    )
