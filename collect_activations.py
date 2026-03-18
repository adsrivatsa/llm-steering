from collections import Counter
import os
import argparse
from typing import Literal

from tqdm.auto import tqdm
import checkpoint
from dataset import SQuAD
from llm import ModelName
from steer import faithfulness_activations
from transformers import AutoTokenizer


def faithfulness_unique_token_ids(
    model_name: ModelName, split: str = "train"
) -> set[int]:
    """
    Collect the set of unique token IDs produced by `model_name`'s tokenizer
    over all (document, question) pairs in the specified dataset split.
    """
    dataset = SQuAD()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    unique_token_ids: set[int] = set()
    for example in tqdm(dataset, desc="Collecting unique tokens"):
        for prompt in (
            f"Document: {example['context']}, Question: {example['question']}, Answer: ",
            f"Question: {example['question']}, Answer: ",
        ):
            encoded = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            unique_token_ids.update(encoded)

    return unique_token_ids


def build_token_index_lookup(
    task: str, model_name: ModelName, split: str = "train"
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Build lookup tables between a dense local index space and the
    (sparse) encoded token IDs that actually appear in the dataset.

    Returns:
        index_to_token_id: maps local index -> token ID
        token_id_to_index: maps token ID -> local index
    """
    if task == "faithfulness":
        unique_token_ids = sorted(faithfulness_unique_token_ids(model_name, split))

    index_to_token_id: dict[int, int] = {}
    token_id_to_index: dict[int, int] = {}

    for idx, tid in enumerate(unique_token_ids):
        index_to_token_id[idx] = tid
        token_id_to_index[tid] = idx

    return index_to_token_id, token_id_to_index


# def faithfulness_frequency_table(
#     model_name: ModelName, split: str = "train"
# ) -> dict[int, int]:
#     """
#     Build a frequency table mapping token ID -> count across the dataset
#     for the given model's tokenizer.
#     """
#     dataset = SQuAD()
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         use_fast=True,
#         trust_remote_code=True,
#     )

#     counter: Counter[int] = Counter()
#     for example in tqdm(
#         dataset, desc=f"Building frequency table ({dataset_name}, {split})"
#     ):
#         for prompt in (
#             f"Document: {example['context']}, Question: {example['question']}, Answer: ",
#             f"Question: {example['question']}, Answer: ",
#         ):
#             input_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
#             counter.update(input_ids)

#     # Convert Counter to a plain dict[int, int]
#     return dict(counter)


def main(
    task: str,
    model_name: ModelName,
) -> None:
    # Build the token index lookup once and pass the mapping from
    # token_id -> local index into the SteerMoE computation so we can
    # track expert activations at the (expert, token) level.
    _, token_id_to_index = build_token_index_lookup(
        task=task, model_name=model_name, split="train"
    )

    if task == "faithfulness":
        faithfulness_activations(
            model_name=model_name,
            token_id_to_index=token_id_to_index,
            checkpoint_dir=checkpoint.DIR,
            checkpoint_interval=checkpoint.INTERVAL,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save raw SteerMoE activations."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["faithfulness"],
        default="faithfulness",
        help="Task to collect expert activations on.",
    )
    parser.add_argument(
        "--llm",
        "--model",
        dest="model_name",
        type=str,
        default="allenai/OLMoE-1B-7B-0125",
        help="Hugging Face model identifier for the MoE LLM.",
    )

    args = parser.parse_args()
    task: str = args.task
    model_name: ModelName = args.model_name  # type: ignore[assignment]

    main(
        task=task,
        model_name=model_name,
    )
