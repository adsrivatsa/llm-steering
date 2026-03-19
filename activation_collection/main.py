from collections import Counter
import os
import argparse
from typing import Literal

from tqdm.auto import tqdm
import checkpoint
from dataset import SQuAD
from llm import ModelName, get_moe_llm
from transformers import AutoTokenizer
import torch


def faithfulness_unique_token_ids(
    model_name: ModelName, split: str = "train"
) -> set[int]:
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
    if task == "faithfulness":
        unique_token_ids = sorted(faithfulness_unique_token_ids(model_name, split))

    index_to_token_id: dict[int, int] = {}
    token_id_to_index: dict[int, int] = {}

    for idx, tid in enumerate(unique_token_ids):
        index_to_token_id[idx] = tid
        token_id_to_index[tid] = idx

    return index_to_token_id, token_id_to_index


def save_expert_activations(
    prompts: list[str],
    moe_model,
    token_id_to_index: dict[int, int],
    *,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = checkpoint.INTERVAL,
    checkpoint_pass_name: str = "",
    checkpoint_metadata: dict[str, str] | None = None,
    resume_from: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return moe_model.save_expert_activations(
        prompts,
        token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_pass_name=checkpoint_pass_name,
        checkpoint_metadata=checkpoint_metadata,
        resume_from=resume_from,
    )


def faithfulness_activations(
    model_name: ModelName,
    token_id_to_index: dict[int, int],
    *,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = checkpoint.INTERVAL,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    moe_model = get_moe_llm(model_name)

    dataset = SQuAD()

    prompts_x1: list[str] = []
    prompts_x2: list[str] = []

    for example in tqdm(dataset, desc="Building prompts"):
        prompts_x1.append(
            f"Document: {example['context']}, Question: {example['question']}, Answer: "
        )
        prompts_x2.append(f"Question: {example['question']}, Answer: ")

    dataset_name = "squad"
    metadata = {"dataset_name": dataset_name, "model_name": model_name}
    resume_x1 = None
    if checkpoint_dir:
        loaded_x1 = checkpoint.load(
            checkpoint_dir,
            pass_name="x1",
            dataset_name=dataset_name,
            model_name=model_name,
        )
        if loaded_x1 is not None:
            step = int(loaded_x1.get("step") or 0)
            if step > 0:
                resume_x1 = (
                    loaded_x1["expert_counts_by_token"],
                    loaded_x1["token_counts"],
                    step,
                )
                remaining = max(0, len(prompts_x1) - step)
                print(f"Resuming x1 from prompt {step} ({remaining} remaining)")

    expert_counts_x1, token_counts_x1 = moe_model.save_expert_activations(
        prompts_x1,
        token_id_to_index=token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_pass_name="x1",
        checkpoint_metadata=metadata,
        resume_from=resume_x1,
    )

    resume_x2 = None
    if checkpoint_dir:
        loaded_x2 = checkpoint.load(
            checkpoint_dir,
            pass_name="x2",
            dataset_name=dataset_name,
            model_name=model_name,
        )
        if loaded_x2 is not None:
            step = int(loaded_x2.get("step") or 0)
            if step > 0:
                resume_x2 = (
                    loaded_x2["expert_counts_by_token"],
                    loaded_x2["token_counts"],
                    step,
                )
                remaining = max(0, len(prompts_x2) - step)
                print(f"Resuming x2 from prompt {step} ({remaining} remaining)")

    expert_counts_x2, token_counts_x2 = moe_model.save_expert_activations(
        prompts_x2,
        token_id_to_index=token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_pass_name="x2",
        checkpoint_metadata=metadata,
        resume_from=resume_x2,
    )

    return expert_counts_x1, token_counts_x1, expert_counts_x2, token_counts_x2


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


def main(task: str, model_name: ModelName, checkpoint_dir: str) -> None:
    _, token_id_to_index = build_token_index_lookup(
        task=task, model_name=model_name, split="train"
    )

    if task == "faithfulness":
        faithfulness_activations(
            model_name=model_name,
            token_id_to_index=token_id_to_index,
            checkpoint_dir=checkpoint_dir,
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
        default="allenai/OLMoE-1B-7B-0125-Instruct",
        help="Hugging Face model identifier for the MoE LLM.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="activations")

    args = parser.parse_args()
    task: str = args.task
    model_name: ModelName = args.model_name
    checkpoint_dir: str = args.checkpoint_dir

    main(task=task, model_name=model_name, checkpoint_dir=checkpoint_dir)
