from __future__ import annotations

import torch
from tqdm.auto import tqdm

import checkpoint
from dataset import FaithfulnessDataset
from llm import ModelName, get_moe_llm


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
    """
    Run the model on the given prompts and collect, for each layer/expert/token,
    the number of times that expert was selected for that token.

    Returns:
        expert_counts_by_token: tensor of shape [num_layers, num_experts, num_tokens]
        token_counts: tensor of shape [num_tokens], giving how many times each
                      token appears across all prompts.
    """
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
    dataset_name: str,
    model_name: ModelName,
    token_id_to_index: dict[int, int],
    *,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = checkpoint.INTERVAL,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the risk-difference matrix for a given dataset and MoE model.

    For now, only the 'faithfulness' task is supported.

    Definitions:
        x1 = \"Document: {document}, Question: {question}, Answer: \"
        x2 = \"Question: {question}, Answer: \"

        For a given expert e:
            p1 = (# tokens in x1 for which e was selected) / (total tokens in x1)
            p2 = (# tokens in x2 for which e was selected) / (total tokens in x2)
            risk_diff = p1 - p2

    Internally, we keep counts per (expert, token_index) and only aggregate
    across tokens at the final step when computing risk differences.

    Returns:
        risk_diff: tensor of shape [num_layers, num_experts]
        raw_state: dict with expert_counts_x1, expert_counts_x2, token_counts_x1,
                   token_counts_x2 (unaggregated, for saving raw deltas).
    """
    moe_model = get_moe_llm(model_name)

    dataset = FaithfulnessDataset(dataset_name=dataset_name, split="train")

    prompts_x1: list[str] = []
    prompts_x2: list[str] = []

    for document, question in tqdm(
        dataset, desc=f"Building prompts for steermoe ({dataset_name})"
    ):
        prompts_x1.append(f"Document: {document}, Question: {question}, Answer: ")
        prompts_x2.append(f"Question: {question}, Answer: ")

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
