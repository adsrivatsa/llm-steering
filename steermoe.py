from __future__ import annotations

import os
from typing import Literal

import torch
from tqdm.auto import tqdm

from checkpoint import CHECKPOINT_INTERVAL, load_collection_checkpoint
from faithfulness import FaithfulnessDataset
from llm import ModelName, get_moe_llm


TaskName = Literal["faithfulness"]


def _collect_expert_activation_counts(
    prompts: list[str],
    moe_model,
    token_id_to_index: dict[int, int],
    *,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
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
    return moe_model.collect_expert_activation_counts(
        prompts,
        token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_pass_name=checkpoint_pass_name,
        checkpoint_metadata=checkpoint_metadata,
        resume_from=resume_from,
    )


def get_steermoe_activations(
    dataset_name: str,
    task: TaskName,
    model_name: ModelName,
    token_id_to_index: dict[int, int],
    *,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
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

    If checkpoint_dir is set, collection state is saved every checkpoint_interval
    prompts and can be resumed on the next run.

    Returns:
        risk_diff: tensor of shape [num_layers, num_experts]
        raw_state: dict with expert_counts_x1, expert_counts_x2, token_counts_x1,
                   token_counts_x2 (unaggregated, for saving raw deltas).
    """
    if task != "faithfulness":
        raise ValueError("Only the 'faithfulness' task is currently supported.")

    # Load once and reuse across x1/x2 passes to avoid a second heavyweight
    # model initialization (which can trigger additional OOMs).
    moe_model = get_moe_llm(model_name)

    dataset = FaithfulnessDataset(dataset_name=dataset_name, split="train")

    prompts_x1: list[str] = []
    prompts_x2: list[str] = []

    for document, question in tqdm(
        dataset, desc=f"Building prompts for steermoe ({dataset_name})"
    ):
        prompts_x1.append(f"Document: {document}, Question: {question}, Answer: ")
        prompts_x2.append(f"Question: {question}, Answer: ")

    meta = {"dataset_name": dataset_name, "model_name": model_name}

    # Resolve to absolute path so resume works regardless of cwd
    if checkpoint_dir:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        print(f"Checkpoint directory: {checkpoint_dir}")

    resume_x1 = None
    if checkpoint_dir:
        loaded = load_collection_checkpoint(
            checkpoint_dir, "x1", dataset_name=dataset_name, model_name=model_name
        )
        if loaded is not None:
            expert_counts_x1, token_counts_x1, step, _ = loaded
            resume_x1 = (expert_counts_x1, token_counts_x1, step)
            remaining = len(prompts_x1) - step
            print(f"Resuming x1 from prompt {step} ({remaining} remaining)")

    expert_counts_x1, token_counts_x1 = _collect_expert_activation_counts(
        prompts_x1,
        moe_model=moe_model,
        token_id_to_index=token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_pass_name="x1",
        checkpoint_metadata=meta,
        resume_from=resume_x1,
    )

    resume_x2 = None
    if checkpoint_dir:
        loaded = load_collection_checkpoint(
            checkpoint_dir, "x2", dataset_name=dataset_name, model_name=model_name
        )
        if loaded is not None:
            expert_counts_x2, token_counts_x2, step, _ = loaded
            resume_x2 = (expert_counts_x2, token_counts_x2, step)
            remaining = len(prompts_x2) - step
            print(f"Resuming x2 from prompt {step} ({remaining} remaining)")

    expert_counts_x2, token_counts_x2 = _collect_expert_activation_counts(
        prompts_x2,
        moe_model=moe_model,
        token_id_to_index=token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_pass_name="x2",
        checkpoint_metadata=meta,
        resume_from=resume_x2,
    )

    total_tokens_x1 = int(token_counts_x1.sum().item())
    total_tokens_x2 = int(token_counts_x2.sum().item())

    if total_tokens_x1 == 0 or total_tokens_x2 == 0:
        raise RuntimeError("No tokens processed when computing expert activations.")

    # Aggregate over the token dimension at the end.
    expert_counts_x1_agg = expert_counts_x1.sum(dim=2)  # [num_layers, num_experts]
    expert_counts_x2_agg = expert_counts_x2.sum(dim=2)  # [num_layers, num_experts]

    p1 = expert_counts_x1_agg.to(torch.float32) / float(total_tokens_x1)
    p2 = expert_counts_x2_agg.to(torch.float32) / float(total_tokens_x2)

    risk_diff = p1 - p2  # [num_layers, num_experts]
    raw_state = {
        "expert_counts_x1": expert_counts_x1,
        "expert_counts_x2": expert_counts_x2,
        "token_counts_x1": token_counts_x1,
        "token_counts_x2": token_counts_x2,
    }
    return risk_diff, raw_state
