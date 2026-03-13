from __future__ import annotations

from typing import Literal

import torch
from tqdm.auto import tqdm

from faithfulness import FaithfulnessDataset
from llm import ModelName, get_moe_llm


TaskName = Literal["faithfulness"]


def _collect_expert_activation_counts(
    prompts: list[str],
    model_name: ModelName,
    token_id_to_index: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the model on the given prompts and collect, for each layer/expert/token,
    the number of times that expert was selected for that token.

    Returns:
        expert_counts_by_token: tensor of shape [num_layers, num_experts, num_tokens]
        token_counts: tensor of shape [num_tokens], giving how many times each
                      token appears across all prompts.
    """

    moe_model = get_moe_llm(model_name)
    return moe_model.collect_expert_activation_counts(prompts, token_id_to_index)


def get_steermoe_activations(
    dataset_name: str,
    task: TaskName,
    model_name: ModelName,
    token_id_to_index: dict[int, int],
) -> torch.Tensor:
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
    """
    if task != "faithfulness":
        raise ValueError("Only the 'faithfulness' task is currently supported.")

    dataset = FaithfulnessDataset(dataset_name=dataset_name, split="train")

    prompts_x1: list[str] = []
    prompts_x2: list[str] = []

    for document, question in tqdm(
        dataset, desc=f"Building prompts for steermoe ({dataset_name})"
    ):
        prompts_x1.append(f"Document: {document}, Question: {question}, Answer: ")
        prompts_x2.append(f"Question: {question}, Answer: ")

    expert_counts_x1, token_counts_x1 = _collect_expert_activation_counts(
        prompts_x1,
        model_name=model_name,
        token_id_to_index=token_id_to_index,
    )
    expert_counts_x2, token_counts_x2 = _collect_expert_activation_counts(
        prompts_x2,
        model_name=model_name,
        token_id_to_index=token_id_to_index,
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
    return risk_diff
