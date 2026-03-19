import argparse

from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer

from activation_collection import checkpoint
from llm import ModelName
from activation_collection.dataset import SQuAD
from inference import faitheval_couterfactual_inference


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


def main(
    task: str,
    model_name: ModelName,
    dataset: str,
    activations_dir: str,
    inference_dir: str,
):
    # _, token_id_to_index = build_token_index_lookup(
    #     task=task, model_name=model_name, split="train"
    # )

    if task == "faithfulness":
        train_dataset = "squad"

    x1_data = checkpoint.load(activations_dir, "x1", train_dataset, model_name)
    x2_data = checkpoint.load(activations_dir, "x2", train_dataset, model_name)

    x1_expert_activation_counts, x1_token_counts = (
        x1_data["expert_counts_by_token"],
        x1_data["token_counts"],
    )
    x2_expert_activation_counts, x2_token_counts = (
        x2_data["expert_counts_by_token"],
        x2_data["token_counts"],
    )
    p1 = x1_expert_activation_counts.sum(dim=-1) / x1_token_counts.sum()
    p2 = x2_expert_activation_counts.sum(dim=-1) / x2_token_counts.sum()
    delta = torch.nan_to_num(p1) - torch.nan_to_num(p2)

    if task == "faithfulness":
        if dataset == "faitheval_counterfactual":
            faitheval_couterfactual_inference(
                model_name=model_name,
                checkpoint_dir=inference_dir,
                pass_name="steered",
                deltas_per_layer=delta,
                n_activate=0,
                n_deactivate=50,
                max_new_tokens=32,
                batch_size=4,
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
        "--dataset",
        type=str,
        choices=[
            "faitheval_counterfactual",
            "faitheval_inconsistent",
            "faitheval_unanswerable",
        ],
    )
    parser.add_argument(
        "--llm",
        "--model",
        dest="model_name",
        type=str,
        default="allenai/OLMoE-1B-7B-0125-Instruct",
        help="Hugging Face model identifier for the MoE LLM.",
    )
    parser.add_argument(
        "--activations-dir", type=str, default="activation_collection/activations"
    )
    parser.add_argument("--inference-dir", type=str, default="inference")

    args = parser.parse_args()
    task: str = args.task
    model_name: ModelName = args.model_name
    dataset: str = args.dataset
    activations_dir: str = args.activations_dir
    inference_dir: str = args.inference_dir

    main(
        task=task,
        model_name=model_name,
        dataset=dataset,
        activations_dir=activations_dir,
        inference_dir=inference_dir,
    )
