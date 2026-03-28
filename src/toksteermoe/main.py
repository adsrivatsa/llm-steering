import os

from transformers import AutoTokenizer

from src import checkpoint
from src.activation.dataset import SQuAD


os.environ["LLM_REGISTRATION"] = "toksteermoe"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import argparse
from typing import Literal

import torch
from tqdm.std import tqdm
from vllm import LLM

from src.benchmark import (
    cf_trivia_qa,
    faitheval_counterfactual,
    faitheval_inconsistent,
    faitheval_unanswerable,
    mctest,
    mquake,
)
from src.vllm_plugin import register as register_vllm_models


ModelName = Literal[
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    "microsoft/Phi-3.5-MoE-instruct",
]

MOE_EXPERT_CONFIG: dict[ModelName, tuple[int, int, int]] = {
    "openai/gpt-oss-20b": (24, 4, 32),
    "openai/gpt-oss-120b": (36, 4, 128),
    "Qwen/Qwen3-30B-A3B": (48, 8, 128),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (32, 2, 8),
    "allenai/OLMoE-1B-7B-0125-Instruct": (16, 8, 64),
    "microsoft/Phi-3.5-MoE-instruct": (32, 2, 16),
}


def calculate_delta(model_name: ModelName, task: str, activations_dir: str):
    if task == "faithfulness":
        train_dataset = "squad"

    x1_data = checkpoint.load(activations_dir, "x1", train_dataset, model_name)
    x2_data = checkpoint.load(activations_dir, "x2", train_dataset, model_name)

    A1, N1 = x1_data["A"], x1_data["N"]
    A2, N2 = x2_data["A"], x2_data["N"]

    p1 = A1.sum(dim=-1) / N1.sum()
    p2 = A2.sum(dim=-1) / N2.sum()

    delta = torch.nan_to_num(p1) - torch.nan_to_num(p2)
    token_wise_delta = A1 - A2

    assert (N1 - N2).sum() == 0

    return delta, token_wise_delta, N1


def risk_diff_to_manual_weights(
    delta: torch.Tensor, num_experts_per_tok: int, n_activated: int, n_deactivated: int
) -> torch.Tensor:
    d = delta.detach().cpu().float()
    num_layers, num_experts = d.shape
    moe_manual_weights = torch.zeros(num_layers, num_experts, dtype=torch.float32)

    flat = d.reshape(-1)
    order = torch.argsort(flat.abs(), descending=True)
    pos_per_layer = torch.zeros(num_layers, dtype=torch.int64)
    total_pos = 0
    total_neg = 0

    for t in order:
        idx = int(t.item())
        layer = idx // num_experts
        expert = idx % num_experts
        v = float(flat[idx].item())
        if v > 0:
            if (
                int(pos_per_layer[layer].item()) < num_experts_per_tok
                and total_pos < n_activated
            ):
                moe_manual_weights[layer, expert] = 1.0
                pos_per_layer[layer] += 1
                total_pos += 1
        elif v < 0:
            if total_neg < n_deactivated:
                moe_manual_weights[layer, expert] = -1.0
                total_neg += 1

    return moe_manual_weights


def steer(
    llm: LLM,
    token_wise_delta: torch.Tensor,
    token_freq: torch.Tensor,
    fallback_delta: torch.Tensor,
    idx_to_token_id: dict[int, int],
    n_activated: int,
    n_deactivated: int,
    eps: float = 0.01,
    fallback_threshold: int = 5,
) -> None:
    mc = llm.model_config
    _, num_experts_per_tok, _ = MOE_EXPERT_CONFIG[mc.model]

    fallback_weights = risk_diff_to_manual_weights(
        fallback_delta, num_experts_per_tok, n_activated, n_deactivated
    )  # (layers, experts)

    vocab_size = int(mc.hf_config.vocab_size)
    manual_weights = [fallback_weights.clone() for _ in range(vocab_size)]

    for i, (delta, freq) in tqdm(
        enumerate(zip(token_wise_delta, token_freq)),
        desc="Building manual weights",
        total=len(token_wise_delta),
    ):
        if freq <= fallback_threshold:
            continue
        token_id = idx_to_token_id[i]
        manual_weights[token_id] = risk_diff_to_manual_weights(
            delta, num_experts_per_tok, n_activated, n_deactivated
        )

    manual_weights = torch.stack(manual_weights)  # (vocab, layers, experts)

    def apply(self):
        model = self.model_runner.model
        model.add_toksteermoe_manual_args(manual_weights.clone(), eps)
        return ""

    llm.collective_rpc(apply)


def faithfulness_unique_token_ids(model_name: ModelName) -> set[int]:
    dataset = SQuAD()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    unique_token_ids: set[int] = set()
    for example in tqdm(dataset, desc="Collecting unique tokens"):
        question = f"Question: {example['question']}"
        for q_variant in [question, f" {question}"]:
            unique_token_ids.update(
                tokenizer.encode(q_variant, add_special_tokens=False)
            )

    return unique_token_ids


def build_token_index_lookup(
    task: str, model_name: ModelName
) -> tuple[dict[int, int], dict[int, int]]:
    if task == "faithfulness":
        unique_token_ids = sorted(faithfulness_unique_token_ids(model_name))

    index_to_token_id: dict[int, int] = {}
    token_id_to_index: dict[int, int] = {}

    for idx, tid in enumerate(unique_token_ids):
        index_to_token_id[idx] = tid
        token_id_to_index[tid] = idx

    return index_to_token_id, token_id_to_index


def main(
    model_name: ModelName,
    task: str,
    activations_dir: str,
    inference_dir: str,
    n_activated: int,
    n_deactivated: int,
):
    register_vllm_models()

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.49,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )

    delta, token_wise_delta, token_freq = calculate_delta(
        model_name=model_name, task=task, activations_dir=activations_dir
    )  # delta: (layers, experts), token_wise_delta: (layers, experts, vocab), token_freq: (vocab)

    token_wise_delta = token_wise_delta.permute(2, 0, 1)  # (vocab, layers, experts)

    idx_to_token_id, _ = build_token_index_lookup(task=task, model_name=model_name)

    steer(
        llm=llm,
        token_wise_delta=token_wise_delta,
        token_freq=token_freq,
        fallback_delta=delta,
        idx_to_token_id=idx_to_token_id,
        n_activated=n_activated,
        n_deactivated=n_deactivated,
        eps=0.01,
    )

    pass_name = f"a{n_activated}_d{n_deactivated}"

    if task == "faithfulness":
        score = faitheval_counterfactual.infer(
            llm=llm,
            checkpoint_dir=inference_dir,
            pass_name=pass_name,
            batch_size=4,
        )
        print(score)

        score = faitheval_unanswerable.infer(
            llm=llm,
            checkpoint_dir=inference_dir,
            pass_name=pass_name,
            batch_size=4,
        )
        print(score)

        score = faitheval_inconsistent.infer(
            llm=llm,
            checkpoint_dir=inference_dir,
            pass_name=pass_name,
            batch_size=4,
        )
        print(score)

        score = cf_trivia_qa.infer(
            llm=llm,
            checkpoint_dir=inference_dir,
            pass_name=pass_name,
            batch_size=4,
        )
        print(score)

        score = mquake.infer(
            llm=llm,
            checkpoint_dir=inference_dir,
            pass_name=pass_name,
            batch_size=4,
        )
        print(score)

        score = mctest.infer(
            llm=llm,
            checkpoint_dir=inference_dir,
            pass_name=pass_name,
            batch_size=4,
        )
        print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        choices=["faithfulness"],
        default="faithfulness",
    )
    parser.add_argument(
        "--llm",
        "--model",
        dest="model_name",
        type=str,
        default="allenai/OLMoE-1B-7B-0125-Instruct",
    )
    parser.add_argument("--activations-dir", type=str, default="activations")
    parser.add_argument("--inference-dir", type=str, default="inference")
    parser.add_argument("--experts-activated", type=int)
    parser.add_argument("--experts-deactivated", type=int)

    args = parser.parse_args()
    task: str = args.task
    model_name: ModelName = args.model_name
    activations_dir: str = args.activations_dir
    inference_dir: str = args.inference_dir
    n_activated: int = args.experts_activated
    n_deactivated: int = args.experts_deactivated

    main(
        model_name=model_name,
        task=task,
        activations_dir=activations_dir,
        inference_dir=inference_dir,
        n_activated=n_activated,
        n_deactivated=n_deactivated,
    )
