import argparse
import os

from src import checkpoint

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


import torch
from typing import Literal, Tuple

from vllm import LLM, SamplingParams

from src.steermoe.vllm_plugin import register as register_vllm_models


ModelName = Literal[
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    "microsoft/Phi-3.5-MoE-instruct",
]

MOE_EXPERT_CONFIG: dict[ModelName, Tuple[int, int]] = {
    "openai/gpt-oss-20b": (4, 32),
    "openai/gpt-oss-120b": (4, 128),
    "Qwen/Qwen3-30B-A3B": (8, 128),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (2, 8),
    "allenai/OLMoE-1B-7B-0125-Instruct": (8, 64),
    "microsoft/Phi-3.5-MoE-instruct": (2, 16),
}

EXPERT_ACTIVATION_DEACTIVATION: dict[str, Tuple[int, int]] = {
    "openai/gpt-oss-120b": (5, 100),
    "openai/gpt-oss-20b": (10, 50),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (10, 100),
    "allenai/OLMoE-1B-7B-0125-Instruct": (0, 50),
    "microsoft/Phi-3.5-MoE-instruct": (10, 75),
    "Qwen/Qwen3-30B-A3B": (0, 500),
}


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


def steer(llm: LLM, delta: torch.Tensor, eps: float = 0.01) -> None:
    mc = llm.model_config
    num_experts_per_tok, _ = MOE_EXPERT_CONFIG[mc.model]
    n_activated, n_deactivated = EXPERT_ACTIVATION_DEACTIVATION[mc.model]
    manual_weights: torch.Tensor = risk_diff_to_manual_weights(
        delta, num_experts_per_tok, n_activated, n_deactivated
    )

    def apply(self):
        model = self.model_runner.model
        model.add_steermoe_manual_args(manual_weights.clone(), eps)
        return ""

    llm.collective_rpc(apply)


def calculate_delta(model_name: ModelName, task: str, activations_dir: str):
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

    return delta


def main(
    model_name: ModelName,
    task: str,
    activations_dir: str,
    inference_dir: str,
    use_steering: bool,
):
    register_vllm_models()

    llm = LLM(
        model=model_name,
        # max_seq_len_to_capture=4096,
        max_model_len=4096,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )

    if use_steering:
        delta = calculate_delta(
            model_name=model_name, task=task, activations_dir=activations_dir
        )
        steer(llm=llm, delta=delta, eps=0.01)

    batch_messages = [
        [
            {
                "role": "user",
                "content": "Document: iPod was developed by Google\n Question: Who is the developer of iPod? \n Final Answer Only:",
            }
        ],
        [
            {
                "role": "user",
                "content": "Document: The chief executive officer of Google is Lakshmi Mittal\n Question: Who is the chief executive officer of Google? \n Final Answer Only:",
            }
        ],
        [
            {
                "role": "user",
                "content": "Document: Anderson Cooper is employed by National Review\n Question: Who is the employer of Anderson Cooper? \n Final Answer Only:",
            }
        ],
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        top_k=1,
        min_p=0,
        max_tokens=512,
        seed=0,
    )
    outputs = llm.chat(
        batch_messages,
        sampling_params,
        use_tqdm=True,
        chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"},
    )
    generations = [output.outputs[0].text for output in outputs]
    for generation in generations:
        print(generation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        choices=["faithfulness"],
        default="faithfulness",
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
        default="openai/gpt-oss-20b",
    )
    parser.add_argument("--activations-dir", type=str, default="activations")
    parser.add_argument("--inference-dir", type=str, default="inference")
    parser.add_argument(
        "--no-steering",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    task: str = args.task
    model_name: ModelName = args.model_name
    dataset: str = args.dataset
    activations_dir: str = args.activations_dir
    inference_dir: str = args.inference_dir
    no_steering: bool = args.no_steering

    main(
        model_name=model_name,
        task=task,
        activations_dir=activations_dir,
        inference_dir=inference_dir,
        use_steering=not no_steering,
    )
