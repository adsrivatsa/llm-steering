import argparse
import os

from transformers import AutoConfig, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["LLM_REGISTRATION"] = "activation"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import torch
from tqdm.std import tqdm
from typing import Literal
from vllm import LLM, SamplingParams

from src import checkpoint
from src.activation.dataset import SQuAD
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


def find_question_token_range(
    tokenizer: AutoTokenizer,
    content: str,
    question: str,
) -> tuple[int, int]:
    """
    Returns the (start, end) exclusive token indices of the question text within
    the full chat-formatted + generation-prompt token sequence.

    This uses token-subsequence matching against the exact prompt token IDs that
    the model sees, which is more robust than char-offset mapping for BPE
    boundary effects (e.g. leading-space tokenization differences between x1/x2).
    """
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )
    full_token_ids: list[int] = tokenizer(full_text, add_special_tokens=False)[
        "input_ids"
    ]

    # Two variants are needed because in x1 the question follows "Question: "
    # and often starts with a leading-space token, while x2 is standalone.
    candidate_questions = [question, f" {question}"]
    candidates: list[list[int]] = []
    for q in candidate_questions:
        ids = tokenizer(q, add_special_tokens=False)["input_ids"]
        if ids and ids not in candidates:
            candidates.append(ids)

    if not candidates:
        return 0, 0

    best_start: int | None = None
    best_end: int | None = None

    for target_ids in candidates:
        t_len = len(target_ids)
        if t_len == 0 or t_len > len(full_token_ids):
            continue
        # Keep the last match, mirroring prior rfind behavior when question
        # text appears earlier in context.
        for i in range(len(full_token_ids) - t_len + 1):
            if full_token_ids[i : i + t_len] == target_ids:
                best_start = i
                best_end = i + t_len

    if best_start is None or best_end is None:
        return 0, 0
    return best_start, best_end


def resolve_parallelism(model_name: str) -> tuple[int, int]:
    """Return (tensor_parallel_size, pipeline_parallel_size) for the available GPUs.

    Prefers full tensor parallelism. Falls back to full pipeline parallelism when
    the number of KV attention heads is not divisible by the GPU count (e.g. 3 GPUs).
    """
    n_gpus = torch.cuda.device_count()
    print(n_gpus)
    if n_gpus <= 1:
        return 1, 1

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # num_key_value_heads is the binding TP constraint; fall back to full heads for MHA models
    kv_heads = getattr(
        config,
        "num_key_value_heads",
        getattr(config, "num_attention_heads", None),
    )

    if kv_heads is not None and kv_heads % n_gpus == 0:
        return n_gpus, 1  # tensor parallelism

    return 1, n_gpus  # pipeline parallelism


def collect_prompt_activations(
    llm: LLM,
    dataset_name: str,
    pass_name: str,
    prompts: list[tuple[str, int, int]],
    token_id_to_index: dict[int, int],
    *,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = checkpoint.INTERVAL,
):
    model_name = llm.model_config.model

    layers, k, experts = MOE_EXPERT_CONFIG[model_name]

    A = torch.zeros(layers, experts, len(token_id_to_index))
    N = torch.zeros(len(token_id_to_index), dtype=torch.long)

    start_step = 0
    if checkpoint_dir:
        loaded = checkpoint.load(
            raw_dir=checkpoint_dir,
            pass_name=pass_name,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        if loaded:
            start_step = int(loaded.get("step", 0))
            if start_step > 0:
                A = loaded["A"]
                N = loaded["N"]
                remaining = max(0, len(prompts) - start_step)
                print(
                    f"Resuming {pass_name} from prompt {start_step}, {remaining} remaining"
                )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = SamplingParams(
        temperature=0, top_p=0.8, top_k=1, min_p=0, max_tokens=1, seed=0
    )

    raw_save_dir = f"{checkpoint_dir}/raw/{checkpoint.safe_model_name(model_name)}"
    os.makedirs(raw_save_dir, exist_ok=True)

    llm.collective_rpc(lambda self: self.model_runner.model.add_save_dir(raw_save_dir))

    for idx, prompt in tqdm(
        enumerate(prompts[start_step:]),
        desc=f"Running MoE inference: {model_name} {dataset_name} {pass_name}",
        total=len(prompts) - start_step,
    ):
        prompt, q_start, q_end = prompt
        message = [{"role": "user", "content": prompt}]
        _ = llm.chat(
            message,
            sampling_params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"},
        )

        expert_activations = []
        for activations_file in sorted(
            os.listdir(raw_save_dir), key=lambda f: int(os.path.splitext(f)[0])
        ):
            activations_path = os.path.join(raw_save_dir, activations_file)
            activations = torch.load(activations_path)
            expert_activations.append(activations)
        activation_logits = torch.stack(expert_activations).permute(0, 2, 1)

        activation_logits = activation_logits[
            :, :, q_start:q_end
        ]  # [layers, experts, q_len]

        q_len = q_end - q_start

        # Top-k expert indices per (layer, token): [layers, k, q_len]
        top_k_indices = torch.topk(activation_logits, k, dim=1).indices

        # Binary mask: 1 where expert was among top-k, else 0, shape [layers, experts, q_len]
        expert_mask = torch.full((layers, experts, q_len), 0.0, dtype=torch.float32)
        expert_mask.scatter_(1, top_k_indices, 1.0)

        # Get token IDs for the question tokens from the full chat-formatted
        # sequence so they match exactly what the model processed.
        full_text = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        full_token_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        question_token_ids = full_token_ids[q_start:q_end]

        # Filter to positions whose token IDs are in the index map
        valid_positions = [
            i for i, tid in enumerate(question_token_ids) if tid in token_id_to_index
        ]
        token_indices = torch.tensor(
            [token_id_to_index[question_token_ids[i]] for i in valid_positions],
            dtype=torch.long,
        )

        # Accumulate: A1[layer, expert, token_index] += 1 for each activated (layer, expert, token)
        A.index_add_(2, token_indices, expert_mask[:, :, valid_positions])

        # Count occurrences of each token across all question prompts
        N.index_add_(0, token_indices, torch.ones(len(token_indices), dtype=torch.long))

        if checkpoint_dir and (start_step + idx + 1) % checkpoint_interval == 0:
            checkpoint.save(
                raw_dir=checkpoint_dir,
                pass_name=pass_name,
                state={"A": A, "N": N},
                dataset_name=dataset_name,
                model_name=model_name,
                step=start_step + idx + 1,
            )

    # final checkpoint save
    if checkpoint_dir:
        checkpoint.save(
            raw_dir=checkpoint_dir,
            pass_name=pass_name,
            state={"A": A, "N": N},
            dataset_name=dataset_name,
            model_name=model_name,
            step=start_step + idx + 1,
        )

    return A, N


def faithfulness_activations(
    model_name: ModelName,
    token_id_to_index: dict[int, int],
    *,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = checkpoint.INTERVAL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    dataset = SQuAD()

    prompts_x1: list[tuple[str, int, int]] = []
    prompts_x2: list[tuple[str, int, int]] = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for example in tqdm(dataset, desc="Building prompts"):
        document = f"Document {example['context']}"
        question = f"Question {example['question']}"

        x1 = f"{document} {question}"
        x2 = question

        # Find the exact token positions of the question within the
        # chat-formatted sequence so we can slice activation_logits correctly.
        q_start_x1, q_end_x1 = find_question_token_range(tokenizer, x1, question)
        assert q_start_x1 != 0 and q_end_x1 != 0

        q_start_x2, q_end_x2 = find_question_token_range(tokenizer, x2, question)
        assert q_start_x2 != 0 and q_end_x2 != 0

        prompts_x1.append((x1, q_start_x1, q_end_x1))
        prompts_x2.append((x2, q_start_x2, q_end_x2))

    tp, pp = resolve_parallelism(model_name)
    print(f"Parallelism: tensor_parallel_size={tp}, pipeline_parallel_size={pp}")

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.49,
    )

    A1, N1 = collect_prompt_activations(
        llm=llm,
        dataset_name="squad",
        pass_name="x1",
        prompts=prompts_x1,
        token_id_to_index=token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )
    A2, N2 = collect_prompt_activations(
        llm=llm,
        dataset_name="squad",
        pass_name="x2",
        prompts=prompts_x2,
        token_id_to_index=token_id_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )

    return A1, N1, A2, N2


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


def main(task: str, model_name: ModelName, checkpoint_dir: str):
    register_vllm_models()

    _, token_id_to_idx = build_token_index_lookup(task=task, model_name=model_name)

    faithfulness_activations(
        model_name=model_name,
        token_id_to_index=token_id_to_idx,
        checkpoint_dir=checkpoint_dir,
    )


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
    parser.add_argument("--checkpoint-dir", type=str, default="activations")

    args = parser.parse_args()
    task: str = args.task
    model_name: ModelName = args.model_name
    checkpoint_dir: str = args.checkpoint_dir

    main(task=task, model_name=model_name, checkpoint_dir=checkpoint_dir)
