from typing import Callable, Literal, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from device import device


def make_expert_token_router_hook(
    layer_idx: int,
    expert_counts_by_token: torch.Tensor,
    current_token_indices_ref: List[torch.Tensor | None],
    num_experts: int,
    num_token_indices: int,
    router_indices_output_index: int = 2,
    min_output_len: int | None = None,
) -> Callable[..., None]:
    """
    Returns a forward hook that updates expert_counts_by_token[layer_idx] from
    the router's output (output[router_indices_output_index] = router_indices
    with shape (tokens, k)). The caller must set current_token_indices_ref[0]
    to the per-position token index tensor before each forward.
    """
    if min_output_len is None:
        min_output_len = router_indices_output_index + 1

    def hook(module: torch.nn.Module, inputs: Tuple, output: Tuple) -> None:
        if not (isinstance(output, tuple) and len(output) >= min_output_len):
            return
        router_indices = output[router_indices_output_index]
        if not isinstance(router_indices, torch.Tensor) or router_indices.dim() != 2:
            return
        tokens, k = router_indices.shape
        if k <= 0 or k > num_experts:
            return

        current_token_indices = current_token_indices_ref[0]
        if current_token_indices is None or current_token_indices.numel() != tokens:
            return

        local_token_indices = current_token_indices.to(router_indices.device)
        token_idx_expanded = local_token_indices.view(-1, 1).expand_as(router_indices)
        valid = token_idx_expanded >= 0
        if not torch.any(valid):
            return

        expert_flat = router_indices[valid].to(torch.long)
        token_idx_flat = token_idx_expanded[valid].to(torch.long)
        pair_ids = expert_flat * num_token_indices + token_idx_flat
        pair_counts = torch.bincount(
            pair_ids,
            minlength=num_experts * num_token_indices,
        )
        expert_token_counts = pair_counts.view(num_experts, num_token_indices)
        expert_counts_by_token[layer_idx] += expert_token_counts.to(
            expert_counts_by_token.device
        )

    return hook


def _collect_expert_activation_counts_impl(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    layers: List[torch.nn.Module],
    prompts: List[str],
    token_id_to_index: dict[int, int],
    num_experts: int,
    get_router_module: Callable[[torch.nn.Module], torch.nn.Module],
    router_indices_output_index: int = 2,
    desc: str = "Running MoE inference",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generic implementation: run the model on prompts and collect per (layer, expert,
    token_index) activation counts. get_router_module(layer) should return the
    router submodule to hook (e.g. layer.mlp.router or layer.mlp.gate).
    """
    num_layers = len(layers)
    num_token_indices = len(token_id_to_index)
    expert_counts_by_token = torch.zeros(
        num_layers, num_experts, num_token_indices, dtype=torch.long
    )
    token_counts = torch.zeros(num_token_indices, dtype=torch.long)

    model_device = getattr(model, "device", next(model.parameters()).device)
    vocab_size = tokenizer.vocab_size
    id_to_index = torch.full(
        (vocab_size,), -1, dtype=torch.long, device=model_device
    )
    for tid, idx in token_id_to_index.items():
        if 0 <= tid < vocab_size:
            id_to_index[tid] = int(idx)

    hooks = []
    current_token_indices_ref: List[torch.Tensor | None] = [None]
    for i, layer in enumerate(layers):
        router_module = get_router_module(layer)
        h = router_module.register_forward_hook(
            make_expert_token_router_hook(
                layer_idx=i,
                expert_counts_by_token=expert_counts_by_token,
                current_token_indices_ref=current_token_indices_ref,
                num_experts=num_experts,
                num_token_indices=num_token_indices,
                router_indices_output_index=router_indices_output_index,
            )
        )
        hooks.append(h)

    with torch.inference_mode():
        for prompt in tqdm(prompts, desc=desc):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            encoded = {k: v.to(model_device) for k, v in encoded.items()}
            input_ids = encoded["input_ids"][0]
            current_token_indices_ref[0] = id_to_index[input_ids]
            for idx in current_token_indices_ref[0].tolist():
                if 0 <= idx < num_token_indices:
                    token_counts[idx] += 1
            _ = model(**encoded)

    for h in hooks:
        h.remove()
    return expert_counts_by_token, token_counts


ModelName = Literal[
    # OpenAI GPT-OSS series
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    # Qwen3
    "Qwen/Qwen3-30B-A3B",
    # Mixtral
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # OLMoE
    "allenai/OLMoE-1B-7B-0125",
    # Phi
    "microsoft/Phi-3.5-MoE-instruct",
]

# MoE configuration taken from the paper table:
#   Active / Total experts per layer.
MOE_CONFIG: dict[ModelName, Tuple[int, int]] = {
    "openai/gpt-oss-20b": (4, 32),
    "openai/gpt-oss-120b": (4, 128),
    "Qwen/Qwen3-30B-A3B": (8, 128),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (2, 8),
    "allenai/OLMoE-1B-7B-0125": (8, 64),
    "microsoft/Phi-3.5-MoE-instruct": (2, 16),
}


class GPT20MoELLM:
    """
    MoE LLM wrapper specialized for `openai/gpt-oss-20b`.

    GPT-OSS exposes router activations as a tuple:
        (router_logits, router_indices)
    on `layer.mlp.router`, with `router_indices` already containing the
    top-k expert indices per token.
    """

    def __init__(self, model_name: ModelName) -> None:
        if model_name != "openai/gpt-oss-20b":
            raise ValueError("GPT20MoELLM only supports 'openai/gpt-oss-20b'")

        self.model_name: ModelName = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        # Use architecture-known values from MOE_CONFIG.
        k_paper, num_experts_paper = MOE_CONFIG[model_name]
        # Total experts per layer from paper; assume matches implementation.
        self.num_experts = int(num_experts_paper)
        self.k = int(k_paper)
        self.num_layers = len(self.model.model.layers)

    def collect_expert_activation_counts(
        self,
        prompts: List[str],
        token_id_to_index: dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect expert activation counts using GPT-OSS router hooks."""
        return _collect_expert_activation_counts_impl(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=list(self.model.model.layers),
            prompts=prompts,
            token_id_to_index=token_id_to_index,
            num_experts=self.num_experts,
            get_router_module=lambda layer: layer.mlp.router,
            router_indices_output_index=2,
            desc=f"Running MoE inference ({self.model_name})",
        )


class Qwen30MoELLM:
    """
    MoE LLM wrapper specialized for `Qwen/Qwen3-30B-A3B`.

    For now this reuses the generic `MoELLM` behavior (gate detection +
    softmax + top-k), but is split out so we can add Qwen-specific routing
    logic or debugging without affecting other models.
    """

    def __init__(self, model_name: ModelName) -> None:
        if model_name != "Qwen/Qwen3-30B-A3B":
            raise ValueError("Qwen30MoELLM only supports 'Qwen/Qwen3-30B-A3B'")

        self.model_name: ModelName = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        # Use architecture-known values from MOE_CONFIG.
        k_paper, num_experts_paper = MOE_CONFIG[model_name]
        # Total experts per layer from paper; assume matches implementation.
        self.num_experts = int(num_experts_paper)
        self.k = int(k_paper)
        self.num_layers = len(self.model.model.layers)

    def collect_expert_activation_counts(
        self,
        prompts: List[str],
        token_id_to_index: dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect expert activation counts using Qwen router (gate) hooks."""
        return _collect_expert_activation_counts_impl(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=list(self.model.model.layers),
            prompts=prompts,
            token_id_to_index=token_id_to_index,
            num_experts=self.num_experts,
            get_router_module=lambda layer: layer.mlp.gate,
            router_indices_output_index=2,
            desc=f"Running MoE inference ({self.model_name})",
        )


def get_moe_llm(model_name: ModelName):
    """
    Factory to obtain a MoE LLM wrapper for the given model.

    Currently supports:
      - openai/gpt-oss-20b  -> GPT20MoELLM
      - Qwen/Qwen3-30B-A3B  -> Qwen30MoELLM

    Other model names are not yet implemented.
    """
    if model_name == "openai/gpt-oss-20b":
        return GPT20MoELLM(model_name=model_name)
    if model_name == "Qwen/Qwen3-30B-A3B":
        return Qwen30MoELLM(model_name=model_name)

    raise NotImplementedError(
        f"MoE wrapper not implemented for model {model_name!r}. "
        "Add a dedicated wrapper similar to GPT20MoELLM or Qwen30MoELLM."
    )
