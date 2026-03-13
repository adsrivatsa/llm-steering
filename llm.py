from typing import Literal, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from device import device


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


def get_tokenizer(model_name: ModelName) -> PreTrainedTokenizerBase:
    """
    Return the *actual* tokenizer used by the specified model.

    This uses Hugging Face `transformers.AutoTokenizer` under the hood, so
    `model_name` must correspond to a valid pretrained model identifier or
    local path.
    """
    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )


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
        self.tokenizer: PreTrainedTokenizerBase = get_tokenizer(model_name)

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
    ) -> Tuple[torch.Tensor, int]:
        """
        Collect expert activation counts using GPT-OSS router hooks.

        We register a hook on each layer's `mlp.router`, which returns:
            (router_logits, router_indices)
        where `router_indices` has shape (batch, seq, k).
        """
        num_layers = self.num_layers
        num_experts = self.num_experts

        expert_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
        total_tokens = 0

        hooks = []

        def make_router_hook(layer_idx: int):
            def hook(module, inputs, output):
                nonlocal expert_counts, total_tokens

                # HF MoE router returns a tuple of three tensors:
                # (logits, scores, indices). From inspection, the third tensor
                # (output[2]) contains the top-k expert indices with shape
                # (tokens, k).
                if not (isinstance(output, tuple) and len(output) >= 3):
                    return

                router_indices = output[2]
                if (
                    not isinstance(router_indices, torch.Tensor)
                    or router_indices.dim() != 2
                ):
                    return

                tokens, k = router_indices.shape
                if k <= 0 or k > num_experts:
                    return

                selected_experts = router_indices.reshape(-1)  # (tokens * k,)
                counts = torch.bincount(
                    selected_experts.to(torch.long),
                    minlength=num_experts,
                )  # (num_experts,)
                expert_counts[layer_idx] += counts.to(expert_counts.device)

                total_tokens += tokens

            return hook

        # Register hooks on each layer's router module.
        for i, layer in enumerate(self.model.model.layers):
            h = layer.mlp.router.register_forward_hook(make_router_hook(i))
            hooks.append(h)

        with torch.inference_mode():
            for prompt in tqdm(
                prompts, desc=f"Running MoE inference ({self.model_name})"
            ):
                text = prompt
                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                _ = self.model(**encoded)

        for h in hooks:
            h.remove()

        return expert_counts, total_tokens


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
        self.tokenizer: PreTrainedTokenizerBase = get_tokenizer(model_name)

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
    ) -> Tuple[torch.Tensor, int]:
        num_layers = self.num_layers
        num_experts = self.num_experts

        expert_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
        total_tokens = 0

        hooks = []

        def make_router_hook(layer_idx: int):
            def hook(module, inputs, output):
                nonlocal expert_counts, total_tokens

                # HF MoE router returns a tuple of three tensors:
                # (logits, scores, indices). We know from inspection that
                # the third tensor (output[2]) contains the top-k expert
                # indices with shape (tokens, k).
                if not (isinstance(output, tuple) and len(output) >= 2):
                    return

                router_indices = output[2]
                if (
                    not isinstance(router_indices, torch.Tensor)
                    or router_indices.dim() != 2
                ):
                    return

                tokens, k = router_indices.shape
                if k <= 0 or k > num_experts:
                    return

                selected_experts = router_indices.reshape(-1)  # (tokens * k,)
                counts = torch.bincount(
                    selected_experts.to(torch.long),
                    minlength=num_experts,
                )  # (num_experts,)
                expert_counts[layer_idx] += counts.to(expert_counts.device)

                total_tokens += tokens

            return hook

        for i, layer in enumerate(self.model.model.layers):
            h = layer.mlp.gate.register_forward_hook(make_router_hook(i))
            hooks.append(h)

        with torch.inference_mode():
            for prompt in tqdm(
                prompts, desc=f"Running MoE inference ({self.model_name})"
            ):
                text = prompt
                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                _ = self.model(**encoded)

        for h in hooks:
            h.remove()

        return expert_counts, total_tokens


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
