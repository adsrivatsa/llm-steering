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
        token_id_to_index: dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect expert activation counts using GPT-OSS router hooks.

        We register a hook on each layer's `mlp.router`, which returns:
            (router_logits, router_indices)
        where `router_indices` has shape (tokens, k).

        Instead of aggregating only over experts, we record counts per
        (expert, token_index) pair, where `token_index` is a dense local
        index derived from `token_id_to_index`.
        """
        num_layers = self.num_layers
        num_experts = self.num_experts
        num_token_indices = len(token_id_to_index)

        # expert_counts_by_token[layer, expert, token_index] = count
        expert_counts_by_token = torch.zeros(
            num_layers, num_experts, num_token_indices, dtype=torch.long
        )
        # token_counts[token_index] = number of times this token appears
        token_counts = torch.zeros(num_token_indices, dtype=torch.long)

        # Dense lookup table from tokenizer vocab id -> local token index.
        vocab_size = self.tokenizer.vocab_size
        id_to_index = torch.full(
            (vocab_size,),
            -1,
            dtype=torch.long,
            device=self.model.device,
        )
        for tid, idx in token_id_to_index.items():
            if 0 <= tid < vocab_size:
                id_to_index[tid] = int(idx)

        hooks = []
        current_token_indices: torch.Tensor | None = None

        def make_router_hook(layer_idx: int):
            def hook(module, inputs, output):
                nonlocal expert_counts_by_token, current_token_indices

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

                # `current_token_indices` is a 1D tensor of length `tokens`
                # giving the local token_index for each token position.
                if current_token_indices is None:
                    return
                if current_token_indices.numel() != tokens:
                    return

                # Move token indices onto the same device as router_indices.
                local_token_indices = current_token_indices.to(router_indices.device)

                # Vectorized accumulation of counts per (expert, token_index).
                # Expand token indices to align with router_indices (tokens, k).
                token_idx_expanded = local_token_indices.view(-1, 1).expand_as(
                    router_indices
                )
                # Mask out invalid token indices (< 0).
                valid = token_idx_expanded >= 0
                if not torch.any(valid):
                    return

                expert_flat = router_indices[valid].to(torch.long)
                token_idx_flat = token_idx_expanded[valid].to(torch.long)

                # Encode (expert, token_index) pairs into a single id and bincount.
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
                # Move encoded inputs to the model device.
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

                # Track which local token index each position maps to using
                # the dense lookup table.
                input_ids = encoded["input_ids"][0]  # (tokens,)
                current_token_indices = id_to_index[input_ids]

                # Update per-token occurrence counts (independent of layer).
                for idx in current_token_indices.tolist():
                    if 0 <= idx < num_token_indices:
                        token_counts[idx] += 1

                _ = self.model(**encoded)

        for h in hooks:
            h.remove()

        return expert_counts_by_token, token_counts


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
        token_id_to_index: dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_layers = self.num_layers
        num_experts = self.num_experts
        num_token_indices = len(token_id_to_index)

        # expert_counts_by_token[layer, expert, token_index] = count
        expert_counts_by_token = torch.zeros(
            num_layers, num_experts, num_token_indices, dtype=torch.long
        )
        # token_counts[token_index] = number of times this token appears
        token_counts = torch.zeros(num_token_indices, dtype=torch.long)

        # Dense lookup table from tokenizer vocab id -> local token index.
        vocab_size = self.tokenizer.vocab_size
        id_to_index = torch.full(
            (vocab_size,),
            -1,
            dtype=torch.long,
            device=self.model.device,
        )
        for tid, idx in token_id_to_index.items():
            if 0 <= tid < vocab_size:
                id_to_index[tid] = int(idx)

        hooks = []
        current_token_indices: torch.Tensor | None = None

        def make_router_hook(layer_idx: int):
            def hook(module, inputs, output):
                nonlocal expert_counts_by_token, current_token_indices

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
                # `current_token_indices` is a 1D tensor of length `tokens`
                # giving the local token_index for each token position.
                if current_token_indices is None:
                    return
                if current_token_indices.numel() != tokens:
                    return

                # Move token indices onto the same device as router_indices.
                local_token_indices = current_token_indices.to(router_indices.device)

                # Vectorized accumulation of counts per (expert, token_index).
                token_idx_expanded = local_token_indices.view(-1, 1).expand_as(
                    router_indices
                )
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
                # Move encoded inputs to the model device.
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

                # Track which local token index each position maps to using
                # the dense lookup table.
                input_ids = encoded["input_ids"][0]  # (tokens,)
                current_token_indices = id_to_index[input_ids]

                # Update per-token occurrence counts (independent of layer).
                for idx in current_token_indices.tolist():
                    if 0 <= idx < num_token_indices:
                        token_counts[idx] += 1

                _ = self.model(**encoded)

        for h in hooks:
            h.remove()

        return expert_counts_by_token, token_counts


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
