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


class MoELLM:
    """
    Generic MoE LLM wrapper that exposes a unified interface for:
      - tokenization
      - model loading
      - collecting expert activation counts via HF hooks

    If you want per-architecture behavior, subclass this and override
    `_get_moe_gate_modules` or related pieces.
    """

    def __init__(self, model_name: ModelName) -> None:
        self.model_name: ModelName = model_name
        self.tokenizer: PreTrainedTokenizerBase = get_tokenizer(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        self.gate_modules = self._get_moe_gate_modules()
        self.num_experts, self.k = self._infer_num_experts_and_k()
        self.num_layers = len(self.gate_modules)

    # ---- MoE-specific helpers -------------------------------------------------

    def _get_moe_gate_modules(self) -> List[Tuple[int, torch.nn.Module]]:
        """
        Heuristically find MoE gate/router linear layers and assign them layer indices.

        This is model-architecture dependent, but for modern MoE models (Mixtral, Qwen3,
        OLMoE, Phi-3.5-MoE) the gate/router is typically a Linear whose name contains
        'gate' or 'router'.
        """
        gate_modules: List[Tuple[int, torch.nn.Module]] = []
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            lname = name.lower()
            if "gate" in lname or "router" in lname:
                gate_modules.append((len(gate_modules), module))

        if not gate_modules:
            raise RuntimeError(
                f"No MoE gate/router modules found for model {self.model_name!r}; "
                "unsupported architecture?"
            )

        return gate_modules

    def _infer_num_experts_and_k(self) -> Tuple[int, int]:
        # Assume all gates share the same out_features.
        first_gate = self.gate_modules[0][1]
        num_experts = int(first_gate.out_features)

        # Prefer paper-derived k / num_experts when available.
        cfg_entry = MOE_CONFIG.get(self.model_name)
        if cfg_entry is not None:
            k_paper, num_experts_paper = cfg_entry
            # If the paper-specified total experts disagrees with the model,
            # trust the model but still use k from the paper.
            k = int(min(k_paper, num_experts))
            return num_experts, k

        # Fallback: try to infer how many experts are active per token from config.
        cfg = self.model.config
        k = (
            getattr(cfg, "num_experts_per_tok", None)
            or getattr(cfg, "num_experts_per_token", None)
            or getattr(cfg, "moe_num_experts_per_token", None)
        )
        if k is None:
            # Last-resort default.
            k = 2

        k = int(min(k, num_experts))
        return num_experts, k

    # ---- Public API for expert activations ------------------------------------

    def collect_expert_activation_counts(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, int]:
        """
        Run the model on the given prompts and collect, for each layer/expert, the
        number of tokens for which that expert was selected by the router.

        Returns:
            expert_counts: tensor of shape [num_layers, num_experts]
            total_tokens: total number of tokens (after tokenization) across prompts
                          for which routing occurred.
        """
        num_layers = self.num_layers
        num_experts = self.num_experts
        k = self.k

        expert_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
        total_tokens = 0

        hooks = []

        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                nonlocal expert_counts, total_tokens

                logits = output  # router logits: [..., num_experts]
                if not isinstance(logits, torch.Tensor):
                    return

                if logits.dim() == 3:
                    # (batch, seq, num_experts) -> (tokens, num_experts)
                    b, s, e = logits.shape
                    logits_flat = logits.reshape(b * s, e)
                elif logits.dim() == 2:
                    # (tokens, num_experts)
                    logits_flat = logits
                else:
                    return

                probs = torch.softmax(logits_flat, dim=-1)
                topk = probs.topk(k=k, dim=-1).indices  # (tokens, k)

                # Count how many times each expert appears in the top-k for this layer.
                selected_experts = topk.reshape(-1)  # (tokens * k,)
                counts = torch.bincount(
                    selected_experts,
                    minlength=num_experts,
                )  # (num_experts,)
                expert_counts[layer_idx] += counts.to(expert_counts.device)

                total_tokens += logits_flat.size(0)

            return hook

        # Register hooks on each gate module.
        for layer_idx, module in self.gate_modules:
            hooks.append(module.register_forward_hook(make_hook(layer_idx)))

        with torch.inference_mode():
            for prompt in tqdm(
                prompts, desc=f"Running MoE inference ({self.model_name})"
            ):
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                # With device_map="auto", inputs can stay on CPU and Accelerate
                # will handle moving them to the right shards.
                _ = self.model(**encoded)

        # Clean up hooks.
        for h in hooks:
            h.remove()

        return expert_counts, total_tokens


class GPT20MoELLM(MoELLM):
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
                # (logits, indices, scores). We know from inspection that
                # the second tensor (output[1]) contains the top-k expert
                # indices with shape (tokens, k).
                if not (isinstance(output, tuple) and len(output) >= 2):
                    return

                router_indices = output[1]
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
                prompts, desc="Running MoE inference (openai/gpt-oss-20b)"
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


def get_moe_llm(model_name: ModelName) -> MoELLM:
    """
    Factory to obtain a MoELLM wrapper for the given model.
    """
    if model_name == "openai/gpt-oss-20b":
        return GPT20MoELLM(model_name=model_name)

    # Fallback: generic implementation (may not work for all architectures).
    return MoELLM(model_name=model_name)
