from typing import Literal, Tuple
import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.device import device

ModelName = Literal[
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    "microsoft/Phi-3.5-MoE-instruct",
]

# MoE configuration taken from the paper table:
#   Active / Total experts per layer.
MOE_CONFIG: dict[ModelName, Tuple[int, int]] = {
    "openai/gpt-oss-20b": (4, 32),
    "openai/gpt-oss-120b": (4, 128),
    "Qwen/Qwen3-30B-A3B": (8, 128),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (2, 8),
    "allenai/OLMoE-1B-7B-0125-Instruct": (8, 64),
    "microsoft/Phi-3.5-MoE-instruct": (2, 16),
}


def _global_manual_weights(
    deltas_per_layer: torch.Tensor,
    n_activate: int,
    n_deactivate: int,
    num_experts_per_tok: int = 0,
) -> torch.Tensor:
    """
    Rank all (layer, expert) pairs globally and return a (num_layers, num_experts)
    int8 tensor of {-1, 0, +1} manual weights.

    Matches the paper's steer_moe logic exactly:
    - Positive experts are promoted (+1) in descending delta order, with a
      per-layer cap of `num_experts_per_tok` (the model's k).  Any candidate
      that would exceed that cap for its layer is skipped (not promoted).
    - Negative experts are suppressed (-1) with no per-layer cap.
    """
    delta = deltas_per_layer.detach().to(dtype=torch.float32).cpu()
    num_layers, num_experts = delta.shape
    manual = torch.zeros(num_layers, num_experts, dtype=torch.int8)

    if n_activate > 0:
        flat = delta.flatten()
        k = min(n_activate, flat.numel())
        layer_pos_counts = torch.zeros(num_layers, dtype=torch.int32)
        for idx in torch.topk(flat, k=k).indices:
            layer = int(idx) // num_experts
            expert = int(idx) % num_experts
            if num_experts_per_tok > 0 and layer_pos_counts[layer] >= num_experts_per_tok:
                continue
            manual[layer, expert] = 1
            layer_pos_counts[layer] += 1

    if n_deactivate > 0:
        flat = delta.flatten()
        k = min(n_deactivate, flat.numel())
        for idx in torch.topk(-flat, k=k).indices:
            manual[int(idx) // num_experts, int(idx) % num_experts] = -1

    return manual


@torch.inference_mode()
def _infer(
    model,
    tokenizer,
    layers,
    prompts: list[str],
    *,
    num_experts: int,
    num_experts_per_tok: int = 0,
    get_router,
    make_steered_forward,
    patch_router,
    restore_router,
    deltas_per_layer: torch.Tensor | None = None,
    n_activate: int = 0,
    n_deactivate: int = 0,
    eps: float = 0.01,
    max_new_tokens: int = 128,
    chat_template_kwargs: dict | None = None,
    **generate_kwargs,
) -> list[str]:
    """
    Shared inference implementation for all MoE wrappers.

    Model-specific behaviour is injected via four callables:
      get_router(layer)              -> router module for that layer
      make_steered_forward(manual, eps) -> forward function to patch onto the router
      patch_router(router, fwd)      -> applies the patch; returns an opaque restore tuple
      restore_router(*restore_tuple) -> undoes the patch

    num_experts_per_tok: the model's k (active experts per token).  Used to cap
        promoted experts per layer, matching the paper's steer_moe behaviour.

    chat_template_kwargs: if provided, each prompt is wrapped as a user message and
        formatted through the tokenizer's chat template before tokenisation.
    """
    patched = []

    if deltas_per_layer is not None:
        num_layers = len(layers)
        if deltas_per_layer.shape != (num_layers, num_experts):
            raise ValueError(
                f"deltas_per_layer must have shape ({num_layers}, {num_experts})"
            )
        model_device = next(model.parameters()).device
        manual_all = _global_manual_weights(
            deltas_per_layer,
            n_activate=n_activate,
            n_deactivate=n_deactivate,
            num_experts_per_tok=num_experts_per_tok,
        ).to(device=model_device)

        for layer_idx, layer in enumerate(layers):
            router = get_router(layer)
            fwd = make_steered_forward(manual_all[layer_idx], eps)
            patched.append(patch_router(router, fwd))

    if chat_template_kwargs is not None:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            for p in prompts
        ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=chat_template_kwargs is None,
    )
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        **generate_kwargs,
    )

    prompt_len = int(input_ids.shape[1])
    decoded = [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out
    ]

    for restore_info in patched:
        restore_router(*restore_info)

    return decoded


class OLMoE7B:
    """
    Minimal OLMoE wrapper providing `.infer()` with optional paper-style routing steering.
    """

    def __init__(self) -> None:
        self.model_name: ModelName = "allenai/OLMoE-1B-7B-0125-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        k_paper, num_experts_paper = MOE_CONFIG[self.model_name]
        self.num_experts = int(num_experts_paper)
        self.k = int(k_paper)

    def infer(
        self,
        prompts: list[str],
        *,
        deltas_per_layer: torch.Tensor | None = None,
        n_activate: int = 0,
        n_deactivate: int = 0,
        eps: float = 0.01,
        max_new_tokens: int = 128,
        chat_template_kwargs: dict | None = None,
        **generate_kwargs,
    ) -> list[str]:
        def get_router(layer):
            return layer.mlp.gate

        def make_steered_forward(manual, eps):
            def steered_forward(self_gate, hidden_states, _manual=manual, _eps=eps):
                hs = hidden_states.reshape(-1, self_gate.hidden_dim)
                logits = F.linear(hs, self_gate.weight)
                _manual = _manual.to(device=logits.device)
                s = torch.log_softmax(logits, dim=-1)
                s_max = s.max(dim=-1, keepdim=True).values
                s_min = s.min(dim=-1, keepdim=True).values
                s[:, _manual > 0] = s_max + _eps
                s[:, _manual < 0] = s_min - _eps
                probs = torch.softmax(s, dim=-1, dtype=torch.float32).to(logits.dtype)
                top_vals, top_idx = torch.topk(probs, k=self_gate.top_k, dim=-1)
                return probs, top_vals.to(probs.dtype), top_idx

            return steered_forward

        def patch_router(router, fwd):
            old_forward = router.forward
            old_norm = bool(getattr(router, "norm_topk_prob", False))
            if hasattr(router, "norm_topk_prob"):
                router.norm_topk_prob = False
            router.forward = types.MethodType(fwd, router)
            return (router, old_forward, old_norm)

        def restore_router(router, old_forward, old_norm):
            router.forward = old_forward
            if hasattr(router, "norm_topk_prob"):
                router.norm_topk_prob = old_norm

        return _infer(
            self.model,
            self.tokenizer,
            self.model.model.layers,
            prompts,
            num_experts=self.num_experts,
            num_experts_per_tok=self.k,
            get_router=get_router,
            make_steered_forward=make_steered_forward,
            patch_router=patch_router,
            restore_router=restore_router,
            deltas_per_layer=deltas_per_layer,
            n_activate=n_activate,
            n_deactivate=n_deactivate,
            eps=eps,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
            **generate_kwargs,
        )


class GPT20B:
    """
    Minimal GPT-OSS-20B wrapper providing `.infer()` with optional paper-style routing steering.

    GPT-OSS exposes routers via `layer.mlp.router`, returning
    (router_logits, router_weights, router_indices). Steering intervenes in log-softmax
    space before top-k, then applies a local softmax over the selected k experts to
    match the normalisation that GptOssExperts expects.

    Architecture: 4 active experts out of 32 total per layer.
    Model is loaded with dtype="auto" since MXFP4 quantization dequantizes to bf16.
    """

    def __init__(self) -> None:
        self.model_name: ModelName = "openai/gpt-oss-20b"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype="auto", device_map="auto"
        )
        self.model.eval()

        self.num_experts = 32
        self.k = 4

    def infer(
        self,
        prompts: list[str],
        *,
        deltas_per_layer: torch.Tensor | None = None,
        n_activate: int = 0,
        n_deactivate: int = 0,
        eps: float = 0.01,
        max_new_tokens: int = 128,
        chat_template_kwargs: dict | None = None,
        **generate_kwargs,
    ) -> list[str]:
        _k = self.k

        def get_router(layer):
            return layer.mlp.router

        def make_steered_forward(manual, eps):
            def steered_forward(
                self_router, hidden_states, _manual=manual, _eps=eps, _k=_k
            ):
                router_logits = F.linear(
                    hidden_states, self_router.weight, self_router.bias
                )
                _manual = _manual.to(device=router_logits.device)
                s = torch.log_softmax(router_logits.to(torch.float32), dim=-1)
                s_max = s.max(dim=-1, keepdim=True).values
                s_min = s.min(dim=-1, keepdim=True).values
                s[:, _manual > 0] = s_max + _eps
                s[:, _manual < 0] = s_min - _eps
                top_s_vals, top_idx = torch.topk(s, k=_k, dim=-1)
                router_scores = torch.softmax(
                    top_s_vals, dim=1, dtype=top_s_vals.dtype
                ).to(router_logits.dtype)
                return s.to(router_logits.dtype), router_scores, top_idx

            return steered_forward

        def patch_router(router, fwd):
            old_forward = router.forward
            router.forward = types.MethodType(fwd, router)
            return (router, old_forward)

        def restore_router(router, old_forward):
            router.forward = old_forward

        return _infer(
            self.model,
            self.tokenizer,
            self.model.model.layers,
            prompts,
            num_experts=self.num_experts,
            num_experts_per_tok=self.k,
            get_router=get_router,
            make_steered_forward=make_steered_forward,
            patch_router=patch_router,
            restore_router=restore_router,
            deltas_per_layer=deltas_per_layer,
            n_activate=n_activate,
            n_deactivate=n_deactivate,
            eps=eps,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
            **generate_kwargs,
        )


class Qwen30B:
    """
    Minimal Qwen3-30B-A3B wrapper providing `.infer()` with optional paper-style routing steering.

    Architecture: 8 active experts out of 128 total per layer.
    Router is at `layer.mlp.gate` and uses the same OLMoE-style routing
    (global softmax over all experts → top-k selection).
    """

    def __init__(self) -> None:
        self.model_name: ModelName = "Qwen/Qwen3-30B-A3B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        k_paper, num_experts_paper = MOE_CONFIG[self.model_name]
        self.num_experts = int(num_experts_paper)
        self.k = int(k_paper)

    def infer(
        self,
        prompts: list[str],
        *,
        deltas_per_layer: torch.Tensor | None = None,
        n_activate: int = 0,
        n_deactivate: int = 0,
        eps: float = 0.01,
        max_new_tokens: int = 128,
        chat_template_kwargs: dict | None = None,
        **generate_kwargs,
    ) -> list[str]:
        def get_router(layer):
            return layer.mlp.gate

        def make_steered_forward(manual, eps):
            def steered_forward(self_gate, hidden_states, _manual=manual, _eps=eps):
                hs = hidden_states.reshape(-1, self_gate.hidden_dim)
                logits = F.linear(hs, self_gate.weight)
                _manual = _manual.to(device=logits.device)
                s = torch.log_softmax(logits, dim=-1)
                s_max = s.max(dim=-1, keepdim=True).values
                s_min = s.min(dim=-1, keepdim=True).values
                s[:, _manual > 0] = s_max + _eps
                s[:, _manual < 0] = s_min - _eps
                probs = torch.softmax(s, dim=-1, dtype=torch.float32).to(logits.dtype)
                top_vals, top_idx = torch.topk(probs, k=self_gate.top_k, dim=-1)
                return probs, top_vals.to(probs.dtype), top_idx

            return steered_forward

        def patch_router(router, fwd):
            old_forward = router.forward
            old_norm = bool(getattr(router, "norm_topk_prob", False))
            if hasattr(router, "norm_topk_prob"):
                router.norm_topk_prob = False
            router.forward = types.MethodType(fwd, router)
            return (router, old_forward, old_norm)

        def restore_router(router, old_forward, old_norm):
            router.forward = old_forward
            if hasattr(router, "norm_topk_prob"):
                router.norm_topk_prob = old_norm

        return _infer(
            self.model,
            self.tokenizer,
            self.model.model.layers,
            prompts,
            num_experts=self.num_experts,
            num_experts_per_tok=self.k,
            get_router=get_router,
            make_steered_forward=make_steered_forward,
            patch_router=patch_router,
            restore_router=restore_router,
            deltas_per_layer=deltas_per_layer,
            n_activate=n_activate,
            n_deactivate=n_deactivate,
            eps=eps,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
            **generate_kwargs,
        )


class Mixtral8x7B:
    """
    Minimal Mixtral-8x7B-Instruct-v0.1 wrapper providing `.infer()` with optional
    paper-style routing steering.

    Architecture: 2 active experts out of 8 total per layer.
    Router is `layer.block_sparse_moe.gate` (plain nn.Linear, no bias). The parent
    block applies softmax + top-k to its output, so the steered gate returns modified
    log-softmax values as pseudo-logits: softmax(s_steered) correctly promotes and
    suppresses the target experts without changing how the block selects top-k.
    """

    def __init__(self) -> None:
        self.model_name: ModelName = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        k_paper, num_experts_paper = MOE_CONFIG[self.model_name]
        self.num_experts = int(num_experts_paper)
        self.k = int(k_paper)

    def infer(
        self,
        prompts: list[str],
        *,
        deltas_per_layer: torch.Tensor | None = None,
        n_activate: int = 0,
        n_deactivate: int = 0,
        eps: float = 0.01,
        max_new_tokens: int = 128,
        chat_template_kwargs: dict | None = None,
        **generate_kwargs,
    ) -> list[str]:
        def get_router(layer):
            return layer.mlp.gate

        _k = self.k

        def make_steered_forward(manual, eps):
            def steered_forward(
                self_gate, hidden_states, _manual=manual, _eps=eps, _k=_k
            ):
                hidden_states = hidden_states.reshape(-1, self_gate.hidden_dim)
                logits = F.linear(hidden_states, self_gate.weight)
                _manual = _manual.to(device=logits.device)
                s = torch.log_softmax(logits.float(), dim=-1)
                s_max = s.max(dim=-1, keepdim=True).values
                s_min = s.min(dim=-1, keepdim=True).values
                s[:, _manual > 0] = s_max + _eps
                s[:, _manual < 0] = s_min - _eps
                top_s_vals, top_idx = torch.topk(s, k=_k, dim=-1)
                router_scores = torch.softmax(top_s_vals, dim=-1).to(logits.dtype)
                return s.to(logits.dtype), router_scores, top_idx

            return steered_forward

        def patch_router(router, fwd):
            old_forward = router.forward
            router.forward = types.MethodType(fwd, router)
            return (router, old_forward)

        def restore_router(router, old_forward):
            router.forward = old_forward

        return _infer(
            self.model,
            self.tokenizer,
            self.model.model.layers,
            prompts,
            num_experts=self.num_experts,
            num_experts_per_tok=self.k,
            get_router=get_router,
            make_steered_forward=make_steered_forward,
            patch_router=patch_router,
            restore_router=restore_router,
            deltas_per_layer=deltas_per_layer,
            n_activate=n_activate,
            n_deactivate=n_deactivate,
            eps=eps,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
            **generate_kwargs,
        )


class Phi42B:
    """
    Minimal Phi-3.5-MoE-instruct wrapper providing `.infer()` with optional
    paper-style routing steering.

    Architecture: 2 active experts out of 16 total per layer.
    Router is at `layer.mlp.router`, returning (router_logits, router_weights,
    router_indices). Steering follows the same log-softmax → top-k → local-softmax
    pattern as GPT20B.
    """

    def __init__(self) -> None:
        self.model_name: ModelName = "microsoft/Phi-3.5-MoE-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        k_paper, num_experts_paper = MOE_CONFIG[self.model_name]
        self.num_experts = int(num_experts_paper)
        self.k = int(k_paper)

    def infer(
        self,
        prompts: list[str],
        *,
        deltas_per_layer: torch.Tensor | None = None,
        n_activate: int = 0,
        n_deactivate: int = 0,
        eps: float = 0.01,
        max_new_tokens: int = 128,
        chat_template_kwargs: dict | None = None,
        **generate_kwargs,
    ) -> list[str]:
        def get_router(layer):
            return layer.mlp.router

        def make_steered_forward(manual, eps):
            def steered_forward(self_router, hidden_states, _manual=manual, _eps=eps):
                from transformers.models.phimoe.modeling_phimoe import sparsemixer

                logits = F.linear(hidden_states, self_router.weight)
                _manual = _manual.to(device=logits.device)
                s = torch.log_softmax(logits.to(torch.float32), dim=-1)
                s_max = s.max(dim=-1, keepdim=True).values
                s_min = s.min(dim=-1, keepdim=True).values
                s[:, _manual > 0] = s_max + _eps
                s[:, _manual < 0] = s_min - _eps
                steered = s.to(logits.dtype)

                routing_weights, selected_experts = sparsemixer(
                    steered,
                    jitter_eps=self_router.router_jitter_noise,
                    training=self_router.training,
                    top_k=self_router.top_k,
                )
                return steered, routing_weights, selected_experts

            return steered_forward

        def patch_router(router, fwd):
            old_forward = router.forward
            router.forward = types.MethodType(fwd, router)
            return (router, old_forward)

        def restore_router(router, old_forward):
            router.forward = old_forward

        return _infer(
            self.model,
            self.tokenizer,
            self.model.model.layers,
            prompts,
            num_experts=self.num_experts,
            num_experts_per_tok=self.k,
            get_router=get_router,
            make_steered_forward=make_steered_forward,
            patch_router=patch_router,
            restore_router=restore_router,
            deltas_per_layer=deltas_per_layer,
            n_activate=n_activate,
            n_deactivate=n_deactivate,
            eps=eps,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
            **generate_kwargs,
        )


def get_moe_llm(model_name: ModelName):
    if model_name == "openai/gpt-oss-20b":
        return GPT20B()
    if model_name == "allenai/OLMoE-1B-7B-0125-Instruct":
        return OLMoE7B()
    if model_name == "Qwen/Qwen3-30B-A3B":
        return Qwen30B()
    if model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        return Mixtral8x7B()
    if model_name == "microsoft/Phi-3.5-MoE-instruct":
        return Phi42B()
    raise NotImplementedError(f"MoE wrapper not implemented for {model_name!r}")
