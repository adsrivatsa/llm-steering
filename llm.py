from __future__ import annotations

from typing import Literal
import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from device import device

ModelName = Literal[
    "allenai/OLMoE-1B-7B-0125-Instruct",
]


def _model_name_safe(model_name: str) -> str:
    return model_name.replace("/", "_")


class OLMoE7B:
    """
    Minimal OLMoE wrapper providing `.infer()` with optional paper-style routing steering.

    Steering matches the paper snippet:
      - compute router_logits = log_softmax(W x)
      - overwrite sparse promoted/suppressed experts to max/min +/- eps
      - pass the resulting distribution through the model's normal top-k routing path
    """

    def __init__(self, model_name: ModelName) -> None:
        if model_name != "allenai/OLMoE-1B-7B-0125-Instruct":
            raise ValueError("OLMoE7B only supports 'allenai/OLMoE-1B-7B-0125-Instruct'")
        self.model_name: ModelName = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        # OLMoE config
        cfg = self.model.config
        self.num_experts = int(getattr(cfg, "num_experts"))
        self.k = int(getattr(cfg, "num_experts_per_tok"))
        self.num_layers = int(getattr(cfg, "num_hidden_layers"))

    def _manual_weights_from_delta(
        self,
        delta_per_expert: torch.Tensor,
        n_activate: int = 0,
        n_deactivate: int = 0,
    ) -> torch.Tensor:
        """
        Convert delta vector into sparse {-1,0,+1} manual weights (paper-style).

        Selects the top `n_activate` experts by most positive delta (promote) and
        top `n_deactivate` by most negative delta (suppress). These are independent
        selections — do not mix them via abs-value ranking, which would force-activate
        experts and degrade fluency (see paper Appendix A.1.3 and Table A.2).

        For faithfulness steering the paper uses n_activate=0, n_deactivate=50.
        """
        delta = delta_per_expert.detach().to(dtype=torch.float32).cpu()
        manual = torch.zeros(self.num_experts, dtype=torch.int8)

        if n_activate > 0:
            k = min(n_activate, self.num_experts)
            top_idx = torch.topk(delta, k=k, dim=0).indices
            manual[top_idx] = 1

        if n_deactivate > 0:
            k = min(n_deactivate, self.num_experts)
            # topk of -delta picks the most negative delta values
            bot_idx = torch.topk(-delta, k=k, dim=0).indices
            manual[bot_idx] = -1

        return manual

    @torch.inference_mode()
    def infer(
        self,
        prompts: list[str],
        *,
        deltas_per_layer: torch.Tensor | None = None,
        n_activate: int = 0,
        n_deactivate: int = 0,
        eps: float = 0.01,
        max_new_tokens: int = 128,
        **generate_kwargs,
    ) -> list[str]:
        """
        Deterministic generation (do_sample=False) with optional steering.
        """
        patched: list[tuple[object, object, bool]] = []

        if deltas_per_layer is not None:
            if deltas_per_layer.shape != (self.num_layers, self.num_experts):
                raise ValueError(
                    f"deltas_per_layer must have shape ({self.num_layers}, {self.num_experts})"
                )

            for layer_idx, layer in enumerate(self.model.model.layers):
                gate = layer.mlp.gate
                manual = self._manual_weights_from_delta(
                    deltas_per_layer[layer_idx],
                    n_activate=n_activate,
                    n_deactivate=n_deactivate,
                ).to(device=next(self.model.parameters()).device)

                old_forward = gate.forward
                old_norm = bool(getattr(gate, "norm_topk_prob", False))
                if hasattr(gate, "norm_topk_prob"):
                    gate.norm_topk_prob = False

                def steered_forward(self_gate, hidden_states, _manual=manual, _eps=eps):
                    # Mirror HF router forward, but intervene in log-softmax space.
                    # _manual and _eps are default args to capture per-layer values at
                    # definition time, avoiding the Python loop closure pitfall where
                    # a free variable always refers to the final loop iteration's value.
                    hidden_dim = self_gate.hidden_dim
                    hs = hidden_states.reshape(-1, hidden_dim)
                    logits = F.linear(hs, self_gate.weight)  # (S, E)
                    s = torch.log_softmax(logits, dim=-1)
                    max_per_tok = s.max(dim=-1, keepdim=True).values
                    min_per_tok = s.min(dim=-1, keepdim=True).values
                    pos = _manual > 0
                    neg = _manual < 0
                    if torch.any(pos):
                        s[:, pos] = max_per_tok + _eps
                    if torch.any(neg):
                        s[:, neg] = min_per_tok - _eps
                    probs = torch.softmax(s, dim=-1, dtype=torch.float32).to(logits.dtype)
                    top_vals, top_idx = torch.topk(probs, k=self_gate.top_k, dim=-1)
                    router_scores = top_vals.to(probs.dtype)
                    return probs, router_scores, top_idx

                gate.forward = types.MethodType(steered_forward, gate)
                patched.append((gate, old_forward, old_norm))

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(next(self.model.parameters()).device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **generate_kwargs,
        )

        prompt_len = int(input_ids.shape[1])
        decoded: list[str] = []
        for seq in out:
            decoded.append(self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))

        # Restore gates
        for gate, old_forward, old_norm in patched:
            gate.forward = old_forward  # type: ignore[assignment]
            if hasattr(gate, "norm_topk_prob"):
                gate.norm_topk_prob = old_norm  # type: ignore[attr-defined]

        return decoded


def get_moe_llm(model_name: ModelName):
    return OLMoE7B(model_name=model_name)

