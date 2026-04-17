import glob
import re

added_block_original = """        # * Added

        moe_manual_weights = self.steermoe_manual_weights.to(
            router_logits.device
        )  # (experts)
        eps = self.eps

        router_logits = torch.nn.functional.log_softmax(
            router_logits, dim=-1
        )  # (T, experts)

        s_max = router_logits.max(dim=-1).values.unsqueeze(-1)  # (T, 1)
        s_min = router_logits.min(dim=-1).values.unsqueeze(-1)  # (T, 1)
        pos_mask = moe_manual_weights > 0  # (T)
        neg_mask = moe_manual_weights < 0  # (T)

        router_logits[:, pos_mask] = s_max + eps
        router_logits[:, neg_mask] = s_min - eps

        # * Added"""

added_block_new = """        # * Added

        router_logits = torch.nn.functional.log_softmax(
            router_logits, dim=-1
        )  # (T, experts)
        
        eps = getattr(self, "eps", 0.0)

        if hasattr(self, "steermoe3d_delta") and self.steermoe3d_delta is not None and (getattr(self, "n_activated", 0) > 0 or getattr(self, "n_deactivated", 0) > 0):
            e = hidden_states @ self.steermoe3d_delta.to(hidden_states.device).to(hidden_states.dtype)

            s_max = router_logits.max(dim=-1).values.unsqueeze(-1)
            s_min = router_logits.min(dim=-1).values.unsqueeze(-1)

            if getattr(self, "n_activated", 0) > 0:
                _, pos_indices = torch.topk(e, self.n_activated, dim=-1)
                pos_mask = torch.zeros_like(e, dtype=torch.bool).scatter_(1, pos_indices, True)
                router_logits[pos_mask] = s_max.expand_as(router_logits)[pos_mask] + eps

            if getattr(self, "n_deactivated", 0) > 0:
                _, neg_indices = torch.topk(e, self.n_deactivated, dim=-1, largest=False)
                neg_mask = torch.zeros_like(e, dtype=torch.bool).scatter_(1, neg_indices, True)
                router_logits[neg_mask] = s_min.expand_as(router_logits)[neg_mask] - eps
        # * Added"""


def patch_modelling():
    for fpath in glob.glob("src/steermoe3d/modelling/*.py"):
        with open(fpath, "r") as f:
            content = f.read()

        # Replace forward block
        content = content.replace(added_block_original, added_block_new)

        # Replace add_steermoe_manual_args -> add_steermoe3d_args
        content = content.replace("def add_steermoe_manual_args(self, manual_weights, eps):", "def add_steermoe3d_args(self, steermoe3d_delta, n_activated, n_deactivated, eps):")
        content = content.replace("layer_moe_block.steermoe_manual_weights = manual_weights[", "layer_moe_block.steermoe3d_delta = steermoe3d_delta[")
        # Add assignment for n_activated and n_deactivated right after eps
        content = re.sub(
            r"(layer_moe_block\.eps = eps\n?)",
            r"\1            layer_moe_block.n_activated = n_activated\n            layer_moe_block.n_deactivated = n_deactivated\n",
            content
        )

        # Fix constructor dummy injection
        # zero_manual_weights = torch.zeros(self.config.num_hidden_layers, self.config.num_experts)
        content = re.sub(
            r"zero_manual_weights = torch\.zeros\([^)]+\)\s*# \(layers, experts\)",
            r"zero_delta = torch.zeros(self.config.num_hidden_layers, self.config.hidden_size, self.config.num_experts)",
            content
        )
        content = content.replace("self.add_steermoe_manual_args(zero_manual_weights, 0)", "self.add_steermoe3d_args(zero_delta, n_activated=0, n_deactivated=0, eps=0)")
        
        # In QWEN they use config.num_hidden_layers? wait. We just rely on regex properly.

        with open(fpath, "w") as f:
            f.write(content)

patch_modelling()
print("Modelling files patched.")
