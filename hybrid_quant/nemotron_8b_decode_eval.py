from __future__ import annotations

from types import MethodType

import torch


def patch_attention_cache_in_blocks(model) -> None:
    """Nemotron-H-8B remote block currently drops cache_params for attention."""
    for block in model.backbone.layers:
        if getattr(block, "_decode_eval_block_patched", False):
            continue

        def forward_with_attention_cache(
            self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
                residual = hidden_states
                hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                if self.block_type == "mamba":
                    hidden_states = self.mixer(
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                    )
                elif self.block_type == "attention":
                    hidden_states = self.mixer(
                        hidden_states,
                        attention_mask=attention_mask,
                        past_key_value=cache_params,
                        use_cache=cache_params is not None,
                        cache_position=cache_position,
                    )[0]
                elif self.block_type == "mlp":
                    hidden_states = self.mixer(hidden_states)
                else:
                    raise ValueError(f"Invalid block_type: {self.block_type}")

                return residual + hidden_states

        block.forward = MethodType(forward_with_attention_cache, block)
        block._decode_eval_block_patched = True


class DeviceList(list):
    def __init__(self, values, device):
        super().__init__(values)
        self.device = device


def make_hybrid_cache(model, batch_size: int):
    module = __import__(model.__class__.__module__, fromlist=["HybridMambaAttentionDynamicCache"])
    cache_cls = getattr(module, "HybridMambaAttentionDynamicCache")
    cache = cache_cls(model.config, batch_size, torch.bfloat16, device=model.device)
    cache.conv_kernel_size = model.config.conv_kernel
    cache.conv_states = DeviceList(cache.conv_states, model.device)
    cache.ssm_states = DeviceList(cache.ssm_states, model.device)
    return cache


def layer_groups(config) -> tuple[list[int], list[int], list[int]]:
    pattern = config.hybrid_override_pattern
    mamba = [i for i, block in enumerate(pattern) if block == "M"]
    attention = [i for i, block in enumerate(pattern) if block == "*"]
    mlp = [i for i, block in enumerate(pattern) if block == "-"]
    return mamba, attention, mlp


def adjacent_mamba_layers(mamba_layers: list[int], attention_layers: list[int]) -> set[int]:
    adjacent: set[int] = set()
    for attn in attention_layers:
        before = [idx for idx in mamba_layers if idx < attn]
        after = [idx for idx in mamba_layers if idx > attn]
        if before:
            adjacent.add(before[-1])
        if after:
            adjacent.add(after[0])
    return adjacent
