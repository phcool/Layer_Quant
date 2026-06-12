from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def int4_quant_dequant(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"head_dim={x.shape[-1]} must be divisible by group_size={group_size}")
    orig_shape = x.shape
    dtype = x.dtype
    grouped = x.float().reshape(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)
    scale = torch.clamp(grouped.abs().amax(dim=-1, keepdim=True) / 7.0, min=1.0e-8)
    q = torch.clamp(torch.round(grouped / scale), -8.0, 7.0)
    return (q * scale).reshape(orig_shape).to(dtype)


def _forward_sdpa_int4_kv(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
):
    if output_attentions:
        return self._int4_kv_original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    key_states = int4_quant_dequant(key_states, group_size=self._int4_kv_group_size)
    value_states = int4_quant_dequant(value_states, group_size=self._int4_kv_group_size)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

    repeat_kv = self._int4_kv_repeat_kv
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if self.is_causal and causal_mask is None and q_len > 1 else False
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def patch_nemotron_h_attention_int4_kv(model, group_size: int = 64):
    module_name = model.__class__.__module__
    remote_module = __import__(module_name, fromlist=["repeat_kv"])
    repeat_kv = remote_module.repeat_kv
    for layer in model.backbone.layers:
        if getattr(layer, "block_type", None) != "attention":
            continue
        attn = getattr(layer, "mixer", None)
        if attn is None or getattr(attn, "_int4_kv_patched", False):
            continue
        attn._int4_kv_original_forward = attn.forward
        attn._int4_kv_group_size = group_size
        attn._int4_kv_repeat_kv = repeat_kv
        attn.forward = MethodType(_forward_sdpa_int4_kv, attn)
        attn._int4_kv_patched = True
