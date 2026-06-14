from __future__ import annotations

from types import MethodType

import torch


def _flash_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
):
    if (
        output_attentions
        or attention_mask is not None
        or hidden_states.device.type != "cuda"
        or hidden_states.dtype not in {torch.float16, torch.bfloat16}
    ):
        return self._flash_attn_original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    from flash_attn import flash_attn_func

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

    key_states = self._flash_attn_repeat_kv(key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
    value_states = self._flash_attn_repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
    query_states = query_states.contiguous()

    # Decode has no future positions in the cache, so causal masking is only needed for multi-token prefill.
    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout_p=self.attention_dropout if self.training else 0.0,
        causal=q_len > 1,
    )
    attn_output = attn_output.contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def patch_nemotron_h_attention_flash(model) -> None:
    try:
        import flash_attn  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on GPU environment
        raise RuntimeError("flash_attn is required for the FlashAttention backend") from exc

    module_name = model.__class__.__module__
    remote_module = __import__(module_name, fromlist=["repeat_kv"])
    repeat_kv = remote_module.repeat_kv
    for layer in model.backbone.layers:
        if getattr(layer, "block_type", None) != "attention":
            continue
        attn = getattr(layer, "mixer", None)
        if attn is None or getattr(attn, "_flash_attn_patched", False):
            continue
        attn._flash_attn_original_forward = attn.forward
        attn._flash_attn_repeat_kv = repeat_kv
        attn.forward = MethodType(_flash_forward, attn)
        attn._flash_attn_patched = True
