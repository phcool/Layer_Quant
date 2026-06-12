from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType

import torch
import triton
import triton.language as tl


@dataclass
class Int4KVLayerCache:
    k_packed: torch.Tensor
    v_packed: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    length: int = 0


def int4_quant_dequant(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"head_dim={x.shape[-1]} must be divisible by group_size={group_size}")
    orig_shape = x.shape
    dtype = x.dtype
    grouped = x.float().reshape(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)
    scale = torch.clamp(grouped.abs().amax(dim=-1, keepdim=True) / 7.0, min=1.0e-8)
    q = torch.clamp(torch.round(grouped / scale), -8.0, 7.0)
    return (q * scale).reshape(orig_shape).to(dtype)


@triton.jit
def _pack_int4_kv_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    positions_ptr,
    x_stride_b,
    x_stride_h,
    x_stride_t,
    x_stride_d,
    packed_stride_b,
    packed_stride_h,
    packed_stride_t,
    packed_stride_p,
    scale_stride_b,
    scale_stride_h,
    scale_stride_t,
    scale_stride_g,
    n_tokens: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
):
    bh = tl.program_id(0)
    token = tl.program_id(1)
    group = tl.program_id(2)

    batch = bh // n_heads
    head = bh - batch * n_heads
    pos = tl.load(positions_ptr + token)
    d_start = group * group_size
    d_offsets = tl.arange(0, BLOCK_GROUP)
    d_mask = d_offsets < group_size
    x = tl.load(
        x_ptr
        + batch * x_stride_b
        + head * x_stride_h
        + token * x_stride_t
        + (d_start + d_offsets) * x_stride_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(absmax / 7.0, 1.0e-8)
    q = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [x / scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q = tl.minimum(tl.maximum(q, -8), 7)
    q_nibble = q & 15

    pair_offsets = tl.arange(0, BLOCK_GROUP // 2)
    low = tl.sum(
        tl.where(d_offsets[:, None] == pair_offsets[None, :] * 2, q_nibble[:, None], 0),
        axis=0,
    )
    high = tl.sum(
        tl.where(d_offsets[:, None] == pair_offsets[None, :] * 2 + 1, q_nibble[:, None], 0),
        axis=0,
    )
    packed = (high << 4) | low

    tl.store(
        packed_ptr
        + batch * packed_stride_b
        + head * packed_stride_h
        + pos * packed_stride_t
        + (d_start // 2 + pair_offsets) * packed_stride_p,
        packed,
        mask=pair_offsets < (group_size // 2),
    )
    tl.store(
        scale_ptr + batch * scale_stride_b + head * scale_stride_h + pos * scale_stride_t + group * scale_stride_g,
        scale,
    )


@triton.jit
def _int4_prefill_attention_kernel(
    q_ptr,
    k_packed_ptr,
    v_packed_ptr,
    k_scale_ptr,
    v_scale_ptr,
    out_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_t,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_t,
    k_stride_p,
    v_stride_b,
    v_stride_h,
    v_stride_t,
    v_stride_p,
    ks_stride_b,
    ks_stride_h,
    ks_stride_t,
    ks_stride_g,
    vs_stride_b,
    vs_stride_h,
    vs_stride_t,
    vs_stride_g,
    out_stride_b,
    out_stride_h,
    out_stride_t,
    out_stride_d,
    n_query_heads: tl.constexpr,
    q_len: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0)
    q_pos = program % q_len
    bh = program // q_len
    batch = bh // n_query_heads
    q_head = bh - batch * n_query_heads
    kv_head = q_head // num_key_value_groups
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    scale_sm = 1.4426950408889634 / tl.sqrt(tl.full((), head_dim, tl.float32))

    n_offsets = tl.arange(0, BLOCK_N)
    causal_len = q_pos + 1
    start = 0
    while start < causal_len:
        n = start + n_offsets
        n_mask = n < causal_len
        scores = tl.zeros((BLOCK_N,), tl.float32)
        for d_start in tl.static_range(0, BLOCK_D, group_size):
            gd = d_start + tl.arange(0, group_size)
            gd_mask = gd < head_dim
            q_group = tl.load(
                q_ptr + batch * q_stride_b + q_head * q_stride_h + q_pos * q_stride_t + gd * q_stride_d,
                mask=gd_mask,
                other=0.0,
            ).to(tl.float32)
            pair = gd // 2
            byte = tl.load(
                k_packed_ptr
                + batch * k_stride_b
                + kv_head * k_stride_h
                + n[:, None] * k_stride_t
                + pair[None, :] * k_stride_p,
                mask=n_mask[:, None] & gd_mask[None, :],
                other=0,
            ).to(tl.int32)
            nibble = tl.where((gd[None, :] & 1) == 0, byte & 15, (byte >> 4) & 15)
            signed = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
            scale = tl.load(
                k_scale_ptr
                + batch * ks_stride_b
                + kv_head * ks_stride_h
                + n * ks_stride_t
                + (d_start // group_size) * ks_stride_g,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            k = signed * scale[:, None]
            scores += tl.sum(k * q_group[None, :], axis=1)

        scores = tl.where(n_mask, scores * scale_sm, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc *= alpha

        for d_start in tl.static_range(0, BLOCK_D, group_size):
            gd = d_start + tl.arange(0, group_size)
            gd_mask = gd < head_dim
            pair = gd // 2
            byte = tl.load(
                v_packed_ptr
                + batch * v_stride_b
                + kv_head * v_stride_h
                + n[:, None] * v_stride_t
                + pair[None, :] * v_stride_p,
                mask=n_mask[:, None] & gd_mask[None, :],
                other=0,
            ).to(tl.int32)
            nibble = tl.where((gd[None, :] & 1) == 0, byte & 15, (byte >> 4) & 15)
            signed = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
            scale = tl.load(
                v_scale_ptr
                + batch * vs_stride_b
                + kv_head * vs_stride_h
                + n * vs_stride_t
                + (d_start // group_size) * vs_stride_g,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            v = signed * scale[:, None]
            acc += tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new
        start += BLOCK_N

    out = acc / l_i
    tl.store(
        out_ptr + batch * out_stride_b + q_head * out_stride_h + q_pos * out_stride_t + d_offsets * out_stride_d,
        out,
        mask=d_mask,
    )


@triton.jit
def _int4_decode_attention_kernel(
    q_ptr,
    k_packed_ptr,
    v_packed_ptr,
    k_scale_ptr,
    v_scale_ptr,
    out_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_t,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_t,
    k_stride_p,
    v_stride_b,
    v_stride_h,
    v_stride_t,
    v_stride_p,
    ks_stride_b,
    ks_stride_h,
    ks_stride_t,
    ks_stride_g,
    vs_stride_b,
    vs_stride_h,
    vs_stride_t,
    vs_stride_g,
    out_stride_b,
    out_stride_h,
    out_stride_d,
    seq_len,
    n_query_heads: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    batch = bh // n_query_heads
    q_head = bh - batch * n_query_heads
    kv_head = q_head // num_key_value_groups
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    scale_sm = 1.4426950408889634 / tl.sqrt(tl.full((), head_dim, tl.float32))

    n_offsets = tl.arange(0, BLOCK_N)
    start = 0
    while start < seq_len:
        n = start + n_offsets
        n_mask = n < seq_len
        scores = tl.zeros((BLOCK_N,), tl.float32)
        for d_start in tl.static_range(0, BLOCK_D, group_size):
            gd = d_start + tl.arange(0, group_size)
            gd_mask = gd < head_dim
            q_group = tl.load(
                q_ptr + batch * q_stride_b + q_head * q_stride_h + gd * q_stride_d,
                mask=gd_mask,
                other=0.0,
            ).to(tl.float32)
            pair = gd // 2
            byte = tl.load(
                k_packed_ptr
                + batch * k_stride_b
                + kv_head * k_stride_h
                + n[:, None] * k_stride_t
                + pair[None, :] * k_stride_p,
                mask=n_mask[:, None] & gd_mask[None, :],
                other=0,
            ).to(tl.int32)
            nibble = tl.where((gd[None, :] & 1) == 0, byte & 15, (byte >> 4) & 15)
            signed = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
            scale = tl.load(
                k_scale_ptr
                + batch * ks_stride_b
                + kv_head * ks_stride_h
                + n * ks_stride_t
                + (d_start // group_size) * ks_stride_g,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            k = signed * scale[:, None]
            scores += tl.sum(k * q_group[None, :], axis=1)

        scores = tl.where(n_mask, scores * scale_sm, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc *= alpha

        for d_start in tl.static_range(0, BLOCK_D, group_size):
            gd = d_start + tl.arange(0, group_size)
            gd_mask = gd < head_dim
            pair = gd // 2
            byte = tl.load(
                v_packed_ptr
                + batch * v_stride_b
                + kv_head * v_stride_h
                + n[:, None] * v_stride_t
                + pair[None, :] * v_stride_p,
                mask=n_mask[:, None] & gd_mask[None, :],
                other=0,
            ).to(tl.int32)
            nibble = tl.where((gd[None, :] & 1) == 0, byte & 15, (byte >> 4) & 15)
            signed = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
            scale = tl.load(
                v_scale_ptr
                + batch * vs_stride_b
                + kv_head * vs_stride_h
                + n * vs_stride_t
                + (d_start // group_size) * vs_stride_g,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            v = signed * scale[:, None]
            acc += tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new
        start += BLOCK_N

    out = acc / l_i
    tl.store(
        out_ptr + batch * out_stride_b + q_head * out_stride_h + d_offsets * out_stride_d,
        out,
        mask=d_mask,
    )


def _cache_dict(past_key_value) -> dict[int, Int4KVLayerCache]:
    caches = getattr(past_key_value, "_int4_kv_layer_caches", None)
    if caches is None:
        caches = {}
        setattr(past_key_value, "_int4_kv_layer_caches", caches)
    return caches


def _positions(cache_position, q_len: int, device: torch.device) -> torch.Tensor:
    if cache_position is None:
        return torch.arange(q_len, device=device, dtype=torch.long)
    if cache_position.numel() == q_len:
        return cache_position.to(device=device, dtype=torch.long)
    if cache_position.numel() == 1 and q_len == 1:
        return cache_position.to(device=device, dtype=torch.long)
    raise ValueError(f"cache_position has {cache_position.numel()} entries for q_len={q_len}")


def _allocate_layer_cache(
    batch: int,
    kv_heads: int,
    capacity: int,
    head_dim: int,
    groups: int,
    device: torch.device,
) -> Int4KVLayerCache:
    return Int4KVLayerCache(
        k_packed=torch.empty((batch, kv_heads, capacity, head_dim // 2), device=device, dtype=torch.uint8),
        v_packed=torch.empty((batch, kv_heads, capacity, head_dim // 2), device=device, dtype=torch.uint8),
        k_scale=torch.empty((batch, kv_heads, capacity, groups), device=device, dtype=torch.float32),
        v_scale=torch.empty((batch, kv_heads, capacity, groups), device=device, dtype=torch.float32),
    )


def _ensure_layer_cache(
    past_key_value,
    layer_idx: int,
    batch: int,
    kv_heads: int,
    required_len: int,
    head_dim: int,
    group_size: int,
    device: torch.device,
) -> Int4KVLayerCache:
    groups = head_dim // group_size
    caches = _cache_dict(past_key_value)
    current = caches.get(layer_idx)
    if current is None:
        capacity = max(16, 1 << math.ceil(math.log2(max(required_len, 1))))
        current = _allocate_layer_cache(batch, kv_heads, capacity, head_dim, groups, device)
        caches[layer_idx] = current
        return current
    if current.k_packed.shape[:2] != (batch, kv_heads):
        raise ValueError(
            f"Layer {layer_idx} INT4 cache shape {tuple(current.k_packed.shape[:2])} "
            f"does not match batch/kv_heads {(batch, kv_heads)}"
        )
    if required_len <= current.k_packed.shape[2]:
        return current

    new_capacity = 1 << math.ceil(math.log2(required_len))
    grown = _allocate_layer_cache(batch, kv_heads, new_capacity, head_dim, groups, device)
    old_len = current.k_packed.shape[2]
    grown.k_packed[:, :, :old_len].copy_(current.k_packed)
    grown.v_packed[:, :, :old_len].copy_(current.v_packed)
    grown.k_scale[:, :, :old_len].copy_(current.k_scale)
    grown.v_scale[:, :, :old_len].copy_(current.v_scale)
    grown.length = current.length
    caches[layer_idx] = grown
    return grown


def _pack_kv_into_cache(
    cache: Int4KVLayerCache,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    positions: torch.Tensor,
    group_size: int,
) -> None:
    batch, kv_heads, q_len, head_dim = key_states.shape
    groups = head_dim // group_size
    grid = (batch * kv_heads, q_len, groups)
    _pack_int4_kv_kernel[grid](
        key_states,
        cache.k_packed,
        cache.k_scale,
        positions,
        *key_states.stride(),
        *cache.k_packed.stride(),
        *cache.k_scale.stride(),
        q_len,
        kv_heads,
        head_dim,
        group_size,
        BLOCK_GROUP=triton.next_power_of_2(group_size),
    )
    _pack_int4_kv_kernel[grid](
        value_states,
        cache.v_packed,
        cache.v_scale,
        positions,
        *value_states.stride(),
        *cache.v_packed.stride(),
        *cache.v_scale.stride(),
        q_len,
        kv_heads,
        head_dim,
        group_size,
        BLOCK_GROUP=triton.next_power_of_2(group_size),
    )
    cache.length = max(cache.length, int(positions.max().item()) + 1)


def _decode_attention(
    query_states: torch.Tensor,
    cache: Int4KVLayerCache,
    seq_len: int,
    num_key_value_groups: int,
    group_size: int,
) -> torch.Tensor:
    batch, n_heads, q_len, head_dim = query_states.shape
    if q_len != 1:
        raise ValueError(f"Fused INT4 decode attention expects q_len=1, got {q_len}")
    out = torch.empty((batch, n_heads, head_dim), device=query_states.device, dtype=query_states.dtype)
    block_d = triton.next_power_of_2(head_dim)
    grid = (batch * n_heads,)
    _int4_decode_attention_kernel[grid](
        query_states,
        cache.k_packed,
        cache.v_packed,
        cache.k_scale,
        cache.v_scale,
        out,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        *cache.k_packed.stride(),
        *cache.v_packed.stride(),
        *cache.k_scale.stride(),
        *cache.v_scale.stride(),
        *out.stride(),
        seq_len,
        n_heads,
        num_key_value_groups,
        head_dim,
        group_size,
        BLOCK_N=64,
        BLOCK_D=block_d,
    )
    return out


def _prefill_attention(
    query_states: torch.Tensor,
    cache: Int4KVLayerCache,
    num_key_value_groups: int,
    group_size: int,
) -> torch.Tensor:
    batch, n_heads, q_len, head_dim = query_states.shape
    out = torch.empty((batch, n_heads, q_len, head_dim), device=query_states.device, dtype=query_states.dtype)
    block_d = triton.next_power_of_2(head_dim)
    grid = (batch * n_heads * q_len,)
    _int4_prefill_attention_kernel[grid](
        query_states,
        cache.k_packed,
        cache.v_packed,
        cache.k_scale,
        cache.v_scale,
        out,
        *query_states.stride(),
        *cache.k_packed.stride(),
        *cache.v_packed.stride(),
        *cache.k_scale.stride(),
        *cache.v_scale.stride(),
        *out.stride(),
        n_heads,
        q_len,
        num_key_value_groups,
        head_dim,
        group_size,
        BLOCK_N=64,
        BLOCK_D=block_d,
    )
    return out


def _forward_fused_int4_kv(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
):
    if output_attentions or hidden_states.device.type != "cuda" or past_key_value is None:
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

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
    value_states = (
        value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
    )

    group_size = self._int4_kv_group_size
    if self.head_dim % group_size != 0:
        raise ValueError(f"head_dim={self.head_dim} must be divisible by group_size={group_size}")
    if group_size % 2 != 0:
        raise ValueError(f"group_size={group_size} must be even for INT4 packing")

    positions = _positions(cache_position, q_len, hidden_states.device)
    required_len = int(positions.max().item()) + 1
    layer_cache = _ensure_layer_cache(
        past_key_value,
        self.layer_idx,
        bsz,
        self.num_key_value_heads,
        required_len,
        self.head_dim,
        group_size,
        hidden_states.device,
    )
    _pack_kv_into_cache(layer_cache, key_states, value_states, positions, group_size)

    if q_len == 1:
        attn_output = _decode_attention(
            query_states,
            layer_cache,
            required_len,
            self.num_key_value_groups,
            group_size,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    attn_output = _prefill_attention(
        query_states,
        layer_cache,
        self.num_key_value_groups,
        group_size,
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
        attn.forward = MethodType(_forward_fused_int4_kv, attn)
        attn._int4_kv_patched = True
