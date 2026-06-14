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
    split_accum: torch.Tensor | None = None
    split_m: torch.Tensor | None = None
    split_l: torch.Tensor | None = None
    split_shape: tuple[int, int, int, int] | None = None


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
            contrib = tl.sum(p[:, None] * v, axis=0)
            acc += tl.sum(tl.where(d_offsets[:, None] == gd[None, :], contrib[None, :], 0.0), axis=1)

        m_i = m_new
        l_i = l_new
        start += BLOCK_N

    out = acc / l_i
    tl.store(
        out_ptr + batch * out_stride_b + q_head * out_stride_h + q_pos * out_stride_t + d_offsets * out_stride_d,
        out,
        mask=d_mask,
    )


@triton.heuristics({"HAS_ATTN_MASK": lambda args: args["mask_ptr"] is not None})
@triton.jit
def _int4_decode_attention_kernel(
    q_ptr,
    k_packed_ptr,
    v_packed_ptr,
    k_scale_ptr,
    v_scale_ptr,
    mask_ptr,
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
    mask_stride_b,
    mask_stride_h,
    mask_stride_q,
    mask_stride_k,
    out_stride_b,
    out_stride_h,
    out_stride_d,
    seq_len,
    mask_k_len,
    n_query_heads: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
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
        if HAS_ATTN_MASK:
            mask_valid = n < mask_k_len
            mask_bias = tl.load(
                mask_ptr + batch * mask_stride_b + n * mask_stride_k,
                mask=n_mask & mask_valid,
                other=-float("inf"),
            ).to(tl.float32)
            scores += mask_bias
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
            contrib = tl.sum(p[:, None] * v, axis=0)
            acc += tl.sum(tl.where(d_offsets[:, None] == gd[None, :], contrib[None, :], 0.0), axis=1)

        m_i = m_new
        l_i = l_new
        start += BLOCK_N

    out = acc / l_i
    tl.store(
        out_ptr + batch * out_stride_b + q_head * out_stride_h + d_offsets * out_stride_d,
        out,
        mask=d_mask,
    )


@triton.heuristics({"HAS_ATTN_MASK": lambda args: args["mask_ptr"] is not None})
@triton.jit
def _int4_decode_attention_split_kernel(
    q_ptr,
    k_packed_ptr,
    v_packed_ptr,
    k_scale_ptr,
    v_scale_ptr,
    mask_ptr,
    split_accum_ptr,
    split_m_ptr,
    split_l_ptr,
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
    mask_stride_b,
    mask_stride_h,
    mask_stride_q,
    mask_stride_k,
    accum_stride_s,
    accum_stride_b,
    accum_stride_h,
    accum_stride_d,
    ml_stride_s,
    ml_stride_b,
    ml_stride_h,
    seq_len,
    mask_k_len,
    n_query_heads: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    batch = bh // n_query_heads
    q_head = bh - batch * n_query_heads
    kv_head = q_head // num_key_value_groups
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    scale_sm = 1.4426950408889634 / tl.sqrt(tl.full((), head_dim, tl.float32))

    split_start = (seq_len * split) // num_splits
    split_end = (seq_len * (split + 1)) // num_splits
    n_offsets = tl.arange(0, BLOCK_N)
    start = split_start
    while start < split_end:
        n = start + n_offsets
        n_mask = n < split_end
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
        if HAS_ATTN_MASK:
            mask_valid = n < mask_k_len
            mask_bias = tl.load(
                mask_ptr + batch * mask_stride_b + n * mask_stride_k,
                mask=n_mask & mask_valid,
                other=-float("inf"),
            ).to(tl.float32)
            scores += mask_bias
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
            contrib = tl.sum(p[:, None] * v, axis=0)
            acc += tl.sum(tl.where(d_offsets[:, None] == gd[None, :], contrib[None, :], 0.0), axis=1)

        m_i = m_new
        l_i = l_new
        start += BLOCK_N

    tl.store(
        split_accum_ptr
        + split * accum_stride_s
        + batch * accum_stride_b
        + q_head * accum_stride_h
        + d_offsets * accum_stride_d,
        acc,
        mask=d_mask,
    )
    tl.store(split_m_ptr + split * ml_stride_s + batch * ml_stride_b + q_head * ml_stride_h, m_i)
    tl.store(split_l_ptr + split * ml_stride_s + batch * ml_stride_b + q_head * ml_stride_h, l_i)


@triton.jit
def _int4_decode_attention_split_combine_kernel(
    split_accum_ptr,
    split_m_ptr,
    split_l_ptr,
    out_ptr,
    accum_stride_s,
    accum_stride_b,
    accum_stride_h,
    accum_stride_d,
    ml_stride_s,
    ml_stride_b,
    ml_stride_h,
    out_stride_b,
    out_stride_h,
    out_stride_d,
    n_query_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    batch = bh // n_query_heads
    q_head = bh - batch * n_query_heads
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    m = tl.full((), -float("inf"), tl.float32)
    for split in tl.static_range(0, num_splits):
        m_s = tl.load(split_m_ptr + split * ml_stride_s + batch * ml_stride_b + q_head * ml_stride_h)
        m = tl.maximum(m, m_s)

    l = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for split in tl.static_range(0, num_splits):
        m_s = tl.load(split_m_ptr + split * ml_stride_s + batch * ml_stride_b + q_head * ml_stride_h)
        l_s = tl.load(split_l_ptr + split * ml_stride_s + batch * ml_stride_b + q_head * ml_stride_h)
        alpha = tl.exp2(m_s - m)
        acc_s = tl.load(
            split_accum_ptr
            + split * accum_stride_s
            + batch * accum_stride_b
            + q_head * accum_stride_h
            + d_offsets * accum_stride_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        acc += alpha * acc_s
        l += alpha * l_s

    out = acc / l
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
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, n_heads, q_len, head_dim = query_states.shape
    if q_len != 1:
        raise ValueError(f"Fused INT4 decode attention expects q_len=1, got {q_len}")
    out = torch.empty((batch, n_heads, head_dim), device=query_states.device, dtype=query_states.dtype)
    block_d = triton.next_power_of_2(head_dim)
    mask = _decode_attention_mask(attention_mask, batch, seq_len, query_states.device)
    if mask is None:
        mask_strides = (0, 0, 0, 0)
        mask_k_len = 0
    else:
        mask_strides = mask.stride()
        mask_k_len = mask.shape[-1]
    num_splits = _decode_num_splits(seq_len)
    if num_splits > 1:
        split_accum, split_m, split_l = _ensure_decode_scratch(cache, num_splits, batch, n_heads, head_dim)
        _int4_decode_attention_split_kernel[(batch * n_heads, num_splits)](
            query_states,
            cache.k_packed,
            cache.v_packed,
            cache.k_scale,
            cache.v_scale,
            mask,
            split_accum,
            split_m,
            split_l,
            query_states.stride(0),
            query_states.stride(1),
            query_states.stride(2),
            query_states.stride(3),
            *cache.k_packed.stride(),
            *cache.v_packed.stride(),
            *cache.k_scale.stride(),
            *cache.v_scale.stride(),
            *mask_strides,
            *split_accum.stride(),
            *split_m.stride(),
            seq_len,
            mask_k_len,
            n_heads,
            num_key_value_groups,
            head_dim,
            group_size,
            num_splits,
            BLOCK_N=64,
            BLOCK_D=block_d,
        )
        _int4_decode_attention_split_combine_kernel[(batch * n_heads,)](
            split_accum,
            split_m,
            split_l,
            out,
            *split_accum.stride(),
            *split_m.stride(),
            *out.stride(),
            n_heads,
            head_dim,
            num_splits,
            BLOCK_D=block_d,
        )
        return out

    grid = (batch * n_heads,)
    _int4_decode_attention_kernel[grid](
        query_states,
        cache.k_packed,
        cache.v_packed,
        cache.k_scale,
        cache.v_scale,
        mask,
        out,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        *cache.k_packed.stride(),
        *cache.v_packed.stride(),
        *cache.k_scale.stride(),
        *cache.v_scale.stride(),
        *mask_strides,
        *out.stride(),
        seq_len,
        mask_k_len,
        n_heads,
        num_key_value_groups,
        head_dim,
        group_size,
        BLOCK_N=64,
        BLOCK_D=block_d,
    )
    return out


def _decode_num_splits(seq_len: int) -> int:
    if seq_len >= 2048:
        return 8
    if seq_len >= 1024:
        return 4
    return 1


def _ensure_decode_scratch(
    cache: Int4KVLayerCache,
    num_splits: int,
    batch: int,
    n_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (num_splits, batch, n_heads, head_dim)
    if cache.split_shape != shape or cache.split_accum is None or cache.split_m is None or cache.split_l is None:
        device = cache.k_packed.device
        cache.split_accum = torch.empty(shape, device=device, dtype=torch.float32)
        cache.split_m = torch.empty((num_splits, batch, n_heads), device=device, dtype=torch.float32)
        cache.split_l = torch.empty((num_splits, batch, n_heads), device=device, dtype=torch.float32)
        cache.split_shape = shape
    return cache.split_accum, cache.split_m, cache.split_l


def _decode_attention_mask(
    attention_mask: torch.Tensor | None,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        mask = attention_mask[:, :, -1:, :seq_len].to(device=device)
        if mask.dtype == torch.bool:
            bias = torch.zeros(mask.shape, device=device, dtype=torch.float32)
            return bias.masked_fill(~mask, -float("inf")).contiguous()
        return mask.to(dtype=torch.float32).contiguous()
    if attention_mask.dim() == 2:
        mask = attention_mask[:, -seq_len:].to(device=device)
        if mask.dtype == torch.bool:
            keep = mask
        else:
            keep = mask != 0
        bias = torch.zeros((batch, 1, 1, seq_len), device=device, dtype=torch.float32)
        return bias.masked_fill(~keep[:, None, None, :], -float("inf"))
    raise ValueError(f"INT4 decode attention only supports 2D or 4D attention_mask, got {attention_mask.dim()}D")


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
            attention_mask=attention_mask,
        )
        attn_output = attn_output.unsqueeze(2).transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    key_states_full = self._int4_kv_repeat_kv(key_states, self.num_key_value_groups)
    value_states_full = self._int4_kv_repeat_kv(value_states, self.num_key_value_groups)
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states_full.shape[-2]]
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states_full = key_states_full.contiguous()
        value_states_full = value_states_full.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states_full,
        value_states_full,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=causal_mask is None and q_len > 1,
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
