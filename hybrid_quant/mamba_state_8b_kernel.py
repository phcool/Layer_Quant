from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import triton
import triton.language as tl


MX8_BLOCK_SIZE = 16
MX8_PAIR_SIZE = 2
MX8_MANTISSA_MAX = 63.0
MX4_BLOCK_SIZE = 16
MX4_PAIR_SIZE = 2
MX4_MANTISSA_MAX = 7.0


@dataclass
class MambaStateKernelCache:
    q_state: torch.Tensor
    shared_exp: torch.Tensor
    micro_exp: torch.Tensor
    format_bits: int = 8
    step: int = 0


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.jit
def _state_update_requant_mx8_row_kernel(
    q_state_ptr,
    shared_exp_ptr,
    micro_exp_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_q_batch: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_q_dstate: tl.constexpr,
    stride_exp_batch: tl.constexpr,
    stride_exp_head: tl.constexpr,
    stride_exp_dim: tl.constexpr,
    stride_exp_group: tl.constexpr,
    stride_micro_batch: tl.constexpr,
    stride_micro_head: tl.constexpr,
    stride_micro_dim: tl.constexpr,
    stride_micro_pair: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_x_head: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_dt_batch: tl.constexpr,
    stride_dt_head: tl.constexpr,
    stride_dt_dim: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_A_dim: tl.constexpr,
    stride_A_dstate: tl.constexpr,
    stride_B_batch: tl.constexpr,
    stride_B_group: tl.constexpr,
    stride_B_dstate: tl.constexpr,
    stride_C_batch: tl.constexpr,
    stride_C_group: tl.constexpr,
    stride_C_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    stride_D_dim: tl.constexpr,
    stride_z_batch: tl.constexpr,
    stride_z_head: tl.constexpr,
    stride_z_dim: tl.constexpr,
    stride_out_batch: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_out_dim: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    dt_softplus: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    RNG_SEED: tl.constexpr,
    RNG_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < dstate

    q = tl.load(
        q_state_ptr
        + pid_b * stride_q_batch
        + pid_h * stride_q_head
        + pid_m * stride_q_dim
        + offs_n * stride_q_dstate,
        mask=n_mask,
        other=0,
    ).to(tl.float32)
    block_ids = offs_n // 16
    pair_ids = offs_n // 2
    shared_exp = tl.load(
        shared_exp_ptr
        + pid_b * stride_exp_batch
        + pid_h * stride_exp_head
        + pid_m * stride_exp_dim
        + block_ids * stride_exp_group,
        mask=n_mask,
        other=127,
    ).to(tl.float32)
    micro_exp = tl.load(
        micro_exp_ptr
        + pid_b * stride_micro_batch
        + pid_h * stride_micro_head
        + pid_m * stride_micro_dim
        + pair_ids * stride_micro_pair,
        mask=n_mask,
        other=0,
    ).to(tl.float32)
    state = q * tl.exp2(shared_exp - 127.0 + micro_exp)

    x = tl.load(x_ptr + pid_b * stride_x_batch + pid_h * stride_x_head + pid_m * stride_x_dim).to(tl.float32)
    dt = tl.load(dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head + pid_m * stride_dt_dim).to(tl.float32)
    if HAS_DT_BIAS:
        dt += tl.load(dt_bias_ptr + pid_h * stride_D_head + pid_m * stride_D_dim).to(tl.float32)
    if dt_softplus:
        dt = _softplus(dt)

    group = pid_h // nheads_ngroups_ratio
    A = tl.load(
        A_ptr + pid_h * stride_A_head + pid_m * stride_A_dim + offs_n * stride_A_dstate,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    B = tl.load(
        B_ptr + pid_b * stride_B_batch + group * stride_B_group + offs_n * stride_B_dstate,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    C = tl.load(
        C_ptr + pid_b * stride_C_batch + group * stride_C_group + offs_n * stride_C_dstate,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)

    state = state * tl.exp(A * dt) + B * dt * x
    y = tl.sum(tl.where(n_mask, state * C, 0.0), axis=0)
    if HAS_D:
        D = tl.load(D_ptr + pid_h * stride_D_head + pid_m * stride_D_dim).to(tl.float32)
        y += x * D
    if HAS_Z:
        z = tl.load(z_ptr + pid_b * stride_z_batch + pid_h * stride_z_head + pid_m * stride_z_dim).to(tl.float32)
        y *= z
    tl.store(out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head + pid_m * stride_out_dim, y)

    groups_per_row: tl.constexpr = BLOCK_N // 16
    blocks = tl.reshape(state, (groups_per_row, 16))
    abs_blocks = tl.abs(blocks)
    amax = tl.max(abs_blocks, axis=1)
    raw_exp = tl.ceil(tl.log2(tl.maximum(amax / 126.0, 1.0e-30)))
    biased_exp = tl.minimum(tl.maximum(raw_exp + 127.0, 1.0), 254.0)
    base_scale = tl.where(amax == 0.0, 1.0, tl.exp2(biased_exp - 127.0))

    group_offsets = tl.arange(0, groups_per_row)
    tl.store(
        shared_exp_ptr
        + pid_b * stride_exp_batch
        + pid_h * stride_exp_head
        + pid_m * stride_exp_dim
        + group_offsets * stride_exp_group,
        biased_exp.to(tl.uint8),
        mask=group_offsets < (dstate // 16),
    )

    pair_vals = tl.reshape(blocks, (groups_per_row * 8, 2))
    pair_amax = tl.max(tl.abs(pair_vals), axis=1)
    base_per_pair = tl.reshape(tl.broadcast_to(tl.expand_dims(base_scale, 1), (groups_per_row, 8)), (groups_per_row * 8,))
    micro_vec = tl.where(pair_amax > 63.0 * base_per_pair, 1, 0)
    pair_scale = base_per_pair * tl.exp2(micro_vec.to(tl.float32))
    q_abs = tl.minimum(tl.abs(pair_vals) / pair_scale[:, None], 63.0)
    q_floor = tl.floor(q_abs)
    pair_offsets = tl.arange(0, groups_per_row * 8)
    elem_offsets = tl.reshape(tl.arange(0, BLOCK_N), (groups_per_row * 8, 2))
    if STOCHASTIC:
        row_id = (pid_b * tl.num_programs(1) + pid_h) * dim + pid_m
        rnd = tl.rand(RNG_SEED, RNG_OFFSET + row_id * dstate + elem_offsets)
        q_level = q_floor + (rnd < (q_abs - q_floor)).to(tl.float32)
    else:
        q_level = tl.floor(q_abs + 0.5)
    q_level = tl.minimum(q_level, 63.0)
    q_signed = tl.where(pair_vals < 0.0, -q_level, q_level).to(tl.int8)
    q_flat = tl.reshape(q_signed, (BLOCK_N,))
    tl.store(
        q_state_ptr
        + pid_b * stride_q_batch
        + pid_h * stride_q_head
        + pid_m * stride_q_dim
        + offs_n * stride_q_dstate,
        q_flat,
        mask=n_mask,
    )
    tl.store(
        micro_exp_ptr
        + pid_b * stride_micro_batch
        + pid_h * stride_micro_head
        + pid_m * stride_micro_dim
        + pair_offsets * stride_micro_pair,
        micro_vec.to(tl.uint8),
        mask=pair_offsets < (dstate // 2),
    )


@triton.jit
def _state_update_mx8_kernel(
    q_state_ptr,
    shared_exp_ptr,
    micro_exp_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    tmp_state_ptr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_q_batch: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_q_dstate: tl.constexpr,
    stride_exp_batch: tl.constexpr,
    stride_exp_head: tl.constexpr,
    stride_exp_dim: tl.constexpr,
    stride_exp_group: tl.constexpr,
    stride_micro_batch: tl.constexpr,
    stride_micro_head: tl.constexpr,
    stride_micro_dim: tl.constexpr,
    stride_micro_pair: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_x_head: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_dt_batch: tl.constexpr,
    stride_dt_head: tl.constexpr,
    stride_dt_dim: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_A_dim: tl.constexpr,
    stride_A_dstate: tl.constexpr,
    stride_B_batch: tl.constexpr,
    stride_B_group: tl.constexpr,
    stride_B_dstate: tl.constexpr,
    stride_C_batch: tl.constexpr,
    stride_C_group: tl.constexpr,
    stride_C_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    stride_D_dim: tl.constexpr,
    stride_z_batch: tl.constexpr,
    stride_z_head: tl.constexpr,
    stride_z_dim: tl.constexpr,
    stride_out_batch: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_out_dim: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    dt_softplus: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    FORMAT_BITS: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    RNG_SEED: tl.constexpr,
    RNG_OFFSET: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)

    q_base = (
        q_state_ptr
        + pid_b * stride_q_batch
        + pid_h * stride_q_head
        + offs_m[:, None] * stride_q_dim
        + offs_n[None, :] * stride_q_dstate
    )
    block_ids = offs_n // 16
    pair_ids = offs_n // 2
    exp_ptrs = (
        shared_exp_ptr
        + pid_b * stride_exp_batch
        + pid_h * stride_exp_head
        + offs_m[:, None] * stride_exp_dim
        + block_ids[None, :] * stride_exp_group
    )
    micro_ptrs = (
        micro_exp_ptr
        + pid_b * stride_micro_batch
        + pid_h * stride_micro_head
        + offs_m[:, None] * stride_micro_dim
        + pair_ids[None, :] * stride_micro_pair
    )
    q = tl.load(q_base, mask=mask, other=0).to(tl.float32)
    shared_exp = tl.load(exp_ptrs, mask=mask, other=127).to(tl.float32)
    micro_exp = tl.load(micro_ptrs, mask=mask, other=0).to(tl.float32)
    state = q * tl.exp2(shared_exp - 127.0 + micro_exp)

    x = tl.load(
        x_ptr + pid_b * stride_x_batch + pid_h * stride_x_head + offs_m * stride_x_dim,
        mask=offs_m < dim,
        other=0.0,
    ).to(tl.float32)
    dt = tl.load(
        dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head + offs_m * stride_dt_dim,
        mask=offs_m < dim,
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + pid_h * stride_D_head + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0)
        dt += dt_bias
    if dt_softplus:
        dt = _softplus(dt)

    group = pid_h // nheads_ngroups_ratio
    A = tl.load(
        A_ptr + pid_h * stride_A_head + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    B = tl.load(
        B_ptr + pid_b * stride_B_batch + group * stride_B_group + offs_n * stride_B_dstate,
        mask=offs_n < dstate,
        other=0.0,
    ).to(tl.float32)
    C = tl.load(
        C_ptr + pid_b * stride_C_batch + group * stride_C_group + offs_n * stride_C_dstate,
        mask=offs_n < dstate,
        other=0.0,
    ).to(tl.float32)

    state = state * tl.exp(A * dt[:, None]) + B[None, :] * dt[:, None] * x[:, None]
    y = tl.sum(tl.where(mask, state * C[None, :], 0.0), axis=1)
    if HAS_D:
        D = tl.load(D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        y += x * D
    if HAS_Z:
        z = tl.load(
            z_ptr + pid_b * stride_z_batch + pid_h * stride_z_head + offs_m * stride_z_dim,
            mask=offs_m < dim,
            other=0.0,
        ).to(tl.float32)
        y *= z

    if FORMAT_BITS == 8:
        row_ids = (pid_b * tl.num_programs(2) + pid_h) * dim + offs_m
        state_groups = tl.reshape(state, (BLOCK_SIZE_M, BLOCK_SIZE_N // 16, 16))
        for group_start in tl.static_range(0, BLOCK_SIZE_N, 16):
            if group_start < dstate:
                group_vals = state_groups[:, group_start // 16, :]
                group_vals = tl.where(offs_m[:, None] < dim, group_vals, 0.0)
                abs_vals = tl.abs(group_vals)
                amax = tl.max(abs_vals, axis=1)
                raw_exp = tl.ceil(tl.log2(tl.maximum(amax / 126.0, 1.0e-30)))
                biased_exp = tl.minimum(tl.maximum(raw_exp + 127.0, 1.0), 254.0)
                base_scale = tl.where(amax == 0.0, 1.0, tl.exp2(biased_exp - 127.0))
                tl.store(
                    shared_exp_ptr
                    + pid_b * stride_exp_batch
                    + pid_h * stride_exp_head
                    + offs_m * stride_exp_dim
                    + (group_start // 16) * stride_exp_group,
                    biased_exp.to(tl.uint8),
                    mask=offs_m < dim,
                )
                group_pairs = tl.reshape(group_vals, (BLOCK_SIZE_M, 8, 2))
                for pair_start in tl.static_range(0, 16, 2):
                    pair_vals = group_pairs[:, pair_start // 2, :]
                    pair_amax = tl.max(tl.abs(pair_vals), axis=1)
                    micro_vec = tl.where(pair_amax > 63.0 * base_scale, 1, 0)
                    scale = base_scale * tl.exp2(micro_vec.to(tl.float32))
                    q_abs = tl.minimum(tl.abs(pair_vals) / scale[:, None], 63.0)
                    q_floor = tl.floor(q_abs)
                    if STOCHASTIC:
                        rng_offsets = (
                            RNG_OFFSET
                            + row_ids[:, None] * dstate
                            + (group_start + pair_start + tl.arange(0, 2))[None, :]
                        )
                        rnd = tl.rand(RNG_SEED, rng_offsets)
                        q_level = q_floor + (rnd < (q_abs - q_floor)).to(tl.float32)
                    else:
                        q_level = tl.floor(q_abs + 0.5)
                    q_level = tl.minimum(q_level, 63.0)
                    q_signed = tl.where(pair_vals < 0.0, -q_level, q_level).to(tl.int8)
                    tl.store(
                        q_state_ptr
                        + pid_b * stride_q_batch
                        + pid_h * stride_q_head
                        + offs_m[:, None] * stride_q_dim
                        + (group_start + pair_start + tl.arange(0, 2))[None, :] * stride_q_dstate,
                        q_signed,
                        mask=offs_m[:, None] < dim,
                    )
                    tl.store(
                        micro_exp_ptr
                        + pid_b * stride_micro_batch
                        + pid_h * stride_micro_head
                        + offs_m * stride_micro_dim
                        + ((group_start + pair_start) // 2) * stride_micro_pair,
                        micro_vec.to(tl.uint8),
                        mask=offs_m < dim,
                    )
    else:
        tl.store(
            tmp_state_ptr
            + pid_b * stride_q_batch
            + pid_h * stride_q_head
            + offs_m[:, None] * stride_q_dim
            + offs_n[None, :] * stride_q_dstate,
            state,
            mask=mask,
        )
    tl.store(
        out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head + offs_m * stride_out_dim,
        y,
        mask=offs_m < dim,
    )


@triton.jit
def _requantize_mx8_kernel(
    state_ptr,
    q_state_ptr,
    shared_exp_ptr,
    micro_exp_ptr,
    rows: tl.constexpr,
    dstate: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    RNG_SEED: tl.constexpr,
    RNG_OFFSET: tl.constexpr,
):
    groups_per_row: tl.constexpr = dstate // 16
    pid = tl.program_id(0)
    row = pid // groups_per_row
    group = pid - row * groups_per_row
    offs = group * 16 + tl.arange(0, 16)
    vals = tl.load(state_ptr + row * dstate + offs).to(tl.float32)
    abs_vals = tl.abs(vals)
    amax = tl.max(abs_vals, axis=0)
    raw_exp = tl.ceil(tl.log2(tl.maximum(amax / 126.0, 1.0e-30)))
    biased_exp = tl.minimum(tl.maximum(raw_exp + 127.0, 1.0), 254.0)
    base_scale = tl.where(amax == 0.0, 1.0, tl.exp2(biased_exp - 127.0))

    even_vals = tl.reshape(vals, (8, 2))
    pair_amax = tl.max(tl.abs(even_vals), axis=1)
    micro_vec = tl.where(pair_amax > 63.0 * base_scale, 1, 0)
    micro_pairs = tl.reshape(tl.broadcast_to(tl.expand_dims(micro_vec, 1), (8, 2)), (16,))
    micro = micro_pairs
    scale = base_scale * tl.exp2(micro.to(tl.float32))
    q_abs = tl.minimum(abs_vals / scale, 63.0)
    q_floor = tl.floor(q_abs)
    if STOCHASTIC:
        rnd = tl.rand(RNG_SEED, RNG_OFFSET + row * dstate + offs)
        q_level = q_floor + (rnd < (q_abs - q_floor)).to(tl.float32)
    else:
        q_level = tl.floor(q_abs + 0.5)
    q_level = tl.minimum(q_level, 63.0)
    q_signed = tl.where(vals < 0.0, -q_level, q_level).to(tl.int8)

    tl.store(q_state_ptr + row * dstate + offs, q_signed)
    tl.store(shared_exp_ptr + row * groups_per_row + group, biased_exp.to(tl.uint8))
    tl.store(
        micro_exp_ptr + row * (dstate // 2) + group * 8 + tl.arange(0, 8),
        micro_vec.to(tl.uint8),
    )


@triton.jit
def _requantize_mx4_kernel(
    state_ptr,
    q_state_ptr,
    shared_exp_ptr,
    micro_exp_ptr,
    rows: tl.constexpr,
    dstate: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    RNG_SEED: tl.constexpr,
    RNG_OFFSET: tl.constexpr,
):
    groups_per_row: tl.constexpr = dstate // 16
    pid = tl.program_id(0)
    row = pid // groups_per_row
    group = pid - row * groups_per_row
    offs = group * 16 + tl.arange(0, 16)
    vals = tl.load(state_ptr + row * dstate + offs).to(tl.float32)
    abs_vals = tl.abs(vals)
    amax = tl.max(abs_vals, axis=0)
    raw_exp = tl.ceil(tl.log2(tl.maximum(amax / 14.0, 1.0e-30)))
    biased_exp = tl.minimum(tl.maximum(raw_exp + 127.0, 1.0), 254.0)
    base_scale = tl.where(amax == 0.0, 1.0, tl.exp2(biased_exp - 127.0))

    even_vals = tl.reshape(vals, (8, 2))
    pair_amax = tl.max(tl.abs(even_vals), axis=1)
    micro_vec = tl.where(pair_amax > 7.0 * base_scale, 1, 0)
    micro_pairs = tl.reshape(
        tl.broadcast_to(tl.expand_dims(micro_vec, 1), (8, 2)),
        (16,),
    )
    scale = base_scale * tl.exp2(micro_pairs.to(tl.float32))
    q_abs = tl.minimum(abs_vals / scale, 7.0)
    q_floor = tl.floor(q_abs)
    if STOCHASTIC:
        rnd = tl.rand(RNG_SEED, RNG_OFFSET + row * dstate + offs)
        q_level = q_floor + (rnd < (q_abs - q_floor)).to(tl.float32)
    else:
        q_level = tl.floor(q_abs + 0.5)
    q_level = tl.minimum(q_level, 7.0)
    q_signed = tl.where(vals < 0.0, -q_level, q_level).to(tl.int8)

    tl.store(q_state_ptr + row * dstate + offs, q_signed)
    tl.store(shared_exp_ptr + row * groups_per_row + group, biased_exp.to(tl.uint8))
    tl.store(
        micro_exp_ptr
        + row * (dstate // 2)
        + group * 8
        + tl.arange(0, 8),
        micro_vec.to(tl.uint8),
    )


def _quantize_initial_state_mx(
    state: torch.Tensor,
    *,
    format_bits: int,
    block_size: int,
    pair_size: int,
    mantissa_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.shape[-1] % block_size != 0:
        raise ValueError(f"MX{format_bits} state dim must be divisible by {block_size}, got {state.shape[-1]}")
    working = state.float().contiguous()
    blocks = working.reshape(*working.shape[:-1], working.shape[-1] // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    raw_exp = torch.ceil(torch.log2(torch.clamp(amax / (2.0 * mantissa_max), min=1.0e-30)))
    biased_exp = torch.clamp(raw_exp + 127.0, 1.0, 254.0)
    base_scale = torch.where(amax == 0, torch.ones_like(amax), torch.exp2(biased_exp - 127.0))

    pairs = blocks.reshape(*blocks.shape[:-1], block_size // pair_size, pair_size)
    pair_amax = pairs.abs().amax(dim=-1, keepdim=True)
    micro = (pair_amax > mantissa_max * base_scale.unsqueeze(-2)).to(torch.uint8)
    pair_scale = base_scale.unsqueeze(-2) * torch.exp2(micro.float())
    q = torch.round(torch.clamp(pairs.abs() / pair_scale, 0.0, mantissa_max)) * pairs.sign()
    q_state = q.reshape_as(working).to(torch.int8)
    shared_exp = biased_exp.squeeze(-1).to(torch.uint8).contiguous()
    micro_exp = micro.squeeze(-1).reshape(*working.shape[:-1], working.shape[-1] // pair_size).contiguous()
    return q_state.contiguous(), shared_exp, micro_exp


def allocate_state_kernel_cache(ssm_state_4d: torch.Tensor, group_size: int = MX8_BLOCK_SIZE, format_bits: int = 8) -> MambaStateKernelCache:
    if format_bits == 8:
        if group_size != MX8_BLOCK_SIZE:
            raise ValueError(f"MX8 uses a fixed {MX8_BLOCK_SIZE}-value block, got group_size={group_size}")
        q_state, shared_exp, micro_exp = _quantize_initial_state_mx(
            ssm_state_4d,
            format_bits=8,
            block_size=MX8_BLOCK_SIZE,
            pair_size=MX8_PAIR_SIZE,
            mantissa_max=MX8_MANTISSA_MAX,
        )
        return MambaStateKernelCache(q_state=q_state, shared_exp=shared_exp, micro_exp=micro_exp, format_bits=8)
    if format_bits == 4:
        if group_size != MX4_BLOCK_SIZE:
            raise ValueError(f"MX4 uses a fixed {MX4_BLOCK_SIZE}-value block, got group_size={group_size}")
        q_state, shared_exp, micro_exp = _quantize_initial_state_mx(
            ssm_state_4d,
            format_bits=4,
            block_size=MX4_BLOCK_SIZE,
            pair_size=MX4_PAIR_SIZE,
            mantissa_max=MX4_MANTISSA_MAX,
        )
        return MambaStateKernelCache(q_state=q_state, shared_exp=shared_exp, micro_exp=micro_exp, format_bits=4)
    raise ValueError(f"Only MX8 and MX4 state quantization are supported, got format_bits={format_bits}")


def mx_state_selective_update(
    cache: MambaStateKernelCache,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    stochastic: bool = True,
    seed: int = 1234,
) -> torch.Tensor:
    if cache.q_state.ndim != 4:
        raise ValueError(f"Expected q_state [B,H,P,N], got {tuple(cache.q_state.shape)}")
    batch, nheads, dim, dstate = cache.q_state.shape
    if x.shape != (batch, nheads, dim):
        raise ValueError(f"Expected x shape {(batch, nheads, dim)}, got {tuple(x.shape)}")
    if B.ndim != 3 or C.ndim != 3:
        raise ValueError("Expected B and C with shape [batch, ngroups, dstate] for decode update.")
    nheads_ngroups_ratio = nheads // B.shape[1]
    out = torch.empty_like(x)
    d_strides = D.stride() if D is not None else (0, 0)
    z_strides = z.stride() if z is not None else (0, 0, 0)

    if cache.format_bits == 8:
        grid = (batch, nheads, dim)
        _state_update_requant_mx8_row_kernel[grid](
            cache.q_state,
            cache.shared_exp,
            cache.micro_exp,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            nheads_ngroups_ratio,
            *cache.q_state.stride(),
            *cache.shared_exp.stride(),
            *cache.micro_exp.stride(),
            *x.stride(),
            *dt.stride(),
            *A.stride(),
            *B.stride(),
            *C.stride(),
            *d_strides,
            *z_strides,
            *out.stride(),
            dim,
            dstate,
            dt_softplus,
            STOCHASTIC=stochastic,
            RNG_SEED=seed,
            RNG_OFFSET=cache.step * batch * nheads * dim * dstate,
            BLOCK_N=triton.next_power_of_2(dstate),
        )
        cache.step += 1
        return out

    tmp_state = torch.empty(cache.q_state.shape, dtype=torch.float32, device=cache.q_state.device)
    grid = (triton.cdiv(dim, 16), batch, nheads)
    _state_update_mx8_kernel[grid](
        cache.q_state,
        cache.shared_exp,
        cache.micro_exp,
        x,
        dt,
        dt_bias,
        A,
        B,
        C,
        D,
        z,
        out,
        tmp_state,
        nheads_ngroups_ratio,
        *cache.q_state.stride(),
        *cache.shared_exp.stride(),
        *cache.micro_exp.stride(),
        *x.stride(),
        *dt.stride(),
        *A.stride(),
        *B.stride(),
        *C.stride(),
        *d_strides,
        *z_strides,
        *out.stride(),
        dim,
        dstate,
        dt_softplus,
        FORMAT_BITS=cache.format_bits,
        STOCHASTIC=stochastic,
        RNG_SEED=seed,
        RNG_OFFSET=cache.step * batch * nheads * dim * dstate,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=triton.next_power_of_2(dstate),
    )

    rows = batch * nheads * dim
    if cache.format_bits == 4:
        requant_grid = (rows * (dstate // MX4_BLOCK_SIZE),)
        _requantize_mx4_kernel[requant_grid](
            tmp_state,
            cache.q_state,
            cache.shared_exp,
            cache.micro_exp,
            rows,
            dstate,
            stochastic,
            seed,
            cache.step * rows * dstate,
        )
    else:
        raise ValueError(f"Unsupported MX state format_bits={cache.format_bits}")
    cache.step += 1
    return out


def mx8_state_selective_update(*args, **kwargs) -> torch.Tensor:
    return mx_state_selective_update(*args, **kwargs)


def mx4_state_selective_update(*args, **kwargs) -> torch.Tensor:
    return mx_state_selective_update(*args, **kwargs)


def patch_nemotron_h_mamba_decode_state_kernel(
    model,
    group_size: int = MX8_BLOCK_SIZE,
    stochastic: bool = True,
    seed: int = 1234,
) -> None:
    module = __import__(model.__class__.__module__, fromlist=["selective_state_update"])
    original = getattr(module, "_mx_state_kernel_original", module.selective_state_update)
    module._mx_state_kernel_stochastic = stochastic
    module._mx_state_kernel_seed = seed

    def selective_state_update_mx(ssm_state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
        caches = getattr(module, "_mamba_decode_state_kernel_caches", {})
        kernel_cache = caches.get(id(ssm_state))
        if kernel_cache is None:
            return original(ssm_state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus)
        return mx_state_selective_update(
            kernel_cache,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            stochastic=getattr(module, "_mx_state_kernel_stochastic", True),
            seed=getattr(module, "_mx_state_kernel_seed", 1234),
        )

    if not getattr(module, "_mx_state_kernel_patched", False):
        module.selective_state_update = selective_state_update_mx
        module._mx_state_kernel_original = original
        module._mx_state_kernel_patched = True
    model._mx_state_kernel_group_size = group_size


def register_mamba_state_kernel_caches(model, cache_params, mode_by_layer: dict[int, str]) -> None:
    caches = {}
    for layer_idx, mode in mode_by_layer.items():
        if mode not in {"mx8", "mx4"}:
            raise ValueError(f"Only MX8 and MX4 state caches are supported, got mode={mode!r}")
        layer = model.backbone.layers[layer_idx]
        mixer = layer.mixer
        ssm_state = cache_params.ssm_states[layer_idx]
        state_4d = ssm_state.view(ssm_state.shape[0], mixer.num_heads, mixer.head_dim, mixer.ssm_state_size)
        format_bits = 8 if mode == "mx8" else 4
        block_size = MX8_BLOCK_SIZE if mode == "mx8" else MX4_BLOCK_SIZE
        caches[id(ssm_state)] = allocate_state_kernel_cache(
            state_4d,
            group_size=block_size,
            format_bits=format_bits,
        )
    model._mamba_decode_state_kernel_caches = caches
    module = __import__(model.__class__.__module__, fromlist=["selective_state_update"])
    module._mamba_decode_state_kernel_caches = dict(caches)
    model._mamba_decode_state_kernel_meta = {
        layer_idx: SimpleNamespace(
            mode=mode_by_layer[layer_idx],
            block_size=MX8_BLOCK_SIZE if mode_by_layer[layer_idx] == "mx8" else MX4_BLOCK_SIZE,
            format_bits=8 if mode_by_layer[layer_idx] == "mx8" else 4,
        )
        for layer_idx in mode_by_layer
    }
