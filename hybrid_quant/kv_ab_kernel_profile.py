from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_kv_load_kernel(
    k_ptr,
    v_ptr,
    out_ptr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_d: tl.constexpr,
    n_query_heads: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    block_n = tl.program_id(1)
    batch = bh // n_query_heads
    q_head = bh - batch * n_query_heads
    kv_head = q_head // num_key_value_groups
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, BLOCK_D)
    mask = (n[:, None] < seq_len) & (d[None, :] < head_dim)
    k = tl.load(
        k_ptr + batch * k_stride_b + kv_head * k_stride_h + n[:, None] * k_stride_t + d[None, :] * k_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + batch * v_stride_b + kv_head * v_stride_h + n[:, None] * v_stride_t + d[None, :] * v_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(out_ptr + bh * tl.num_programs(1) + block_n, tl.sum(k + v))


@triton.jit
def _int4_kv_load_kernel(
    k_packed_ptr,
    v_packed_ptr,
    k_scale_ptr,
    v_scale_ptr,
    out_ptr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_p: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_p: tl.constexpr,
    ks_stride_b: tl.constexpr,
    ks_stride_h: tl.constexpr,
    ks_stride_t: tl.constexpr,
    ks_stride_g: tl.constexpr,
    vs_stride_b: tl.constexpr,
    vs_stride_h: tl.constexpr,
    vs_stride_t: tl.constexpr,
    vs_stride_g: tl.constexpr,
    n_query_heads: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    block_n = tl.program_id(1)
    batch = bh // n_query_heads
    q_head = bh - batch * n_query_heads
    kv_head = q_head // num_key_value_groups
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.full((), 0.0, tl.float32)
    for d_start in tl.static_range(0, BLOCK_D, group_size):
        gd = d_start + tl.arange(0, group_size)
        pair = gd // 2
        mask = (n[:, None] < seq_len) & (gd[None, :] < head_dim)
        k_byte = tl.load(
            k_packed_ptr
            + batch * k_stride_b
            + kv_head * k_stride_h
            + n[:, None] * k_stride_t
            + pair[None, :] * k_stride_p,
            mask=mask,
            other=0,
        ).to(tl.float32)
        v_byte = tl.load(
            v_packed_ptr
            + batch * v_stride_b
            + kv_head * v_stride_h
            + n[:, None] * v_stride_t
            + pair[None, :] * v_stride_p,
            mask=mask,
            other=0,
        ).to(tl.float32)
        k_scale = tl.load(
            k_scale_ptr
            + batch * ks_stride_b
            + kv_head * ks_stride_h
            + n * ks_stride_t
            + (d_start // group_size) * ks_stride_g,
            mask=n < seq_len,
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            v_scale_ptr
            + batch * vs_stride_b
            + kv_head * vs_stride_h
            + n * vs_stride_t
            + (d_start // group_size) * vs_stride_g,
            mask=n < seq_len,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(k_byte + v_byte) + tl.sum(k_scale + v_scale)
    tl.store(out_ptr + bh * tl.num_programs(1) + block_n, acc)


@triton.jit
def _int4_kv_dequant_kernel(
    k_packed_ptr,
    v_packed_ptr,
    k_scale_ptr,
    v_scale_ptr,
    out_ptr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_p: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_p: tl.constexpr,
    ks_stride_b: tl.constexpr,
    ks_stride_h: tl.constexpr,
    ks_stride_t: tl.constexpr,
    ks_stride_g: tl.constexpr,
    vs_stride_b: tl.constexpr,
    vs_stride_h: tl.constexpr,
    vs_stride_t: tl.constexpr,
    vs_stride_g: tl.constexpr,
    n_query_heads: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    block_n = tl.program_id(1)
    batch = bh // n_query_heads
    q_head = bh - batch * n_query_heads
    kv_head = q_head // num_key_value_groups
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.full((), 0.0, tl.float32)
    for d_start in tl.static_range(0, BLOCK_D, group_size):
        gd = d_start + tl.arange(0, group_size)
        pair = gd // 2
        mask = (n[:, None] < seq_len) & (gd[None, :] < head_dim)
        k_byte = tl.load(
            k_packed_ptr
            + batch * k_stride_b
            + kv_head * k_stride_h
            + n[:, None] * k_stride_t
            + pair[None, :] * k_stride_p,
            mask=mask,
            other=0,
        ).to(tl.int32)
        v_byte = tl.load(
            v_packed_ptr
            + batch * v_stride_b
            + kv_head * v_stride_h
            + n[:, None] * v_stride_t
            + pair[None, :] * v_stride_p,
            mask=mask,
            other=0,
        ).to(tl.int32)
        k_nibble = tl.where((gd[None, :] & 1) == 0, k_byte & 15, (k_byte >> 4) & 15)
        v_nibble = tl.where((gd[None, :] & 1) == 0, v_byte & 15, (v_byte >> 4) & 15)
        k_signed = tl.where(k_nibble >= 8, k_nibble - 16, k_nibble).to(tl.float32)
        v_signed = tl.where(v_nibble >= 8, v_nibble - 16, v_nibble).to(tl.float32)
        k_scale = tl.load(
            k_scale_ptr
            + batch * ks_stride_b
            + kv_head * ks_stride_h
            + n * ks_stride_t
            + (d_start // group_size) * ks_stride_g,
            mask=n < seq_len,
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            v_scale_ptr
            + batch * vs_stride_b
            + kv_head * vs_stride_h
            + n * vs_stride_t
            + (d_start // group_size) * vs_stride_g,
            mask=n < seq_len,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(k_signed * k_scale[:, None] + v_signed * v_scale[:, None])
    tl.store(out_ptr + bh * tl.num_programs(1) + block_n, acc)


@dataclass
class KVABConfig:
    batch_size: int = 8
    seq_len: int = 2048
    n_query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    group_size: int = 64
    block_n: int = 64
    warmup: int = 20
    repeats: int = 200
    dtype: torch.dtype = torch.bfloat16

    @property
    def num_key_value_groups(self) -> int:
        if self.n_query_heads % self.kv_heads != 0:
            raise ValueError("n_query_heads must be divisible by kv_heads")
        return self.n_query_heads // self.kv_heads

    @property
    def groups(self) -> int:
        if self.head_dim % self.group_size != 0:
            raise ValueError("head_dim must be divisible by group_size")
        return self.head_dim // self.group_size


def _record_ms(fn, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def _make_inputs(config: KVABConfig, device: torch.device):
    k = torch.randn(
        config.batch_size,
        config.kv_heads,
        config.seq_len,
        config.head_dim,
        device=device,
        dtype=config.dtype,
    )
    v = torch.randn_like(k)
    packed_shape = (config.batch_size, config.kv_heads, config.seq_len, config.head_dim // 2)
    scale_shape = (config.batch_size, config.kv_heads, config.seq_len, config.groups)
    k_packed = torch.randint(0, 256, packed_shape, device=device, dtype=torch.uint8)
    v_packed = torch.randint(0, 256, packed_shape, device=device, dtype=torch.uint8)
    k_scale = torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.25
    v_scale = torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.25
    blocks = triton.cdiv(config.seq_len, config.block_n)
    scratch = torch.empty((config.batch_size * config.n_query_heads, blocks), device=device, dtype=torch.float32)
    return k, v, k_packed, v_packed, k_scale, v_scale, scratch


def run_kv_ab_profile(config: KVABConfig, device: torch.device | None = None) -> list[dict[str, float | str]]:
    if device is None:
        device = torch.device("cuda")
    if device.type != "cuda":
        raise ValueError("KV A/B profile requires CUDA")
    k, v, k_packed, v_packed, k_scale, v_scale, scratch = _make_inputs(config, device)
    grid = (config.batch_size * config.n_query_heads, triton.cdiv(config.seq_len, config.block_n))
    block_d = triton.next_power_of_2(config.head_dim)

    def bf16_load():
        _bf16_kv_load_kernel[grid](
            k,
            v,
            scratch,
            *k.stride(),
            *v.stride(),
            config.n_query_heads,
            config.num_key_value_groups,
            config.seq_len,
            config.head_dim,
            BLOCK_N=config.block_n,
            BLOCK_D=block_d,
        )

    def int4_load():
        _int4_kv_load_kernel[grid](
            k_packed,
            v_packed,
            k_scale,
            v_scale,
            scratch,
            *k_packed.stride(),
            *v_packed.stride(),
            *k_scale.stride(),
            *v_scale.stride(),
            config.n_query_heads,
            config.num_key_value_groups,
            config.seq_len,
            config.head_dim,
            config.group_size,
            BLOCK_N=config.block_n,
            BLOCK_D=block_d,
        )

    def int4_dequant():
        _int4_kv_dequant_kernel[grid](
            k_packed,
            v_packed,
            k_scale,
            v_scale,
            scratch,
            *k_packed.stride(),
            *v_packed.stride(),
            *k_scale.stride(),
            *v_scale.stride(),
            config.n_query_heads,
            config.num_key_value_groups,
            config.seq_len,
            config.head_dim,
            config.group_size,
            BLOCK_N=config.block_n,
            BLOCK_D=block_d,
        )

    for _ in range(config.warmup):
        bf16_load()
        int4_load()
        int4_dequant()
    torch.cuda.synchronize()

    bf16_ms = _record_ms(bf16_load, config.repeats)
    int4_load_ms = _record_ms(int4_load, config.repeats)
    int4_dequant_ms = _record_ms(int4_dequant, config.repeats)
    dequant_overhead_ms = max(0.0, int4_dequant_ms - int4_load_ms)
    load_saved_ms = bf16_ms - int4_load_ms

    logical_bf16_bytes = (
        config.batch_size * config.n_query_heads * config.seq_len * config.head_dim * 2 * torch.finfo(config.dtype).bits / 8
    )
    logical_int4_bytes = config.batch_size * config.n_query_heads * config.seq_len * (
        config.head_dim * 2 * 0.5 + config.groups * 2 * 4
    )
    return [
        {
            "component": "bf16_load_unquantized",
            "time_ms": bf16_ms,
            "delta_vs_bf16_ms": 0.0,
            "speedup_vs_bf16": 1.0,
            "logical_bytes": float(logical_bf16_bytes),
            "note": "Load K/V as unquantized bf16, shaped like decode attention KV reads.",
        },
        {
            "component": "int4_load_only",
            "time_ms": int4_load_ms,
            "delta_vs_bf16_ms": int4_load_ms - bf16_ms,
            "speedup_vs_bf16": bf16_ms / int4_load_ms if int4_load_ms else 0.0,
            "logical_bytes": float(logical_int4_bytes),
            "note": "Load packed INT4 K/V plus scales, no nibble unpack or multiply.",
        },
        {
            "component": "int4_load_plus_dequant",
            "time_ms": int4_dequant_ms,
            "delta_vs_bf16_ms": int4_dequant_ms - bf16_ms,
            "speedup_vs_bf16": bf16_ms / int4_dequant_ms if int4_dequant_ms else 0.0,
            "logical_bytes": float(logical_int4_bytes),
            "note": "Current INT4-style load plus nibble unpack, sign extend, scale multiply.",
        },
        {
            "component": "derived_load_saved",
            "time_ms": load_saved_ms,
            "delta_vs_bf16_ms": load_saved_ms,
            "speedup_vs_bf16": 0.0,
            "logical_bytes": float(logical_bf16_bytes - logical_int4_bytes),
            "note": "bf16_load_unquantized - int4_load_only.",
        },
        {
            "component": "derived_dequant_overhead",
            "time_ms": dequant_overhead_ms,
            "delta_vs_bf16_ms": dequant_overhead_ms,
            "speedup_vs_bf16": 0.0,
            "logical_bytes": 0.0,
            "note": "int4_load_plus_dequant - int4_load_only.",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_timeline_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    values = {str(row["component"]): float(row["time_ms"]) for row in rows}
    bf16 = values["bf16_load_unquantized"]
    int4_load = values["int4_load_only"]
    int4_dequant = values["int4_load_plus_dequant"]
    overhead = max(0.0, int4_dequant - int4_load)
    max_ms = max(bf16, int4_dequant, 1.0)
    width = 900
    left = 210
    bar_w = 620
    row_h = 58
    height = 230

    def x(ms: float) -> float:
        return left + bar_w * ms / max_ms

    def rect(y: int, start: float, end: float, color: str) -> str:
        return f'<rect x="{x(start):.2f}" y="{y}" width="{max(0.0, x(end)-x(start)):.2f}" height="26" fill="{color}"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:14px}.label{font-weight:600}.small{font-size:12px;fill:#444}</style>',
        '<text x="20" y="28" class="label">KV Load / Dequant A-B Kernel Timeline</text>',
        '<text x="20" y="58" class="small">Each row is mean CUDA event time per kernel launch.</text>',
    ]
    rowspec = [
        ("BF16 unquantized load", 85, [(0.0, bf16, "#4c78a8")], bf16),
        ("INT4 load only", 85 + row_h, [(0.0, int4_load, "#59a14f")], int4_load),
        ("INT4 load + dequant", 85 + 2 * row_h, [(0.0, int4_load, "#59a14f"), (int4_load, int4_load + overhead, "#f28e2b")], int4_dequant),
    ]
    for label, y, segments, total in rowspec:
        parts.append(f'<text x="20" y="{y+18}" class="label">{label}</text>')
        for start, end, color in segments:
            parts.append(rect(y, start, end, color))
        parts.append(f'<text x="{x(total)+8:.2f}" y="{y+18}" class="small">{total:.4f} ms</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="205" x2="{left+bar_w}" y2="205" stroke="#777"/>',
            f'<text x="{left}" y="222" class="small">0</text>',
            f'<text x="{left+bar_w-45}" y="222" class="small">{max_ms:.3f} ms</text>',
            '<rect x="650" y="22" width="14" height="14" fill="#59a14f"/><text x="670" y="34" class="small">INT4 load</text>',
            '<rect x="650" y="42" width="14" height="14" fill="#f28e2b"/><text x="670" y="54" class="small">dequant overhead</text>',
            '<rect x="650" y="62" width="14" height="14" fill="#4c78a8"/><text x="670" y="74" class="small">BF16 load</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
