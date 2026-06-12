from __future__ import annotations

import torch
import torch.nn.functional as F

from hybrid_quant.int4_8b_attention import (
    _allocate_layer_cache,
    _decode_attention,
    _pack_kv_into_cache,
)


def quant_dequant_ref(x: torch.Tensor, group_size: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    scale = torch.clamp(grouped.abs().amax(dim=-1, keepdim=True) / 7.0, min=1.0e-8)
    q = torch.clamp(torch.round(grouped / scale), -8.0, 7.0)
    return (q * scale).reshape(shape).to(x.dtype)


def main() -> None:
    torch.manual_seed(1234)
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    q_heads = 16
    kv_heads = 4
    groups = q_heads // kv_heads
    seq_len = 257
    head_dim = 128
    group_size = 64

    q = torch.randn(batch, q_heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    positions = torch.arange(seq_len, device=device, dtype=torch.long)

    cache = _allocate_layer_cache(batch, kv_heads, seq_len, head_dim, head_dim // group_size, q.device)
    _pack_kv_into_cache(cache, k, v, positions, group_size)
    out = _decode_attention(q, cache, seq_len, groups, group_size)

    k_ref = quant_dequant_ref(k, group_size).repeat_interleave(groups, dim=1)
    v_ref = quant_dequant_ref(v, group_size).repeat_interleave(groups, dim=1)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=False).squeeze(2)

    diff = (out.float() - ref.float()).abs()
    projected_layout = out.unsqueeze(2).transpose(1, 2).contiguous().view(batch, 1, q_heads * head_dim)
    ref_layout = ref.unsqueeze(2).transpose(1, 2).contiguous().view(batch, 1, q_heads * head_dim)
    layout_diff = (projected_layout.float() - ref_layout.float()).abs()
    print(
        {
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
            "layout_max_abs": layout_diff.max().item(),
            "layout_mean_abs": layout_diff.mean().item(),
            "ref_abs_mean": ref.float().abs().mean().item(),
            "out_abs_mean": out.float().abs().mean().item(),
        }
    )


if __name__ == "__main__":
    main()
