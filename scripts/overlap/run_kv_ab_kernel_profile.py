from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.kv_ab_kernel_profile import KVABConfig, run_kv_ab_profile, write_csv, write_timeline_svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--output-prefix", default="results/overlap/ab_kernel/nemotron_8b_kv_ab_kernel")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config = KVABConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_query_heads=args.n_query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        group_size=args.group_size,
        block_n=args.block_n,
        warmup=args.warmup,
        repeats=args.repeats,
        dtype=dtype,
    )
    rows = run_kv_ab_profile(config)
    prefix = Path(args.output_prefix)
    csv_path = prefix.with_suffix(".csv")
    svg_path = prefix.with_suffix(".svg")
    json_path = prefix.with_suffix(".json")
    write_csv(csv_path, rows)
    write_timeline_svg(svg_path, rows)
    metadata = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_query_heads": args.n_query_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "group_size": args.group_size,
        "block_n": args.block_n,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dtype": args.dtype,
        "csv": str(csv_path),
        "timeline_svg": str(svg_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"event": "outputs", **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
