from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.int4_8b_attention import patch_nemotron_h_attention_int4_kv
from hybrid_quant.mamba_state_8b_kernel import (
    patch_nemotron_h_mamba_decode_state_kernel,
    register_mamba_state_kernel_caches,
)
from hybrid_quant.nemotron_8b_decode_eval import layer_groups, make_hybrid_cache, patch_attention_cache_in_blocks
from run_nemotron_8b_decode_degradation import MODEL_PATH, parse_steps


def make_batch_ids(repo: str, dataset: str, batch_size: int, seq_len: int, decode_steps: int) -> torch.Tensor:
    texts = load_wikitext_texts(dataset, split="test")
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[0]
    needed = seq_len + decode_steps + 1
    rows = []
    stride = needed
    for batch_idx in range(batch_size):
        start = batch_idx * stride
        end = start + needed
        if end > ids.numel():
            start = 0
            end = needed
        rows.append(ids[start:end])
    return torch.stack(rows, dim=0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def run_one(
    *,
    repo: str,
    dataset: str,
    kv_quantization: str,
    state_quantization: str,
    batch_size: int,
    sequence_length: int,
    decode_steps: int,
    warmup_steps: int,
    kv_group_size: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
        patch_attention_cache_in_blocks(model)
        if kv_quantization == "int4":
            patch_nemotron_h_attention_int4_kv(model, group_size=kv_group_size)
        if state_quantization == "mx8":
            patch_nemotron_h_mamba_decode_state_kernel(model, group_size=16, stochastic=True, seed=seed)

        config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        mamba_layers, attention_layers, mlp_layers = layer_groups(config)
        mode_by_layer = {idx: "mx8" for idx in mamba_layers} if state_quantization == "mx8" else {}
        ids = make_batch_ids(repo, dataset, batch_size, sequence_length, warmup_steps + decode_steps).to(device)
        cache = make_hybrid_cache(model, batch_size=batch_size)

        with torch.inference_mode():
            torch.cuda.synchronize(device)
            prefill_start = time.perf_counter()
            model(
                input_ids=ids[:, :sequence_length],
                cache_params=cache,
                cache_position=torch.arange(sequence_length, device=device, dtype=torch.long),
                use_cache=True,
                return_dict=True,
            )
            if state_quantization == "mx8":
                register_mamba_state_kernel_caches(model, cache, mode_by_layer)
            torch.cuda.synchronize(device)
            prefill_latency_s = time.perf_counter() - prefill_start

            next_pos = sequence_length
            for _ in range(warmup_steps):
                model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=torch.tensor([next_pos], device=device, dtype=torch.long),
                    use_cache=True,
                    return_dict=True,
                )
                next_pos += 1
            torch.cuda.synchronize(device)

            step_latencies = []
            decode_start = time.perf_counter()
            for _ in range(decode_steps):
                step_start = time.perf_counter()
                model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=torch.tensor([next_pos], device=device, dtype=torch.long),
                    use_cache=True,
                    return_dict=True,
                )
                torch.cuda.synchronize(device)
                step_latencies.append(time.perf_counter() - step_start)
                next_pos += 1
            decode_total_s = time.perf_counter() - decode_start

        peak_memory_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        tokens = batch_size * decode_steps
        return {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "decode_steps": decode_steps,
            "warmup_steps": warmup_steps,
            "status": "ok",
            "kv_quantization": kv_quantization,
            "state_quantization": state_quantization,
            "prefill_latency_s": prefill_latency_s,
            "decode_total_s": decode_total_s,
            "decode_ms_per_step": decode_total_s * 1000.0 / decode_steps,
            "decode_ms_per_token": decode_total_s * 1000.0 / tokens,
            "tokens_per_s": tokens / decode_total_s,
            "step_p50_ms": percentile(step_latencies, 0.50) * 1000.0,
            "step_p90_ms": percentile(step_latencies, 0.90) * 1000.0,
            "peak_memory_gib": peak_memory_gib,
            "error": "",
        }
    finally:
        if model is not None:
            module = __import__(model.__class__.__module__, fromlist=["selective_state_update"])
            if hasattr(module, "_mamba_decode_state_kernel_caches"):
                module._mamba_decode_state_kernel_caches = {}
            if hasattr(model, "_mamba_decode_state_kernel_caches"):
                model._mamba_decode_state_kernel_caches = {}
        del model
        gc.collect()
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "batch_size",
        "sequence_length",
        "decode_steps",
        "warmup_steps",
        "status",
        "kv_quantization",
        "state_quantization",
        "prefill_latency_s",
        "decode_total_s",
        "decode_ms_per_step",
        "decode_ms_per_token",
        "tokens_per_s",
        "step_p50_ms",
        "step_p90_ms",
        "peak_memory_gib",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=MODEL_PATH)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--batch-sizes", default="8,16,32")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument(
        "--quantization-modes",
        default="none,kv_int4,state_mx8,kv_int4_state_mx8",
        help="Comma-separated modes: none, kv_int4, state_mx8, kv_int4_state_mx8.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="results/latency/data/nemotron_8b_seq2048_latency.csv")
    args = parser.parse_args()

    quantization_modes = {
        "none": ("none", "none"),
        "kv_int4": ("int4", "none"),
        "state_mx8": ("none", "mx8"),
        "kv_int4_state_mx8": ("int4", "mx8"),
    }
    selected_modes = [item.strip() for item in args.quantization_modes.split(",") if item.strip()]
    invalid_modes = sorted(set(selected_modes) - set(quantization_modes))
    if invalid_modes:
        raise ValueError(f"Unknown quantization modes: {invalid_modes}; valid modes are {sorted(quantization_modes)}")

    rows: list[dict] = []
    for mode in selected_modes:
        kv_quantization, state_quantization = quantization_modes[mode]
        for batch_size in parse_steps(args.batch_sizes):
            print(
                json.dumps(
                    {
                        "event": "start_batch",
                        "mode": mode,
                        "kv_quantization": kv_quantization,
                        "state_quantization": state_quantization,
                        "batch_size": batch_size,
                    }
                ),
                flush=True,
            )
            try:
                row = run_one(
                    repo=args.repo,
                    dataset=args.dataset,
                    kv_quantization=kv_quantization,
                    state_quantization=state_quantization,
                    batch_size=batch_size,
                    sequence_length=args.sequence_length,
                    decode_steps=args.decode_steps,
                    warmup_steps=args.warmup_steps,
                    kv_group_size=args.kv_group_size,
                    seed=args.seed,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                row = {
                    "batch_size": batch_size,
                    "sequence_length": args.sequence_length,
                    "decode_steps": args.decode_steps,
                    "warmup_steps": args.warmup_steps,
                    "status": "oom",
                    "kv_quantization": kv_quantization,
                    "state_quantization": state_quantization,
                    "prefill_latency_s": "",
                    "decode_total_s": "",
                    "decode_ms_per_step": "",
                    "decode_ms_per_token": "",
                    "tokens_per_s": "",
                    "step_p50_ms": "",
                    "step_p90_ms": "",
                    "peak_memory_gib": "",
                    "error": str(exc).splitlines()[0],
                }
            rows.append(row)
            write_csv(Path(args.output), rows)
            print(json.dumps({"event": "batch_result", "mode": mode, **row}), flush=True)

    print(json.dumps({"event": "outputs", "csv": args.output}), flush=True)


if __name__ == "__main__":
    main()
