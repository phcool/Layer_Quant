from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.nemotron_8b_decode_eval import DeviceList, layer_groups, patch_attention_cache_in_blocks
from scripts.quamba_compat.run_nemotron_quamba_full_w8a8 import (
    CALIB_SOURCE_LOCAL,
    CALIB_SOURCE_QUAMBA_PILE,
    CALIB_SOURCE_WIKITEXT,
    build_reorder_params,
    calibrate_quamba2_scales,
    configure_nemotron_mamba_layers,
    ensure_quamba_on_path,
    get_channel_stats_for_reorder,
    load_calibration_dataset,
    quantize_all_mamba_layers,
    reorder_nemotron_mamba_layers,
)

MODEL_PATH = "/scratch2/wl730/models/nemotron-h-8b"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def make_eval_ids(
    tokenizer,
    dataset: str,
    batch_size: int,
    context_length: int,
    warmup_steps: int,
    decode_steps: int,
) -> torch.Tensor:
    texts = load_wikitext_texts(dataset, split="test")
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[0]
    needed = context_length + warmup_steps + decode_steps + 1
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


def make_cache(model, batch_size: int):
    module = __import__(model.__class__.__module__, fromlist=["HybridMambaAttentionDynamicCache"])
    cache_cls = getattr(module, "HybridMambaAttentionDynamicCache")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cache = cache_cls(model.config, batch_size, dtype, device=device)
    cache.conv_kernel_size = model.config.conv_kernel
    cache.conv_states = DeviceList(cache.conv_states, device)
    cache.ssm_states = DeviceList(cache.ssm_states, device)
    return cache


@torch.no_grad()
def prepare_quamba_w8a8_model(
    model,
    tokenizer,
    *,
    num_calib_samples: int,
    calib_seq_len: int,
    use_group_heads: bool,
    use_had_transform: bool,
    calib_source: str = "local_prompts",
) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(model.name_or_path, trust_remote_code=True)
    mamba_layers, attention_layers, mlp_layers = layer_groups(config)
    started = time.time()

    configured_layers = configure_nemotron_mamba_layers(
        model,
        mamba_layers,
        use_had_transform=use_had_transform,
    )
    calibration_dataset = load_calibration_dataset(calib_source)
    reorder_params = None
    grouped_available = False
    if use_group_heads:
        channel_stats = get_channel_stats_for_reorder(
            model,
            tokenizer,
            mamba_layers,
            num_calib_samples,
            calib_seq_len,
            calibration_dataset,
        )
        reorder_params = build_reorder_params(model, mamba_layers, channel_stats)
        reorder_nemotron_mamba_layers(model, mamba_layers, reorder_params)
        grouped_available = True

    act_scales = calibrate_quamba2_scales(
        model,
        tokenizer,
        mamba_layers,
        reorder_params,
        num_calib_samples,
        calib_seq_len,
        calibration_dataset,
    )
    layer_summaries = quantize_all_mamba_layers(
        model,
        mamba_layers,
        act_scales,
        use_had_transform=use_had_transform,
        require_grouped=use_group_heads,
    )

    replaced_mamba_layers = [
        idx
        for idx in mamba_layers
        if model.backbone.layers[idx].mixer.__class__.__name__ == "W8A8QMamba2"
    ]
    if replaced_mamba_layers != mamba_layers:
        raise RuntimeError(
            f"Not all Mamba layers were replaced: expected {mamba_layers}, got {replaced_mamba_layers}"
        )

    return {
        "configured_layers": configured_layers,
        "replaced_mamba_layers": replaced_mamba_layers,
        "attention_layers": attention_layers,
        "mlp_layers": mlp_layers,
        "layer_summaries": layer_summaries,
        "num_calib_samples": num_calib_samples,
        "calib_seq_len": calib_seq_len,
        "calib_source": calib_source,
        "use_group_heads": use_group_heads,
        "grouped_available": grouped_available,
        "use_had_transform": use_had_transform,
        "prepare_seconds": time.time() - started,
    }


@torch.no_grad()
def run_decode(
    model,
    ids: torch.Tensor,
    *,
    context_length: int,
    warmup_steps: int,
    decode_steps: int,
    collect_logits: bool,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    device = next(model.parameters()).device
    ids = ids.to(device)
    positions = torch.arange(ids.shape[1], device=device, dtype=torch.long)
    cache = make_cache(model, batch_size=ids.shape[0])
    torch.cuda.reset_peak_memory_stats(device)

    logits_rows = []
    target_rows = []
    step_events = []

    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)

    prefill_start.record(torch.cuda.current_stream(device))
    outputs = model(
        input_ids=ids[:, :context_length],
        cache_params=cache,
        cache_position=positions[:context_length],
        use_cache=True,
        return_dict=True,
    )
    prefill_end.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    prefill_latency_s = prefill_start.elapsed_time(prefill_end) / 1000.0
    del outputs

    next_pos = context_length
    for _ in range(warmup_steps):
        outputs = model(
            input_ids=ids[:, next_pos : next_pos + 1],
            cache_params=cache,
            cache_position=positions[next_pos : next_pos + 1],
            use_cache=True,
            return_dict=True,
        )
        del outputs
        next_pos += 1
    torch.cuda.synchronize(device)

    total_start.record(torch.cuda.current_stream(device))
    for _ in range(decode_steps):
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record(torch.cuda.current_stream(device))
        outputs = model(
            input_ids=ids[:, next_pos : next_pos + 1],
            cache_params=cache,
            cache_position=positions[next_pos : next_pos + 1],
            use_cache=True,
            return_dict=True,
        )
        step_end.record(torch.cuda.current_stream(device))
        step_events.append((step_start, step_end))
        if collect_logits:
            logits_rows.append(outputs.logits[:, -1, :].detach().float().cpu())
            target_rows.append(ids[:, next_pos + 1].detach().cpu())
        del outputs
        next_pos += 1
    total_end.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)

    decode_total_s = total_start.elapsed_time(total_end) / 1000.0
    step_latencies_ms = [start.elapsed_time(end) for start, end in step_events]
    tokens = ids.shape[0] * decode_steps
    row = {
        "status": "ok",
        "batch_size": ids.shape[0],
        "context_length": context_length,
        "warmup_steps": warmup_steps,
        "decode_steps": decode_steps,
        "prefill_latency_s": prefill_latency_s,
        "decode_total_s": decode_total_s,
        "decode_ms_per_step": decode_total_s * 1000.0 / decode_steps,
        "decode_ms_per_token": decode_total_s * 1000.0 / tokens,
        "tokens_per_s": tokens / decode_total_s,
        "step_p50_ms": percentile(step_latencies_ms, 0.50),
        "step_p90_ms": percentile(step_latencies_ms, 0.90),
        "step_min_ms": min(step_latencies_ms),
        "step_max_ms": max(step_latencies_ms),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    logits = torch.stack(logits_rows, dim=1) if collect_logits else None
    targets = torch.stack(target_rows, dim=1) if collect_logits else None
    del cache
    torch.cuda.empty_cache()
    return row, logits, targets


def compute_quality(
    logits: torch.Tensor,
    targets: torch.Tensor,
    baseline_logits: torch.Tensor | None,
    *,
    chunk_steps: int = 4,
) -> dict[str, float]:
    total_tokens = targets.numel()
    ce_sum = 0.0
    kl_sum = 0.0
    top1_sum = 0.0
    for start in range(0, logits.shape[1], chunk_steps):
        end = min(start + chunk_steps, logits.shape[1])
        chunk_logits = logits[:, start:end, :].float()
        chunk_targets = targets[:, start:end].reshape(-1)
        ce = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.shape[-1]),
            chunk_targets,
            reduction="sum",
        )
        ce_sum += float(ce.item())
        if baseline_logits is not None:
            base_chunk = baseline_logits[:, start:end, :].float()
            base_log_probs = F.log_softmax(base_chunk, dim=-1)
            log_probs = F.log_softmax(chunk_logits, dim=-1)
            kl = (base_log_probs.exp() * (base_log_probs - log_probs)).sum(dim=-1).sum()
            kl_sum += float(kl.item())
            top1 = (base_chunk.argmax(dim=-1) == chunk_logits.argmax(dim=-1)).float().sum()
            top1_sum += float(top1.item())
    ce_mean = ce_sum / total_tokens
    row = {
        "ce": ce_mean,
        "ppl": math.exp(ce_mean),
    }
    if baseline_logits is not None:
        row["kl_vs_baseline"] = kl_sum / total_tokens
        row["top1_match_vs_baseline"] = top1_sum / total_tokens
    else:
        row["kl_vs_baseline"] = 0.0
        row["top1_match_vs_baseline"] = 1.0
    return row


def load_model(model_path: str, mode: str):
    dtype = torch.bfloat16 if mode == "baseline_bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).cuda().eval()
    patch_attention_cache_in_blocks(model)
    return model


def cleanup_model(model) -> None:
    if model is not None:
        for layer in getattr(getattr(model, "backbone", None), "layers", []):
            mixer = getattr(layer, "mixer", None)
            if hasattr(mixer, "_nemotron_cache_by_owner"):
                mixer._nemotron_cache_by_owner.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Quamba-aligned W8A8 Nemotron-H Mamba replacement.")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--output-root", default="results/quamba_compat/round5_latency_ppl")
    parser.add_argument("--modes", default="baseline_bf16,quamba_w8a8_mamba")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--latency-batch-size", type=int, default=8)
    parser.add_argument("--quality-batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--quality-decode-steps", type=int, default=64)
    parser.add_argument("--num-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seq-len", type=int, default=128)
    parser.add_argument(
        "--calib-source",
        choices=[CALIB_SOURCE_LOCAL, CALIB_SOURCE_QUAMBA_PILE, CALIB_SOURCE_WIKITEXT],
        default=CALIB_SOURCE_LOCAL,
        help="Calibration source. quamba_pile matches Quamba's default monology/pile-uncopyrighted val split.",
    )
    parser.add_argument("--no-group-heads", action="store_true")
    parser.add_argument("--no-had", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    ensure_quamba_on_path()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    latency_ids = make_eval_ids(
        tokenizer,
        args.dataset,
        args.latency_batch_size,
        args.context_length,
        args.warmup_steps,
        args.decode_steps,
    )
    quality_ids = make_eval_ids(
        tokenizer,
        args.dataset,
        args.quality_batch_size,
        args.context_length,
        0,
        args.quality_decode_steps,
    )

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    latency_rows = []
    quality_rows = []
    prepare_rows = []
    baseline_logits = None
    baseline_targets = None

    for mode in modes:
        print(json.dumps({"event": "start_mode", "mode": mode}), flush=True)
        model = None
        try:
            model = load_model(args.model, "baseline_bf16" if mode == "baseline_bf16" else "quamba")
            prepare_meta: dict[str, Any] = {"mode": mode, "prepare_seconds": 0.0}
            if mode == "quamba_w8a8_mamba":
                prepare_meta.update(
                    prepare_quamba_w8a8_model(
                        model,
                        tokenizer,
                        num_calib_samples=args.num_calib_samples,
                        calib_seq_len=args.calib_seq_len,
                        use_group_heads=not args.no_group_heads,
                        use_had_transform=not args.no_had,
                        calib_source=args.calib_source,
                    )
                )
            elif mode != "baseline_bf16":
                raise ValueError(f"Unsupported mode: {mode}")
            prepare_rows.append(
                {
                    key: value
                    for key, value in prepare_meta.items()
                    if key not in {"layer_summaries", "configured_layers", "replaced_mamba_layers"}
                }
            )
            (output_root / f"{mode}_prepare.json").write_text(
                json.dumps(prepare_meta, indent=2, ensure_ascii=False) + "\n"
            )

            latency_row, _, _ = run_decode(
                model,
                latency_ids,
                context_length=args.context_length,
                warmup_steps=args.warmup_steps,
                decode_steps=args.decode_steps,
                collect_logits=False,
            )
            latency_row["mode"] = mode
            latency_rows.append(latency_row)
            print(json.dumps({"event": "latency_done", **latency_row}), flush=True)

            quality_latency_row, logits, targets = run_decode(
                model,
                quality_ids,
                context_length=args.context_length,
                warmup_steps=0,
                decode_steps=args.quality_decode_steps,
                collect_logits=True,
            )
            if mode == "baseline_bf16":
                baseline_logits = logits
                baseline_targets = targets
                quality = compute_quality(logits, targets, None)
            else:
                if baseline_logits is None or baseline_targets is None:
                    raise RuntimeError("baseline_bf16 must run before quantized modes to compute KL/top1.")
                quality = compute_quality(logits, targets, baseline_logits)
            quality.update(
                {
                    "mode": mode,
                    "batch_size": args.quality_batch_size,
                    "context_length": args.context_length,
                    "decode_steps": args.quality_decode_steps,
                    "quality_prefill_latency_s": quality_latency_row["prefill_latency_s"],
                    "quality_decode_ms_per_step": quality_latency_row["decode_ms_per_step"],
                }
            )
            quality_rows.append(quality)
            print(json.dumps({"event": "quality_done", **quality}), flush=True)
        except Exception as exc:
            error_row = {
                "mode": mode,
                "status": "error",
                "error": repr(exc),
            }
            latency_rows.append(error_row)
            quality_rows.append(error_row)
            print(json.dumps({"event": "mode_error", **error_row}), flush=True)
            raise
        finally:
            cleanup_model(model)

    write_csv(output_root / "latency.csv", latency_rows)
    write_csv(output_root / "quality.csv", quality_rows)
    write_csv(output_root / "prepare.csv", prepare_rows)
    summary = {
        "args": vars(args),
        "latency": latency_rows,
        "quality": quality_rows,
        "prepare": prepare_rows,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "done", "output_root": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
