from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.low_freq_attention import (
    LowFreqAttentionController,
    build_delta_quantile_schedule,
    build_layerwise_schedule,
    patch_low_freq_attention_blocks,
)
from hybrid_quant.nemotron_8b_decode_eval import layer_groups, make_hybrid_cache

MODEL_PATH = "/scratch2/wl730/models/nemotron-h-8b"


def parse_layer_intervals(text: str) -> dict[int, int]:
    if not text:
        return {}
    intervals = {}
    for item in text.split(","):
        if not item.strip():
            continue
        layer, interval = item.split(":")
        intervals[int(layer)] = int(interval)
    return intervals


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


def make_controller(args, attention_layers: list[int], schedule: dict[int, set[int]] | None = None) -> LowFreqAttentionController:
    run_layers = None
    if getattr(args, "run_layers", ""):
        run_layers = {int(item) for item in args.run_layers.split(",") if item.strip()}
    return LowFreqAttentionController(
        attention_layers=attention_layers,
        mode=args.experiment,
        interval=args.interval,
        layer_intervals=parse_layer_intervals(getattr(args, "layer_intervals", "")),
        run_layers=run_layers,
        reuse_correction=getattr(args, "reuse_correction", False),
        correction_decay=getattr(args, "correction_decay", 1.0),
        replay_schedule=schedule,
        max_skip=getattr(args, "max_skip", None),
    )


def load_model(repo: str):
    return AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()


def prefill_and_warmup(model, ids, positions, cache, sequence_length: int, warmup_steps: int, controller):
    with torch.inference_mode():
        model(
            input_ids=ids[:, :sequence_length],
            cache_params=cache,
            cache_position=positions[:sequence_length],
            use_cache=True,
            return_dict=True,
        )
        next_pos = sequence_length
        for warm_idx in range(warmup_steps):
            controller.begin_decode_step(-(warmup_steps - warm_idx))
            model(
                input_ids=ids[:, next_pos : next_pos + 1],
                cache_params=cache,
                cache_position=positions[next_pos : next_pos + 1],
                use_cache=True,
                return_dict=True,
            )
            next_pos += 1
        torch.cuda.synchronize()
    return next_pos


def run_latency_pass(args, schedule: dict[int, set[int]] | None = None) -> tuple[dict, list[dict]]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = None
    try:
        model = load_model(args.repo)
        config = AutoConfig.from_pretrained(args.repo, trust_remote_code=True)
        _, attention_layers, _ = layer_groups(config)
        controller = make_controller(args, attention_layers, schedule=schedule)
        patch_low_freq_attention_blocks(model, controller)
        ids = make_batch_ids(args.repo, args.dataset, args.batch_size, args.sequence_length, args.warmup_steps + args.decode_steps).to(device)
        positions = torch.arange(args.sequence_length + args.warmup_steps + args.decode_steps + 1, device=device, dtype=torch.long)
        cache = make_hybrid_cache(model, batch_size=args.batch_size)
        next_pos = prefill_and_warmup(model, ids, positions, cache, args.sequence_length, args.warmup_steps, controller)

        step_events = []
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            decode_start.record()
            for step_idx in range(args.decode_steps):
                controller.begin_decode_step(step_idx)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=positions[next_pos : next_pos + 1],
                    use_cache=True,
                    return_dict=True,
                )
                end.record()
                step_events.append((start, end))
                next_pos += 1
            decode_end.record()
            torch.cuda.synchronize()
        step_ms = [start.elapsed_time(end) for start, end in step_events]
        total_ms = decode_start.elapsed_time(decode_end)
        attention_runs = sum(row["runs"] for row in controller.summary_rows())
        attention_skips = sum(row["skips"] for row in controller.summary_rows())
        total_attention_slots = attention_runs + attention_skips
        row = {
            "experiment": args.experiment,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "decode_steps": args.decode_steps,
            "warmup_steps": args.warmup_steps,
            "interval": args.interval,
            "reuse_correction": getattr(args, "reuse_correction", False),
            "correction_decay": getattr(args, "correction_decay", 1.0),
            "layer_intervals": getattr(args, "layer_intervals", ""),
            "run_layers": getattr(args, "run_layers", ""),
            "decode_total_ms": total_ms,
            "decode_ms_per_step": total_ms / args.decode_steps,
            "step_p50_ms": percentile(step_ms, 0.50),
            "step_p90_ms": percentile(step_ms, 0.90),
            "attention_runs": attention_runs,
            "attention_skips": attention_skips,
            "attention_run_fraction": attention_runs / total_attention_slots if total_attention_slots else 0.0,
        }
        return row, controller.summary_rows()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_quality_pass(args, schedule: dict[int, set[int]] | None = None) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    def collect_logits(low_freq: bool):
        model = load_model(args.repo)
        config = AutoConfig.from_pretrained(args.repo, trust_remote_code=True)
        _, attention_layers, _ = layer_groups(config)
        controller = make_controller(args, attention_layers, schedule=schedule)
        if low_freq:
            patch_low_freq_attention_blocks(model, controller)
        else:
            from hybrid_quant.nemotron_8b_decode_eval import patch_attention_cache_in_blocks

            patch_attention_cache_in_blocks(model)
        ids = make_batch_ids(args.repo, args.dataset, args.batch_size, args.sequence_length, args.warmup_steps + args.decode_steps).to(device)
        positions = torch.arange(args.sequence_length + args.warmup_steps + args.decode_steps + 1, device=device, dtype=torch.long)
        cache = make_hybrid_cache(model, batch_size=args.batch_size)
        next_pos = prefill_and_warmup(model, ids, positions, cache, args.sequence_length, args.warmup_steps, controller)
        logits = []
        targets = []
        with torch.inference_mode():
            for step_idx in range(args.decode_steps):
                controller.begin_decode_step(step_idx)
                out = model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=positions[next_pos : next_pos + 1],
                    use_cache=True,
                    return_dict=True,
                )
                logits.append(out.logits[:, -1].detach().float().cpu())
                targets.append(ids[:, next_pos + 1].detach().cpu())
                next_pos += 1
            torch.cuda.synchronize()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return torch.stack(logits, dim=1), torch.stack(targets, dim=1)

    baseline_logits, targets = collect_logits(low_freq=False)
    approx_logits, _ = collect_logits(low_freq=True)
    vocab = baseline_logits.shape[-1]
    baseline_ce = F.cross_entropy(baseline_logits.reshape(-1, vocab), targets.reshape(-1)).item()
    approx_ce = F.cross_entropy(approx_logits.reshape(-1, vocab), targets.reshape(-1)).item()
    baseline_logp = F.log_softmax(baseline_logits, dim=-1)
    approx_logp = F.log_softmax(approx_logits, dim=-1)
    baseline_p = baseline_logp.exp()
    kl = (baseline_p * (baseline_logp - approx_logp)).sum(dim=-1).mean().item()
    top1_match = (baseline_logits.argmax(dim=-1) == approx_logits.argmax(dim=-1)).float().mean().item()
    return {
        "experiment": args.experiment,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "decode_steps": args.decode_steps,
        "baseline_ce": baseline_ce,
        "approx_ce": approx_ce,
        "delta_ce": approx_ce - baseline_ce,
        "baseline_ppl": math.exp(baseline_ce),
        "approx_ppl": math.exp(approx_ce),
        "kl_to_baseline": kl,
        "top1_match": top1_match,
    }


def collect_hidden_delta_stats(args) -> tuple[dict[int, list[float]], list[int]]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = None
    try:
        model = load_model(args.repo)
        config = AutoConfig.from_pretrained(args.repo, trust_remote_code=True)
        _, attention_layers, _ = layer_groups(config)
        controller = LowFreqAttentionController(
            attention_layers=attention_layers,
            interval=1,
            collect_hidden_delta=True,
        )
        patch_low_freq_attention_blocks(model, controller)
        ids = make_batch_ids(args.repo, args.dataset, args.batch_size, args.sequence_length, args.warmup_steps + args.decode_steps).to(device)
        positions = torch.arange(args.sequence_length + args.warmup_steps + args.decode_steps + 1, device=device, dtype=torch.long)
        cache = make_hybrid_cache(model, batch_size=args.batch_size)
        next_pos = prefill_and_warmup(model, ids, positions, cache, args.sequence_length, args.warmup_steps, controller)
        with torch.inference_mode():
            for step_idx in range(args.decode_steps):
                controller.begin_decode_step(step_idx)
                model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=positions[next_pos : next_pos + 1],
                    use_cache=True,
                    return_dict=True,
                )
                next_pos += 1
            torch.cuda.synchronize()
        stats = {}
        for layer_idx, state in controller.states.items():
            stats[layer_idx] = list(state.hidden_deltas)
        return stats, attention_layers
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def add_common_args(parser: argparse.ArgumentParser, experiment: str) -> None:
    parser.set_defaults(experiment=experiment)
    parser.add_argument("--repo", default=MODEL_PATH)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--layer-intervals", default="")
    parser.add_argument("--run-layers", default="")
    parser.add_argument("--reuse-correction", action="store_true")
    parser.add_argument("--correction-decay", type=float, default=1.0)
    parser.add_argument("--max-skip", type=int, default=None)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)


def run_experiment(args, schedule: dict[int, set[int]] | None = None) -> None:
    output_dir = Path(args.output_dir)
    latency_row, layer_rows = run_latency_pass(args, schedule=schedule)
    write_dict_csv(output_dir / "latency.csv", [latency_row])
    write_dict_csv(output_dir / "layer_summary.csv", layer_rows)
    payload = {"latency": latency_row, "layer_summary": layer_rows}
    if args.quality:
        quality = run_quality_pass(args, schedule=schedule)
        write_dict_csv(output_dir / "quality.csv", [quality])
        payload["quality"] = quality
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "output_dir": str(output_dir), **latency_row}, indent=2), flush=True)


def calibrated_schedule(args) -> dict[int, set[int]]:
    stats, attention_layers = collect_hidden_delta_stats(args)
    return build_delta_quantile_schedule(
        stats,
        attention_layers,
        args.decode_steps,
        keep_fraction=args.keep_fraction,
        max_skip=args.max_skip or args.interval,
    )


def layerwise_schedule_from_args(args) -> dict[int, set[int]]:
    config = AutoConfig.from_pretrained(args.repo, trust_remote_code=True)
    _, attention_layers, _ = layer_groups(config)
    return build_layerwise_schedule(
        attention_layers,
        args.decode_steps,
        parse_layer_intervals(args.layer_intervals),
        args.interval,
    )
