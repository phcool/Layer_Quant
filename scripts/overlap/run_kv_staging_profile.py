from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.int4_8b_attention import patch_nemotron_h_attention_int4_kv
from hybrid_quant.kv_staging_profile import KVStagingProfiler, patch_profiled_blocks, write_json
from hybrid_quant.nemotron_8b_decode_eval import layer_groups, make_hybrid_cache
from run_nemotron_8b_decode_degradation import MODEL_PATH


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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100.0


def mean_non_null(rows: list[dict], key: str) -> float | None:
    return mean([float(row[key]) for row in rows if row.get(key) not in ("", None)])


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_gap_slowdown_by_step(rows: list[dict]) -> dict[tuple[int, int], dict[str, float | None]]:
    baseline = {
        (int(row["decode_step"]), int(row["attn_layer_id"])): float(row["T_gap_ms"])
        for row in rows
        if row["staging_mode"] == "none"
    }
    out: dict[tuple[int, int], dict[str, float | None]] = {}
    for row in rows:
        if row["staging_mode"] != "dequant":
            continue
        key = (int(row["decode_step"]), int(row["attn_layer_id"]))
        gap = float(row["T_gap_ms"])
        base = baseline.get(key)
        stage = row.get("T_stage_ms")
        stage_ms = None if stage in ("", None) else float(stage)
        slowdown_ms = None if base is None else gap - base
        effective_overlap = None
        if slowdown_ms is not None and stage_ms not in (None, 0):
            effective_overlap = clamp01(1.0 - slowdown_ms / stage_ms)
        out[key] = {
            "gap_slowdown_ms": slowdown_ms,
            "gap_slowdown_pct": pct_change(gap, base),
            "net_stage_cost_ms": None if slowdown_ms is None else max(0.0, slowdown_ms),
            "effective_overlap": effective_overlap,
        }
    return out


def build_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["staging_mode"], int(row["attn_layer_id"])), []).append(row)
    attn_ids = sorted({int(row["attn_layer_id"]) for row in rows})
    out = []
    for attn_layer_id in attn_ids:
        none_rows = grouped.get(("none", attn_layer_id), [])
        dequant_rows = grouped.get(("dequant", attn_layer_id), [])
        sample_rows = dequant_rows or none_rows
        mean_gap_none = mean_non_null(none_rows, "T_gap_ms")
        mean_gap_dequant = mean_non_null(dequant_rows, "T_gap_ms")
        mean_stage = mean_non_null(dequant_rows, "T_stage_ms")
        mean_slowdown_ms = None if mean_gap_dequant is None or mean_gap_none is None else mean_gap_dequant - mean_gap_none
        effective_overlap = None
        if mean_slowdown_ms is not None and mean_stage not in (None, 0):
            effective_overlap = clamp01(1.0 - mean_slowdown_ms / mean_stage)
        out.append(
            {
                "batch_size": sample_rows[0]["batch_size"],
                "context_length": sample_rows[0]["context_length"],
                "attn_layer_id": attn_layer_id,
                "prev_attn_layer_id": sample_rows[0]["prev_attn_layer_id"],
                "steps_none": len(none_rows),
                "steps_dequant": len(dequant_rows),
                "mean_T_gap_none_ms": mean_gap_none,
                "mean_T_gap_dequant_ms": mean_gap_dequant,
                "mean_T_stage_ms": mean_stage,
                "mean_T_attn_none_ms": mean_non_null(none_rows, "T_attn_ms"),
                "mean_T_attn_dequant_ms": mean_non_null(dequant_rows, "T_attn_ms"),
                "mean_wait_before_attention_ms": mean_non_null(dequant_rows, "wait_before_attention_ms"),
                "mean_hideability": mean_non_null(dequant_rows, "hideability"),
                "mean_overlap_ratio_wait_based": mean_non_null(dequant_rows, "overlap_ratio_wait_based"),
                "mean_effective_overlap_wall_clock": effective_overlap,
                "mean_gap_slowdown_ms": mean_slowdown_ms,
                "mean_gap_slowdown_pct": pct_change(mean_gap_dequant, mean_gap_none),
                "mean_net_stage_cost_ms": None if mean_slowdown_ms is None else max(0.0, mean_slowdown_ms),
            }
        )
    return out


def cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def run_mode(
    *,
    repo: str,
    dataset: str,
    staging_mode: str,
    batch_size: int,
    context_length: int,
    decode_steps: int,
    warmup_steps: int,
    kv_group_size: int,
    seed: int,
    enable_nvtx: bool,
    debug_stage_repeats: int,
    cuda_profiler_capture: bool,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
        patch_nemotron_h_attention_int4_kv(model, group_size=kv_group_size)

        config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        mamba_layers, attention_layers, mlp_layers = layer_groups(config)
        layer_types = {idx: "mamba" for idx in mamba_layers}
        layer_types.update({idx: "attention" for idx in attention_layers})
        layer_types.update({idx: "mlp" for idx in mlp_layers})

        profiler = KVStagingProfiler(
            attention_layer_ids=attention_layers,
            layer_types=layer_types,
            staging_mode=staging_mode,
            batch_size=batch_size,
            context_length=context_length,
            kv_group_size=kv_group_size,
            enable_nvtx=enable_nvtx,
        )
        patch_profiled_blocks(model, profiler)

        ids = make_batch_ids(repo, dataset, batch_size, context_length, warmup_steps + decode_steps).to(device)
        positions = torch.arange(context_length + warmup_steps + decode_steps + 1, device=device, dtype=torch.long)
        cache = make_hybrid_cache(model, batch_size=batch_size)

        with torch.inference_mode():
            model(
                input_ids=ids[:, :context_length],
                cache_params=cache,
                cache_position=positions[:context_length],
                use_cache=True,
                return_dict=True,
            )
            next_pos = context_length
            for _ in range(warmup_steps):
                model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=positions[next_pos : next_pos + 1],
                    use_cache=True,
                    return_dict=True,
                )
                next_pos += 1
            if staging_mode == "dequant":
                profiler.cache_params = cache
                profiler.device = device
                if profiler.staging_stream is None:
                    profiler.staging_stream = torch.cuda.Stream(device=device)
                profiler.allocate_stage_buffers(cache, context_length + warmup_steps + decode_steps + 1)
                for attn_layer_id in attention_layers:
                    profiler.debug_stage_repeated(attn_layer_id, debug_stage_repeats)
                profiler.scheduled_stages = {}
                profiler.cache_params = None
            torch.cuda.synchronize(device)

            if cuda_profiler_capture:
                cuda_profiler_start()
            for step_idx in range(decode_steps):
                profiler.start_step(step_idx, next_pos, cache, device)
                model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=positions[next_pos : next_pos + 1],
                    use_cache=True,
                    return_dict=True,
                )
                profiler.end_step()
                next_pos += 1
            if cuda_profiler_capture:
                cuda_profiler_stop()
            torch.cuda.synchronize(device)

        metadata = {
            "staging_mode": staging_mode,
            "batch_size": batch_size,
            "context_length": context_length,
            "decode_steps": decode_steps,
            "warmup_steps": warmup_steps,
            "kv_group_size": kv_group_size,
            "attention_layer_ids": attention_layers,
            "mamba_layer_ids": mamba_layers,
            "mlp_layer_ids": mlp_layers,
            "layer_pattern": config.hybrid_override_pattern,
            "previous_attention_by_layer": {
                attn: attention_layers[idx - 1] if idx > 0 else None for idx, attn in enumerate(attention_layers)
            },
        }
        return profiler.attention_rows(), profiler.layer_rows(), profiler.debug_stage_rows(), metadata
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=MODEL_PATH)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--staging-modes", default="none,dequant")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument("--cuda-profiler-capture", action="store_true")
    parser.add_argument("--debug-stage-repeats", type=int, default=10)
    parser.add_argument("--output-prefix", default="results/overlap/data/nemotron_8b_kv_staging_profile")
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    modes = [item.strip() for item in args.staging_modes.split(",") if item.strip()]
    if not modes:
        raise ValueError("At least one staging mode is required")

    all_attention_rows: list[dict] = []
    all_layer_rows: list[dict] = []
    all_debug_stage_rows: list[dict] = []
    metadata: dict | None = None
    for mode in modes:
        print(json.dumps({"event": "start_mode", "staging_mode": mode}), flush=True)
        attention_rows, layer_rows, debug_stage_rows, mode_metadata = run_mode(
            repo=args.repo,
            dataset=args.dataset,
            staging_mode=mode,
            batch_size=args.batch_size,
            context_length=args.context_length,
            decode_steps=args.decode_steps,
            warmup_steps=args.warmup_steps,
            kv_group_size=args.kv_group_size,
            seed=args.seed,
            enable_nvtx=args.enable_nvtx,
            debug_stage_repeats=args.debug_stage_repeats,
            cuda_profiler_capture=args.cuda_profiler_capture,
        )
        all_attention_rows.extend(attention_rows)
        all_layer_rows.extend(layer_rows)
        all_debug_stage_rows.extend(debug_stage_rows)
        metadata = metadata or mode_metadata
        print(
            json.dumps(
                {
                    "event": "finish_mode",
                    "staging_mode": mode,
                    "attention_rows": len(attention_rows),
                    "layer_rows": len(layer_rows),
                }
            ),
            flush=True,
        )

    gap_slowdown_by_step = build_gap_slowdown_by_step(all_attention_rows)
    for row in all_attention_rows:
        if row["staging_mode"] == "dequant":
            slowdown = gap_slowdown_by_step.get((int(row["decode_step"]), int(row["attn_layer_id"])))
            row["gap_slowdown_ms"] = None if slowdown is None else slowdown["gap_slowdown_ms"]
            row["gap_slowdown_pct"] = None if slowdown is None else slowdown["gap_slowdown_pct"]
            row["net_stage_cost_ms"] = None if slowdown is None else slowdown["net_stage_cost_ms"]
            row["effective_overlap_wall_clock"] = None if slowdown is None else slowdown["effective_overlap"]
        else:
            row["gap_slowdown_ms"] = None
            row["gap_slowdown_pct"] = None
            row["net_stage_cost_ms"] = None
            row["effective_overlap_wall_clock"] = None

    summary_rows = build_summary(all_attention_rows)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_prefix.with_name(output_prefix.name + "_metadata.json"), metadata or {})
    write_jsonl(output_prefix.with_name(output_prefix.name + "_attention_raw.jsonl"), all_attention_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_attention_raw.csv"), all_attention_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_layer_raw.csv"), all_layer_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_stage_debug.csv"), all_debug_stage_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), summary_rows)
    print(
        json.dumps(
            {
                "event": "outputs",
                "metadata": str(output_prefix.with_name(output_prefix.name + "_metadata.json")),
                "attention_raw_jsonl": str(output_prefix.with_name(output_prefix.name + "_attention_raw.jsonl")),
                "attention_raw_csv": str(output_prefix.with_name(output_prefix.name + "_attention_raw.csv")),
                "layer_raw_csv": str(output_prefix.with_name(output_prefix.name + "_layer_raw.csv")),
                "stage_debug_csv": str(output_prefix.with_name(output_prefix.name + "_stage_debug.csv")),
                "summary_csv": str(output_prefix.with_name(output_prefix.name + "_summary.csv")),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
