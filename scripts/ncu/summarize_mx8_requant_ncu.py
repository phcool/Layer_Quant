from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


STATIC_METRICS = {
    "duration_ns": [r"^duration$", r"gpu__time_duration"],
    "sm_throughput_pct": [r"compute \\(sm\\) throughput", r"sm__throughput.*pct"],
    "dram_throughput_pct": [r"^dram throughput$", r"dram__throughput.*pct"],
    "memory_throughput_pct": [r"^memory throughput$"],
    "memory_throughput_bps": [r"^memory throughput$"],
    "l1tex_throughput_pct": [r"l1/tex cache throughput"],
    "l2_throughput_pct": [r"l2 cache throughput"],
    "l1tex_hit_rate_pct": [r"l1/tex hit rate"],
    "l2_hit_rate_pct": [r"l2 hit rate"],
    "theoretical_occupancy_pct": [r"theoretical occupancy"],
    "achieved_occupancy_pct": [r"achieved occupancy"],
    "active_warps_per_sm": [r"achieved active warps per sm"],
    "eligible_warps_per_scheduler": [r"eligible warps per scheduler"],
    "active_warps_per_scheduler": [r"active warps per scheduler"],
    "issued_warp_per_scheduler": [r"issued warp per scheduler"],
    "one_or_more_eligible_pct": [r"one or more eligible"],
    "no_eligible_pct": [r"no eligible"],
    "warp_cycles_per_issued_inst": [r"warp cycles per issued instruction"],
    "warp_cycles_per_executed_inst": [r"warp cycles per executed instruction"],
    "avg_active_threads_per_warp": [r"avg\\. active threads per warp"],
    "avg_not_predicated_off_threads_per_warp": [r"avg\\. not predicated off threads per warp"],
    "block_size": [r"^block size$"],
    "grid_size": [r"^grid size$"],
    "registers_per_thread": [r"registers per thread"],
    "shared_mem_config_bytes": [r"shared memory configuration size"],
    "driver_shared_mem_per_block_bytes": [r"driver shared memory per block"],
    "dynamic_shared_mem_per_block_bytes": [r"dynamic shared memory per block"],
    "static_shared_mem_per_block_bytes": [r"static shared memory per block"],
    "sms": [r"^# sms$"],
    "threads": [r"^threads$"],
    "waves_per_sm": [r"waves per sm"],
    "tensor_active_pct": [r"pipe_tensor.*active.*pct", r"tensor.*active.*pct"],
    "tensor_inst": [r"inst.*pipe_tensor"],
    "integer_inst": [r"inst.*integer", r"thread_inst_executed.*integer"],
    "local_load_bytes": [r"mem_local_op_ld", r"local.*load.*bytes"],
    "local_store_bytes": [r"mem_local_op_st", r"local.*store.*bytes"],
}


STALL_RE = re.compile(r"warp_issue_stalled_([a-z_]+)_per_warp_active\\.pct")


def parse_float(value: str) -> float | None:
    cleaned = value.strip().strip('"').replace(",", "")
    if not cleaned or cleaned.lower() in {"n/a", "nan", "--"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip('"').lower())


def metric_bucket(metric_name: str, metric_unit: str) -> str | None:
    lowered = norm(metric_name)
    stall = STALL_RE.search(lowered)
    if stall:
        return f"stall_{stall.group(1)}_pct"
    for bucket, patterns in STATIC_METRICS.items():
        if bucket == "memory_throughput_pct" and metric_unit != "%":
            continue
        if bucket == "memory_throughput_bps" and metric_unit != "byte/second":
            continue
        for pattern in patterns:
            if re.search(pattern, lowered):
                return bucket
    return None


def read_ncu_csv(path: Path) -> list[dict[str, float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if '"Metric Name"' in line and '"Metric Value"' in line:
            header_idx = idx
            break
    if header_idx is None:
        return []

    launches: dict[str, dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader(lines[header_idx:]):
        launch_id = row.get("ID") or str(len(launches))
        metric = row.get("Metric Name") or ""
        unit = row.get("Metric Unit") or ""
        value = row.get("Metric Value") or ""
        bucket = metric_bucket(metric, unit)
        parsed = parse_float(value)
        if bucket is None or parsed is None:
            continue
        launches[launch_id][bucket] = parsed
    return [launches[key] for key in sorted(launches, key=lambda x: int(x) if x.isdigit() else x)]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_mode(mode: str, path: Path) -> dict:
    launches = read_ncu_csv(path)
    keys = sorted({key for launch in launches for key in launch})
    out = {"mode": mode, "captured_launches": len(launches)}
    for key in keys:
        vals = [launch[key] for launch in launches if key in launch]
        out[f"mean_{key}"] = mean(vals)
    duration = out.get("mean_duration_ns")
    if duration is not None:
        out["mean_duration_ms"] = duration / 1.0e6
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/ncu/data")
    parser.add_argument("--output", default="results/ncu/data/nemotron_8b_mx8_requant_ncu_summary.csv")
    parser.add_argument("--modes", default="state_mx8,kv_int4_state_mx8")
    parser.add_argument("--template", default="nemotron_8b_{mode}_mx8_requant_ncu.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = [
        summarize_mode(mode, input_dir / args.template.format(mode=mode))
        for mode in [item.strip() for item in args.modes.split(",") if item.strip()]
    ]
    write_csv(Path(args.output), rows)
    print(args.output)


if __name__ == "__main__":
    main()
