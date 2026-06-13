from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


MODE_ORDER = ["none", "kv_int4", "state_mx8", "kv_int4_state_mx8"]


def norm(text: str) -> str:
    return text.strip().strip('"').lower()


def parse_float(value: str) -> float | None:
    cleaned = value.strip().strip('"').replace(",", "")
    if not cleaned or cleaned.lower() in {"nan", "n/a"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def classify_kernel(kernel_name: str) -> str:
    name = kernel_name.lower()
    if "_pack_int4_kv_kernel" in name:
        return "kv_int4_pack"
    if "_int4_decode_attention_kernel" in name:
        return "kv_int4_decode_attention"
    if "_int4_prefill_attention_kernel" in name:
        return "kv_int4_prefill_attention_unexpected"
    if "_state_update_mx8_kernel" in name:
        return "mamba_state_mx8_update"
    if "_requantize_mx8_kernel" in name:
        return "mamba_state_mx8_requantize"
    if "_state_update_mx4_kernel" in name or "_requantize_mx4_kernel" in name:
        return "mamba_state_mx4_unexpected"
    if "scaled_dot_product" in name or "flash" in name or "fmha" in name or "attention" in name:
        return "attention_baseline_or_sdpa"
    if "gemm" in name or "matmul" in name or "cutlass" in name or "cublas" in name:
        return "gemm_linear"
    if "layernorm" in name or "rms" in name or "norm" in name:
        return "norm"
    if "copy" in name or "memcpy" in name:
        return "memory_copy"
    if "elementwise" in name or "vectorized" in name or "pointwise" in name:
        return "elementwise"
    return "other"


def candidate_columns(fieldnames: list[str], patterns: list[str]) -> str | None:
    lowered = {norm(name): name for name in fieldnames}
    for pattern in patterns:
        regex = re.compile(pattern)
        for lowered_name, original in lowered.items():
            if regex.search(lowered_name):
                return original
    return None


def read_ncu_csv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "kernel" in lowered and "metric" in lowered:
            header_idx = idx
            break
    if header_idx is None:
        return []

    reader = csv.DictReader(lines[header_idx:])
    if reader.fieldnames is None:
        return []
    kernel_col = candidate_columns(reader.fieldnames, [r"kernel.*name", r"^name$"])
    metric_col = candidate_columns(reader.fieldnames, [r"metric.*name", r"^metric$"])
    value_col = candidate_columns(reader.fieldnames, [r"metric.*value", r"^value$"])
    unit_col = candidate_columns(reader.fieldnames, [r"metric.*unit", r"^unit$"])
    if not kernel_col or not metric_col or not value_col:
        return []

    rows = []
    for row in reader:
        metric_name = norm(row.get(metric_col, ""))
        if metric_name != "gpu__time_duration.sum":
            continue
        value = parse_float(row.get(value_col, ""))
        if value is None:
            continue
        unit = norm(row.get(unit_col, ""))
        if unit in {"nsecond", "ns"}:
            duration_ms = value / 1.0e6
        elif unit in {"usecond", "us"}:
            duration_ms = value / 1.0e3
        elif unit in {"msecond", "ms"}:
            duration_ms = value
        elif unit in {"second", "s"}:
            duration_ms = value * 1.0e3
        else:
            # Nsight Compute often reports gpu__time_duration.sum as ns when a unit column is absent.
            duration_ms = value / 1.0e6
        kernel = row.get(kernel_col, "").strip().strip('"')
        rows.append(
            {
                "kernel_name": kernel,
                "category": classify_kernel(kernel),
                "duration_ms": duration_ms,
            }
        )
    return rows


def summarize(mode_to_rows: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    category_rows = []
    kernel_rows = []
    baseline_by_category: dict[str, float] = {}
    for mode in MODE_ORDER:
        rows = mode_to_rows.get(mode, [])
        total_ms = sum(row["duration_ms"] for row in rows)
        by_category: dict[str, float] = {}
        by_kernel: dict[str, tuple[str, float, int]] = {}
        for row in rows:
            by_category[row["category"]] = by_category.get(row["category"], 0.0) + row["duration_ms"]
            category, duration, count = by_kernel.get(row["kernel_name"], (row["category"], 0.0, 0))
            by_kernel[row["kernel_name"]] = (category, duration + row["duration_ms"], count + 1)
        if mode == "none":
            baseline_by_category = by_category.copy()
        for category, duration in sorted(by_category.items(), key=lambda item: item[1], reverse=True):
            baseline_ms = baseline_by_category.get(category)
            delta_pct = None if baseline_ms in (None, 0) else (duration - baseline_ms) / baseline_ms * 100.0
            category_rows.append(
                {
                    "mode": mode,
                    "category": category,
                    "duration_ms": duration,
                    "total_ms": total_ms,
                    "pct_of_mode": duration / total_ms * 100.0 if total_ms else None,
                    "baseline_category_ms": baseline_ms,
                    "delta_vs_baseline_pct": delta_pct,
                }
            )
        for kernel, (category, duration, count) in sorted(by_kernel.items(), key=lambda item: item[1][1], reverse=True):
            kernel_rows.append(
                {
                    "mode": mode,
                    "category": category,
                    "kernel_name": kernel,
                    "calls": count,
                    "duration_ms": duration,
                    "pct_of_mode": duration / total_ms * 100.0 if total_ms else None,
                }
            )
    return category_rows, kernel_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/ncu/data")
    parser.add_argument("--output-prefix", default="results/ncu/data/nemotron_8b_decode_ncu")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    mode_to_rows = {}
    for mode in MODE_ORDER:
        path = input_dir / f"nemotron_8b_decode_{mode}_ncu.csv"
        mode_to_rows[mode] = read_ncu_csv(path)
    category_rows, kernel_rows = summarize(mode_to_rows)
    output_prefix = Path(args.output_prefix)
    write_csv(output_prefix.with_name(output_prefix.name + "_category_summary.csv"), category_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_kernel_summary.csv"), kernel_rows)
    print(output_prefix.with_name(output_prefix.name + "_category_summary.csv"))
    print(output_prefix.with_name(output_prefix.name + "_kernel_summary.csv"))


if __name__ == "__main__":
    main()
