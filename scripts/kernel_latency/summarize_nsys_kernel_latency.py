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
    if not cleaned or cleaned.lower() in {"nan", "n/a", "--"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def classify_kernel(kernel_name: str) -> str:
    name = kernel_name.lower()
    if "_int4_decode_attention_kernel" in name:
        return "attention_kv_int4"
    if "fmha" in name or "flash" in name or "scaled_dot_product" in name or "attentionkernel" in name:
        return "attention_baseline"
    if "_selective_scan_update_kernel" in name:
        return "mamba_state_update_baseline"
    if "_state_update_mx8_kernel" in name:
        return "mamba_state_update_mx8"
    if "_requantize_mx8_kernel" in name:
        return "mamba_state_requantize_mx8"
    if "causal_conv1d_update_kernel" in name:
        return "mamba_conv_update"
    if "_pack_int4_kv_kernel" in name:
        return "kv_int4_pack"
    return "other"


def find_col(fieldnames: list[str], patterns: list[str]) -> str | None:
    lowered = {norm(name): name for name in fieldnames}
    for pattern in patterns:
        regex = re.compile(pattern)
        for lowered_name, original in lowered.items():
            if regex.search(lowered_name):
                return original
    return None


def read_nsys_kernel_summary(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "name" in lowered and ("total time" in lowered or "total" in lowered) and "instances" in lowered:
            header_idx = idx
            break
    if header_idx is None:
        return []

    reader = csv.DictReader(lines[header_idx:])
    if reader.fieldnames is None:
        return []
    name_col = find_col(reader.fieldnames, [r"^name$", r"kernel"])
    instances_col = find_col(reader.fieldnames, [r"instances", r"calls"])
    total_col = find_col(reader.fieldnames, [r"total.*time", r"^total$"])
    avg_col = find_col(reader.fieldnames, [r"avg", r"average"])
    time_unit = "ns"
    header = ",".join(reader.fieldnames).lower()
    if "(ns)" in header or "nsec" in header:
        time_unit = "ns"
    elif "(us)" in header or "usec" in header:
        time_unit = "us"
    elif "(ms)" in header or "msec" in header:
        time_unit = "ms"

    if name_col is None or total_col is None:
        return []

    rows = []
    for row in reader:
        kernel = (row.get(name_col) or "").strip().strip('"')
        if not kernel:
            continue
        total = parse_float(row.get(total_col, ""))
        if total is None:
            continue
        instances = parse_float(row.get(instances_col, "")) if instances_col else None
        avg = parse_float(row.get(avg_col, "")) if avg_col else None
        if time_unit == "ns":
            total_ms = total / 1.0e6
            avg_ms = None if avg is None else avg / 1.0e6
        elif time_unit == "us":
            total_ms = total / 1.0e3
            avg_ms = None if avg is None else avg / 1.0e3
        else:
            total_ms = total
            avg_ms = avg
        rows.append(
            {
                "kernel_name": kernel,
                "category": classify_kernel(kernel),
                "calls": int(instances) if instances is not None else None,
                "total_ms": total_ms,
                "avg_ms_per_call": avg_ms,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(input_dir: Path, decode_steps: int) -> tuple[list[dict], list[dict]]:
    kernel_rows = []
    category_rows = []
    for mode in MODE_ORDER:
        rows = read_nsys_kernel_summary(input_dir / f"nemotron_8b_{mode}_nsys_cuda_gpu_kern_sum.csv")
        total_mode_ms = sum(row["total_ms"] for row in rows)
        by_category: dict[str, dict] = {}
        for row in rows:
            out = {
                "mode": mode,
                "category": row["category"],
                "kernel_name": row["kernel_name"],
                "calls": row["calls"],
                "total_ms": row["total_ms"],
                "avg_ms_per_call": row["avg_ms_per_call"],
                "avg_ms_per_decode_step": row["total_ms"] / decode_steps,
                "pct_of_mode_kernel_time": row["total_ms"] / total_mode_ms * 100.0 if total_mode_ms else None,
            }
            kernel_rows.append(out)
            cat = by_category.setdefault(
                row["category"],
                {"mode": mode, "category": row["category"], "calls": 0, "total_ms": 0.0},
            )
            cat["calls"] += row["calls"] or 0
            cat["total_ms"] += row["total_ms"]
        for cat in by_category.values():
            cat["avg_ms_per_decode_step"] = cat["total_ms"] / decode_steps
            cat["pct_of_mode_kernel_time"] = cat["total_ms"] / total_mode_ms * 100.0 if total_mode_ms else None
            category_rows.append(cat)

        attention_ms = sum(
            row["total_ms"] for row in rows if row["category"] in {"attention_baseline", "attention_kv_int4"}
        )
        update_ms = sum(
            row["total_ms"]
            for row in rows
            if row["category"] in {"mamba_state_update_baseline", "mamba_state_update_mx8"}
        )
        requant_ms = sum(row["total_ms"] for row in rows if row["category"] == "mamba_state_requantize_mx8")
        conv_ms = sum(row["total_ms"] for row in rows if row["category"] == "mamba_conv_update")
        category_rows.append(
            {
                "mode": mode,
                "category": "rollup_attention",
                "calls": sum(
                    row["calls"] or 0
                    for row in rows
                    if row["category"] in {"attention_baseline", "attention_kv_int4"}
                ),
                "total_ms": attention_ms,
                "avg_ms_per_decode_step": attention_ms / decode_steps,
                "pct_of_mode_kernel_time": attention_ms / total_mode_ms * 100.0 if total_mode_ms else None,
            }
        )
        category_rows.append(
            {
                "mode": mode,
                "category": "rollup_mamba_state_update",
                "calls": sum(
                    row["calls"] or 0
                    for row in rows
                    if row["category"] in {"mamba_state_update_baseline", "mamba_state_update_mx8"}
                ),
                "total_ms": update_ms,
                "avg_ms_per_decode_step": update_ms / decode_steps,
                "pct_of_mode_kernel_time": update_ms / total_mode_ms * 100.0 if total_mode_ms else None,
            }
        )
        category_rows.append(
            {
                "mode": mode,
                "category": "rollup_mamba_state_update_plus_requant",
                "calls": sum(
                    row["calls"] or 0
                    for row in rows
                    if row["category"]
                    in {"mamba_state_update_baseline", "mamba_state_update_mx8", "mamba_state_requantize_mx8"}
                ),
                "total_ms": update_ms + requant_ms,
                "avg_ms_per_decode_step": (update_ms + requant_ms) / decode_steps,
                "pct_of_mode_kernel_time": (update_ms + requant_ms) / total_mode_ms * 100.0 if total_mode_ms else None,
            }
        )
        category_rows.append(
            {
                "mode": mode,
                "category": "rollup_mamba_conv_update",
                "calls": sum(row["calls"] or 0 for row in rows if row["category"] == "mamba_conv_update"),
                "total_ms": conv_ms,
                "avg_ms_per_decode_step": conv_ms / decode_steps,
                "pct_of_mode_kernel_time": conv_ms / total_mode_ms * 100.0 if total_mode_ms else None,
            }
        )
    return category_rows, kernel_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/kernel_latency/data")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--output-prefix", default="results/kernel_latency/data/nemotron_8b_kernel_latency")
    args = parser.parse_args()

    category_rows, kernel_rows = summarize(Path(args.input_dir), args.decode_steps)
    output_prefix = Path(args.output_prefix)
    write_csv(output_prefix.with_name(output_prefix.name + "_category_summary.csv"), category_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_kernel_summary.csv"), kernel_rows)
    print(output_prefix.with_name(output_prefix.name + "_category_summary.csv"))
    print(output_prefix.with_name(output_prefix.name + "_kernel_summary.csv"))


if __name__ == "__main__":
    main()
