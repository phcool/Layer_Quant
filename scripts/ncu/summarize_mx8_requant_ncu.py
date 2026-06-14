from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


INTERESTING_METRICS = {
    "duration_ns": [r"gpu.*time", r"duration"],
    "long_scoreboard_pct": [r"long.*scoreboard.*pct"],
    "lg_throttle_pct": [r"lg.*throttle.*pct"],
    "tensor_active_pct": [r"pipe_tensor.*active.*pct", r"tensor.*active.*pct"],
    "sm_active_pct": [r"sm.*active.*pct"],
    "integer_inst": [r"inst.*integer", r"thread_inst_executed.*integer"],
    "local_load_bytes": [r"local.*load.*bytes"],
    "local_store_bytes": [r"local.*store.*bytes"],
    "local_memory_overhead": [r"local.*memory.*overhead"],
}


def parse_float(value: str) -> float | None:
    cleaned = value.strip().strip('"').replace(",", "")
    if not cleaned or cleaned.lower() in {"n/a", "nan", "--"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def norm_metric_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().strip('"').lower())


def metric_bucket(metric_name: str) -> str | None:
    lowered = norm_metric_name(metric_name)
    for bucket, patterns in INTERESTING_METRICS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                return bucket
    return None


def read_ncu_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    current: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("Metric Name") or row.get("Metric Name:") or row.get("Name") or ""
            value = row.get("Metric Value") or row.get("Metric Value:") or row.get("Value") or ""
            bucket = metric_bucket(metric)
            parsed = parse_float(value)
            if bucket is None or parsed is None:
                continue
            if bucket in current:
                rows.append(current)
                current = {}
            current[bucket] = parsed
    if current:
        rows.append(current)
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_mode(mode: str, path: Path) -> dict:
    launches = read_ncu_csv(path)
    out = {
        "mode": mode,
        "captured_launches": len(launches),
    }
    for key in INTERESTING_METRICS:
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

    rows = []
    input_dir = Path(args.input_dir)
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        rows.append(summarize_mode(mode, input_dir / args.template.format(mode=mode)))
    write_csv(Path(args.output), rows)
    print(args.output)


if __name__ == "__main__":
    main()
