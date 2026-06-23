from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LATENCY_CSV = Path("results/latency/data/nemotron_8b_seq2048_latency.csv")
OUT = Path("results/latency/figures/nemotron_8b_seq2048_latency.png")

SERIES = {
    ("none", "none"): {
        "label": "Baseline",
        "color": "#4d4d4d",
        "marker": "o",
    },
    ("int4", "none"): {
        "label": "KV Cache INT4",
        "color": "#ff7f0e",
        "marker": "s",
    },
    ("none", "mx8"): {
        "label": "SSM State MX8",
        "color": "#2ca02c",
        "marker": "^",
    },
    ("int4", "mx8"): {
        "label": "Both INT4+MX8",
        "color": "#1f77ff",
        "marker": "D",
    },
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["status"] == "ok"]


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = {key: [] for key in SERIES}
    for row in rows:
        key = (row["kv_quantization"], row["state_quantization"])
        if key in grouped:
            grouped[key].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: int(row["batch_size"]))
    return grouped


def baseline_by_batch(grouped: dict[tuple[str, str], list[dict]]) -> dict[int, float]:
    return {
        int(row["batch_size"]): float(row["decode_ms_per_token"])
        for row in grouped[("none", "none")]
    }


def main() -> None:
    rows = read_rows(LATENCY_CSV)
    grouped = group_rows(rows)
    baseline = baseline_by_batch(grouped)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=170)
    fig.suptitle(
        "Decode Latency Under KV/State Quantization\n"
        "(Nemotron-H-8B, sequence=2048, decode=32)",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )

    for key, rows_for_key in grouped.items():
        if not rows_for_key:
            continue
        style = SERIES[key]
        batches = [int(row["batch_size"]) for row in rows_for_key]
        ms_per_token = [float(row["decode_ms_per_token"]) for row in rows_for_key]
        relative = [value / baseline[batch] for batch, value in zip(batches, ms_per_token)]

        ax0.plot(
            batches,
            ms_per_token,
            marker=style["marker"],
            color=style["color"],
            linewidth=1.7,
            markersize=4.2,
            label=style["label"],
        )
        ax1.plot(
            batches,
            relative,
            marker=style["marker"],
            color=style["color"],
            linewidth=1.7,
            markersize=4.2,
            label=style["label"],
        )

    for ax in (ax0, ax1):
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32])
        ax.set_xticklabels(["8", "16", "32"])
        ax.set_xlabel("Batch Size")
        ax.grid(True, alpha=0.22)
        ax.tick_params(labelsize=8)

    ax0.set_title("(a) Decode Latency", fontsize=9)
    ax0.set_ylabel("Decode Latency (ms/token)")
    ax0.legend(loc="upper right", fontsize=7, framealpha=0.85)

    ax1.set_title("(b) Relative to Baseline", fontsize=9)
    ax1.set_ylabel("Latency / Baseline")
    ax1.axhline(1.0, color="0.55", linestyle=":", linewidth=0.9)
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.85)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
