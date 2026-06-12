from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("results/nemotron_8b_decode_mx8_like_reference.png")
KERNEL_CSV = Path("results/nemotron_8b_decode_degradation_ctx1024_kernel.csv")
MX8_CSV = Path("results/nemotron_8b_decode_mx8_kernel_ctx1024.csv")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_experiment(rows: list[dict], *names: str) -> list[dict]:
    selected = [row for row in rows if row["experiment"] in names]
    selected.sort(key=lambda row: int(row["decode_steps"]))
    return selected


def xs_ys(rows: list[dict]) -> tuple[list[int], list[float]]:
    return [int(row["decode_steps"]) for row in rows], [float(row["ppl_degradation_pct"]) for row in rows]


def main() -> None:
    kernel_rows = read_rows(KERNEL_CSV)
    mx8_rows = read_rows(MX8_CSV)

    series = {
        "kv_int4": {
            "label": "KV Cache INT4",
            "color": "#ff7f0e",
            "marker": "o",
            "rows": by_experiment(kernel_rows, "kv_int4"),
        },
        "ssm_mx8": {
            "label": "SSM State MX8",
            "color": "#2ca02c",
            "marker": "s",
            "rows": by_experiment(mx8_rows, "ssm_mx8", "ssm_mxfp8"),
        },
        "both_int4_mx8": {
            "label": "Both INT4+MX8",
            "color": "#1f77ff",
            "marker": "D",
            "rows": by_experiment(mx8_rows, "both_int4_mx8", "both_int4_mxfp8"),
        },
    }
    decode_steps = xs_ys(series["kv_int4"]["rows"])[0]
    tick_labels = ["128", "256", "512", "1K", "2K"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=170)
    fig.suptitle(
        "Cumulative Quantization Error in Hybrid Model Decode\n"
        "(Nemotron-H-8B, context=1024, WikiText)",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )

    for key in ["kv_int4", "ssm_mx8", "both_int4_mx8"]:
        item = series[key]
        x, y = xs_ys(item["rows"])
        ax0.plot(
            x,
            y,
            marker=item["marker"],
            color=item["color"],
            linewidth=1.7,
            markersize=4.2,
            label=item["label"],
        )

    for key in ["kv_int4", "ssm_mx8", "both_int4_mx8"]:
        item = series[key]
        x, y = xs_ys(item["rows"])
        ax1.plot(
            x,
            y,
            marker=item["marker"],
            color=item["color"],
            linewidth=1.7,
            markersize=4.2,
            label=item["label"],
        )

    for ax in (ax0, ax1):
        ax.set_xscale("log", base=2)
        ax.set_xticks(decode_steps)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("Decode Steps")
        ax.set_ylabel("PPL Degradation (%)")
        ax.axhline(5.0, color="0.65", linestyle=":", linewidth=0.9)
        ax.tick_params(labelsize=8)

    ax0.set_title("(a) Cumulative Quantization Error", fontsize=9)
    ax0.set_ylim(-20, 1300)
    ax0.text(135, 45, "5% threshold", color="0.55", fontsize=7)
    ax0.legend(loc="upper left", fontsize=7, framealpha=0.85)

    ax1.set_title("(b) Detail: Near-Lossless / MX8 Configs", fontsize=9)
    ax1.set_ylim(-1, 55)
    ax1.annotate(
        "KV error diluted",
        xy=(2048, 1.45),
        xytext=(1030, 12),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#ff7f0e"},
        color="#ff7f0e",
        fontsize=7,
    )
    ax1.annotate(
        "MX8 bounded",
        xy=(2048, 3.96),
        xytext=(560, 13),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#2ca02c"},
        color="#2ca02c",
        fontsize=7,
    )
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.85)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
