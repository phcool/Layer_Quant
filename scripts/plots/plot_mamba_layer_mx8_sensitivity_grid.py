from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path("results/ppl/data")
FIGURE_DIR = Path("results/ppl/figures")
ATTENTION_LAYERS = [7, 18, 29, 40]
CASES = [
    (1024, 256),
    (1024, 512),
    (2048, 256),
    (2048, 512),
]


def case_path(context_length: int, decode_steps: int) -> Path:
    return DATA_DIR / (
        f"nemotron_8b_mamba_layer_mx8_sensitivity_ctx{context_length}_decode{decode_steps}.csv"
    )


def part_paths(context_length: int, decode_steps: int) -> list[Path]:
    stem = DATA_DIR / (
        f"nemotron_8b_mamba_layer_mx8_sensitivity_ctx{context_length}_decode{decode_steps}"
    )
    return [stem.with_name(stem.name + "_part1.csv"), stem.with_name(stem.name + "_part2.csv")]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rows_for_case(context_length: int, decode_steps: int) -> list[dict]:
    full_path = case_path(context_length, decode_steps)
    if full_path.exists():
        rows = read_rows(full_path)
    else:
        rows = []

    parts = [path for path in part_paths(context_length, decode_steps) if path.exists()]
    if parts:
        by_layer = {int(row["layer_idx"]): row for row in rows}
        for path in parts:
            for row in read_rows(path):
                by_layer[int(row["layer_idx"])] = row
        rows = [by_layer[layer_idx] for layer_idx in sorted(by_layer)]
        write_rows(full_path, rows)

    return rows


def plot_case(ax: plt.Axes, rows: list[dict], context_length: int, decode_steps: int) -> None:
    xs = [int(row["layer_idx"]) for row in rows]
    ys = [float(row["delta_ppl"]) for row in rows]
    colors = ["#1f77b4" if value >= 0 else "#2ca02c" for value in ys]

    ax.bar(xs, ys, width=0.72, color=colors, edgecolor="0.2", linewidth=0.35)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    for index, layer_idx in enumerate(ATTENTION_LAYERS):
        ax.axvline(
            layer_idx,
            color="0.35",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label="Attention layer" if index == 0 else None,
        )

    baseline_ppl = float(rows[0]["baseline_ppl"]) if rows else float("nan")
    ax.set_title(f"context={context_length}, decode={decode_steps} (baseline PPL={baseline_ppl:.4f})")
    ax.set_xlabel("Mamba Layer Index")
    ax.set_ylabel("Delta PPL")
    ax.set_xticks(xs)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.22)


def main() -> None:
    case_rows = {}
    for context_length, decode_steps in CASES:
        rows = rows_for_case(context_length, decode_steps)
        if not rows:
            raise FileNotFoundError(f"No rows for context={context_length}, decode={decode_steps}")
        case_rows[(context_length, decode_steps)] = rows

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), dpi=170, sharex=False)
    fig.suptitle(
        "Per-Mamba-Layer MX8 State Sensitivity\n"
        "(Nemotron-H-8B, WikiText; dashed lines mark attention layers)",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (context_length, decode_steps) in zip(axes.flat, CASES):
        plot_case(ax, case_rows[(context_length, decode_steps)], context_length, decode_steps)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    out_path = FIGURE_DIR / "nemotron_8b_mamba_layer_mx8_sensitivity_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
