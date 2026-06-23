from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def group_token_max(x: torch.Tensor, ngroups: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected activation [token, ngroups*d_state], got {tuple(x.shape)}")
    if x.shape[-1] % ngroups != 0:
        raise ValueError(f"Activation dim {x.shape[-1]} is not divisible by ngroups={ngroups}")
    d_state = x.shape[-1] // ngroups
    return x.abs().reshape(x.shape[0], ngroups, d_state).amax(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-group B/C maximum boxplots from saved Figure-3 B/C data.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ngroups", type=int, default=8)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.data, map_location="cpu")
    B = payload["B_raw"].float()
    C = payload["C_raw"].float()
    B_group_max = group_token_max(B, args.ngroups)
    C_group_max = group_token_max(C, args.ngroups)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), constrained_layout=True)
    for ax, data, title in [
        (axes[0], B_group_max, r"(e) $B$ group maximum"),
        (axes[1], C_group_max, r"(f) $C$ group maximum"),
    ]:
        ax.boxplot(
            [data[:, i].numpy() for i in range(args.ngroups)],
            showfliers=True,
            patch_artist=True,
            medianprops={"color": "#f28e2b", "linewidth": 1.5},
            boxprops={"facecolor": "#1f77b4", "edgecolor": "#1f77b4", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#333333",
                "markersize": 3.5,
                "markeredgewidth": 0.7,
            },
        )
        ax.set_title(title, y=-0.27, fontsize=18, fontfamily="serif")
        ax.set_xlabel("group index")
        ax.set_ylabel("Abs. Values")
        ax.set_xticks(range(1, args.ngroups + 1), range(args.ngroups))
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.savefig(output_root / "figure3ef_B_C_group_maximum.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    torch.save(
        {
            "B_group_max": B_group_max,
            "C_group_max": C_group_max,
            "ngroups": args.ngroups,
            "d_state": B.shape[-1] // args.ngroups,
        },
        output_root / "figure3ef_B_C_group_maximum_data.pt",
    )
    meta = {
        "source_data": args.data,
        "output_root": str(output_root),
        "ngroups": args.ngroups,
        "d_state": B.shape[-1] // args.ngroups,
        "B_group_max_shape": list(B_group_max.shape),
        "C_group_max_shape": list(C_group_max.shape),
        "B_group_max_global_max": float(B_group_max.max()),
        "C_group_max_global_max": float(C_group_max.max()),
        "definition": "for each token and group, max(abs(activation[token, group, state])) over state dim",
    }
    (output_root / "figure3ef_B_C_group_maximum_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **meta}), flush=True)


if __name__ == "__main__":
    main()
