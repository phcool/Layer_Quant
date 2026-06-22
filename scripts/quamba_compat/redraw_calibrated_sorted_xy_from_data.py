from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw calibrated-sorted x/y activation 3D plot from saved data.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--z-clip-quantile", type=float, default=0.999)
    args = parser.parse_args()

    payload = torch.load(args.data, map_location="cpu")
    token_idx = payload["token_idx"].float()
    dim_idx = payload["dim_idx"].float()
    x_z = payload["x_abs_activation"].float()
    y_z = payload["y_abs_activation"].float()
    x_clip = float(torch.quantile(x_z.reshape(-1), args.z_clip_quantile))
    y_clip = float(torch.quantile(y_z.reshape(-1), args.z_clip_quantile))
    token_grid, dim_grid = torch.meshgrid(token_idx, dim_idx, indexing="ij")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(17.5, 7.2), constrained_layout=False)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.035, 1.0, 0.035], wspace=0.22)
    ax_x = fig.add_subplot(grid[0, 0], projection="3d")
    cax_x = fig.add_subplot(grid[0, 1])
    ax_y = fig.add_subplot(grid[0, 2], projection="3d")
    cax_y = fig.add_subplot(grid[0, 3])

    def draw(ax, cax, z: torch.Tensor, clip: float, title: str, label: str) -> None:
        z_plot = z.clamp_max(clip)
        norm = Normalize(vmin=0.0, vmax=max(clip, 1e-12))
        colors = cm.coolwarm(norm(z_plot.numpy()))
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1.45, 2.1, 0.62))
        ax.plot_surface(
            token_grid.numpy(),
            dim_grid.numpy(),
            z_plot.numpy(),
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        ax.set_title(title, pad=8)
        ax.set_xlabel("Token", labelpad=8)
        ax.set_ylabel("Sorted dims", labelpad=10)
        ax.set_zlabel("")
        ax.set_xlim(float(token_idx.min()), float(token_idx.max()))
        ax.set_ylim(float(dim_idx.min()), float(dim_idx.max()))
        ax.set_zlim(0.0, max(clip, 1e-12))
        ax.set_xticks([0, 128, 256, 384, 511])
        ax.set_yticks([0, 2000, 4000, 6000, 8000])
        ax.tick_params(axis="both", which="major", labelsize=8, pad=2)
        ax.tick_params(axis="z", which="major", labelsize=8, pad=2)
        ax.view_init(elev=25, azim=-58)
        scalar = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, cax=cax)
        colorbar.set_label(label, rotation=90, labelpad=10)
        colorbar.ax.tick_params(labelsize=8)

    draw(ax_x, cax_x, x_z, x_clip, "(a) Calibrated-sorted x activation", "|x activation|")
    draw(ax_y, cax_y, y_z, y_clip, "(b) Calibrated-sorted y activation", "|y activation|")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print({"output": args.output, "x_clip": x_clip, "y_clip": y_clip})


if __name__ == "__main__":
    main()
