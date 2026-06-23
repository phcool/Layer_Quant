from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.nemotron_8b_decode_eval import layer_groups, patch_attention_cache_in_blocks
from scripts.quamba_compat.run_nemotron_quamba_full_w8a8 import (
    calibration_ids,
    configure_nemotron_mamba_layers,
    ensure_quamba_on_path,
    load_calibration_dataset,
)


class BCActivationCapture:
    def __init__(self) -> None:
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None

    @torch.no_grad()
    def update_B(self, x: torch.Tensor) -> None:
        self.B = x.detach().float().cpu()[0]

    @torch.no_grad()
    def update_C(self, x: torch.Tensor) -> None:
        self.C = x.detach().float().cpu()[0]


def sample_activation(x: torch.Tensor, token_points: int, dim_points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqlen, ndim = x.shape
    token_idx = torch.arange(0, seqlen, max(1, seqlen // token_points))[:token_points]
    dim_idx = torch.arange(0, ndim, max(1, ndim // dim_points))[:dim_points]
    z = x.abs().index_select(0, token_idx).index_select(1, dim_idx)
    return token_idx, dim_idx, z


def draw_surface(
    fig,
    ax,
    cax,
    token_grid: torch.Tensor,
    dim_grid: torch.Tensor,
    z: torch.Tensor,
    *,
    z_clip: float,
    title: str,
    colorbar_label: str,
) -> None:
    from matplotlib import cm
    from matplotlib.colors import Normalize

    z_plot = z.clamp_max(z_clip)
    norm = Normalize(vmin=0.0, vmax=max(z_clip, 1e-12))
    colors = cm.coolwarm(norm(z_plot.numpy()))
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.45, 1.55, 0.62))
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
    ax.set_ylabel("Dims", labelpad=10)
    ax.set_zlabel("")
    ax.set_xlim(float(token_grid.min()), float(token_grid.max()))
    ax.set_ylim(float(dim_grid.min()), float(dim_grid.max()))
    ax.set_zlim(0.0, max(z_clip, 1e-12))
    ax.tick_params(axis="both", which="major", labelsize=8, pad=2)
    ax.tick_params(axis="z", which="major", labelsize=8, pad=2)
    ax.view_init(elev=25, azim=-58)
    scalar = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=cax)
    colorbar.set_label(colorbar_label, rotation=90, labelpad=10)
    colorbar.ax.tick_params(labelsize=8)


def plot_bc(
    output_root: Path,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    token_points: int,
    dim_points: int,
    z_clip_quantile: float,
) -> dict:
    b_token_idx, b_dim_idx, b_z = sample_activation(B, token_points, dim_points)
    c_token_idx, c_dim_idx, c_z = sample_activation(C, token_points, dim_points)
    if not torch.equal(b_token_idx, c_token_idx) or not torch.equal(b_dim_idx, c_dim_idx):
        raise RuntimeError("B/C render grids do not match.")
    b_z_clip = float(torch.quantile(b_z.reshape(-1), z_clip_quantile))
    c_z_clip = float(torch.quantile(c_z.reshape(-1), z_clip_quantile))
    token_grid, dim_grid = torch.meshgrid(b_token_idx.float(), b_dim_idx.float(), indexing="ij")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17.5, 7.2))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.035, 1.0, 0.035], wspace=0.22)
    ax_b = fig.add_subplot(grid[0, 0], projection="3d")
    cax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2], projection="3d")
    cax_c = fig.add_subplot(grid[0, 3])
    draw_surface(fig, ax_b, cax_b, token_grid, dim_grid, b_z, z_clip=b_z_clip, title="(b) B activation", colorbar_label="|B activation|")
    draw_surface(fig, ax_c, cax_c, token_grid, dim_grid, c_z, z_clip=c_z_clip, title="(c) C activation", colorbar_label="|C activation|")
    fig.savefig(output_root / "figure3bc_B_C_activation_3d.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    torch.save(
        {
            "token_idx": b_token_idx,
            "dim_idx": b_dim_idx,
            "B_abs_activation": b_z,
            "C_abs_activation": c_z,
            "B_clipped": b_z.clamp_max(b_z_clip),
            "C_clipped": c_z.clamp_max(c_z_clip),
            "B_raw": B,
            "C_raw": C,
        },
        output_root / "figure3bc_B_C_activation_3d_data.pt",
    )
    return {
        "source_shape": list(B.shape),
        "render_shape": list(b_z.shape),
        "z_clip_quantile": z_clip_quantile,
        "B_z_clip": b_z_clip,
        "C_z_clip": c_z_clip,
        "B_raw_max": float(b_z.max()),
        "C_raw_max": float(c_z.max()),
        "B_raw_p99": float(torch.quantile(b_z.reshape(-1), 0.99)),
        "C_raw_p99": float(torch.quantile(c_z.reshape(-1), 0.99)),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Nemotron Mamba B_t/C_t activations as Quamba Figure-3-style 3D surfaces.")
    parser.add_argument("--model", default="/scratch2/wl730/models/nemotron-h-8b")
    parser.add_argument("--output-root", default="results/quamba_compat/round23_figure3bc_B_C_activation")
    parser.add_argument("--calib-source", default="quamba_pile", choices=["local_prompts", "quamba_pile", "wikitext"])
    parser.add_argument("--sample-idx", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--layer-idx", type=int, default=-1, help="-1 means the last Nemotron Mamba layer.")
    parser.add_argument("--token-points", type=int, default=256)
    parser.add_argument("--dim-points", type=int, default=512)
    parser.add_argument("--z-clip-quantile", type=float, default=0.999)
    args = parser.parse_args()

    ensure_quamba_on_path()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    mamba_layers, attention_layers, mlp_layers = layer_groups(config)
    layer_idx = mamba_layers[-1] if args.layer_idx < 0 else args.layer_idx
    if layer_idx not in mamba_layers:
        raise ValueError(f"Layer {layer_idx} is not a Nemotron Mamba layer. Mamba layers: {mamba_layers}")

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
    patch_attention_cache_in_blocks(model)
    configure_nemotron_mamba_layers(model, mamba_layers, use_had_transform=False)

    capture = BCActivationCapture()
    mixer = model.backbone.layers[layer_idx].mixer
    handles = [
        mixer.B_conv_out.register_forward_hook(lambda _m, inputs, _out: capture.update_B(inputs[0])),
        mixer.C_conv_out.register_forward_hook(lambda _m, inputs, _out: capture.update_C(inputs[0])),
    ]

    dataset = load_calibration_dataset(args.calib_source)
    device = next(model.parameters()).device
    input_ids = calibration_ids(tokenizer, args.sample_idx, args.seq_len, device, dataset)
    model(input_ids=input_ids, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()

    if capture.B is None or capture.C is None:
        raise RuntimeError(f"Missing B/C activations: B={capture.B is not None}, C={capture.C is not None}")
    if capture.B.shape != capture.C.shape:
        raise RuntimeError(f"B/C shape mismatch: B={tuple(capture.B.shape)} C={tuple(capture.C.shape)}")

    plot_meta = plot_bc(
        output_root,
        capture.B,
        capture.C,
        token_points=args.token_points,
        dim_points=args.dim_points,
        z_clip_quantile=args.z_clip_quantile,
    )
    meta = {
        "model": args.model,
        "calib_source": args.calib_source,
        "sample_idx": args.sample_idx,
        "seq_len": args.seq_len,
        "layer_idx": layer_idx,
        "ngroups": int(mixer.ngroups),
        "d_state": int(mixer.d_state),
        "flattened_dims": "ngroups * d_state",
        "dims_axis": "increasing flattened group/state dim index",
        "mamba_layers": mamba_layers,
        "attention_layers": attention_layers,
        "mlp_layers": mlp_layers,
        **plot_meta,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "output_root": str(output_root), **meta}), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
