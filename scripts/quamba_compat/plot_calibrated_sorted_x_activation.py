from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.nemotron_8b_decode_eval import layer_groups, patch_attention_cache_in_blocks
from scripts.quamba_compat.run_nemotron_quamba_full_w8a8 import (
    calibration_ids,
    configure_nemotron_mamba_layers,
    ensure_quamba_on_path,
    load_calibration_dataset,
)


class XActivationProfiler:
    def __init__(self) -> None:
        self.channel_max: torch.Tensor | None = None
        self.channel_sum: torch.Tensor | None = None
        self.count = 0
        self.test_x: torch.Tensor | None = None
        self.capture_test = False

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        if self.capture_test:
            self.test_x = x.cpu()[0]
            return
        x_abs = x.abs().reshape(-1, x.shape[-1])
        cur_max = x_abs.amax(dim=0).cpu()
        cur_sum = x_abs.sum(dim=0).cpu()
        self.channel_max = cur_max if self.channel_max is None else torch.maximum(self.channel_max, cur_max)
        self.channel_sum = cur_sum if self.channel_sum is None else self.channel_sum + cur_sum
        self.count += x_abs.shape[0]


def plot_sorted_x(output_root: Path, test_x: torch.Tensor, order: torch.Tensor, *, token_points: int, dim_points: int, z_clip_quantile: float) -> dict:
    x_sorted = test_x.abs()[:, order]
    seqlen, ndim = x_sorted.shape
    token_idx = torch.arange(0, seqlen, max(1, seqlen // token_points))[:token_points]
    dim_idx = torch.arange(0, ndim, max(1, ndim // dim_points))[:dim_points]
    z = x_sorted.index_select(0, token_idx).index_select(1, dim_idx)
    z_clip = float(torch.quantile(z.reshape(-1), z_clip_quantile))
    z_plot = z.clamp_max(z_clip)
    token_grid, dim_grid = torch.meshgrid(token_idx.float(), dim_idx.float(), indexing="ij")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(8.8, 6.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.45, 2.2, 0.62))
    norm = Normalize(vmin=0.0, vmax=max(z_clip, 1e-12))
    colors = cm.coolwarm(norm(z_plot.numpy()))
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
    ax.set_title("Calibrated-sorted x activation")
    ax.set_xlabel("Token")
    ax.set_ylabel("Sorted dims")
    ax.set_zlabel("|x activation|")
    ax.set_xlim(float(token_idx.min()), float(token_idx.max()))
    ax.set_ylim(float(dim_idx.max()), float(dim_idx.min()))
    ax.set_zlim(0.0, max(z_clip, 1e-12))
    ax.view_init(elev=26, azim=-58)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.06, label="|x activation|")
    fig.savefig(output_root / "figure3_calibrated_sorted_x_activation_3d.png", dpi=260)
    plt.close(fig)

    torch.save(
        {
            "token_idx": token_idx,
            "dim_idx": dim_idx,
            "z_abs_activation": z,
            "z_clipped": z_plot,
            "channel_order": order,
        },
        output_root / "figure3_calibrated_sorted_x_activation_3d_data.pt",
    )
    return {
        "source_shape": [seqlen, ndim],
        "render_shape": list(z.shape),
        "z_clip_quantile": z_clip_quantile,
        "z_clip": z_clip,
        "z_raw_max": float(z.max()),
        "z_raw_p99": float(torch.quantile(z.reshape(-1), 0.99)),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot original-Nemotron x activation sorted by offline calibration channel maxima.")
    parser.add_argument("--model", default="/scratch2/wl730/models/nemotron-h-8b")
    parser.add_argument("--output-root", default="results/quamba_compat/round22_calibrated_sorted_x")
    parser.add_argument("--calib-source", default="quamba_pile", choices=["local_prompts", "quamba_pile", "wikitext"])
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--test-sample-idx", type=int, default=512)
    parser.add_argument("--layer-idx", type=int, default=-1, help="-1 means the last Nemotron Mamba layer.")
    parser.add_argument("--token-points", type=int, default=256)
    parser.add_argument("--dim-points", type=int, default=768)
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

    profiler = XActivationProfiler()
    mixer = model.backbone.layers[layer_idx].mixer
    handle = mixer.x_conv_out.register_forward_hook(lambda _m, inputs, _out: profiler.update(inputs[0]))

    calibration_dataset = load_calibration_dataset(args.calib_source)
    device = next(model.parameters()).device
    for sample_idx in tqdm(range(args.num_samples), desc="x calibration"):
        input_ids = calibration_ids(tokenizer, sample_idx, args.seq_len, device, calibration_dataset)
        model(input_ids=input_ids, use_cache=False, return_dict=True)

    if profiler.channel_max is None:
        raise RuntimeError("Missing calibration x channel maxima.")
    order = torch.argsort(profiler.channel_max, descending=True)

    profiler.capture_test = True
    test_ids = calibration_ids(tokenizer, args.test_sample_idx, args.seq_len, device, calibration_dataset)
    model(input_ids=test_ids, use_cache=False, return_dict=True)
    profiler.capture_test = False
    handle.remove()

    if profiler.test_x is None:
        raise RuntimeError("Missing test x activation.")

    plot_meta = plot_sorted_x(
        output_root,
        profiler.test_x,
        order,
        token_points=args.token_points,
        dim_points=args.dim_points,
        z_clip_quantile=args.z_clip_quantile,
    )

    raw_path = output_root / "calibrated_sorted_x_profile.pt"
    torch.save(
        {
            "layer_idx": layer_idx,
            "x_channel_max": profiler.channel_max,
            "x_channel_mean": profiler.channel_sum / max(1, profiler.count),
            "channel_order": order,
            "test_x": profiler.test_x,
            "mamba_layers": mamba_layers,
            "attention_layers": attention_layers,
            "mlp_layers": mlp_layers,
        },
        raw_path,
    )
    meta = {
        "model": args.model,
        "calib_source": args.calib_source,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "test_sample_idx": args.test_sample_idx,
        "layer_idx": layer_idx,
        "raw_path": str(raw_path),
        "ordering": "calibration x_channel_max descending",
        "model_reordered": False,
        "post_forward_channel_sort": "fixed offline calibration order pi",
        **plot_meta,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "output_root": str(output_root), **meta}), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
