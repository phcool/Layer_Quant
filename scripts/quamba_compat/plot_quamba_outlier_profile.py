from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tensor_summary(name: str, x: torch.Tensor) -> dict[str, Any]:
    flat = x.detach().float().abs().reshape(-1)
    return {
        "name": name,
        "numel": int(flat.numel()),
        "mean_abs": float(flat.mean()),
        "p99_abs": float(torch.quantile(flat, 0.99)),
        "p999_abs": float(torch.quantile(flat, 0.999)),
        "max_abs": float(flat.max()),
    }


class ActivationCapture:
    def __init__(self, layer_idx: int, ngroups: int) -> None:
        self.layer_idx = layer_idx
        self.ngroups = ngroups
        self.x_max: torch.Tensor | None = None
        self.x_sum: torch.Tensor | None = None
        self.x_count = 0
        self.B_group_max_rows: list[torch.Tensor] = []
        self.C_group_max_rows: list[torch.Tensor] = []
        self.online: dict[str, torch.Tensor] = {}
        self.capture_online = False

    @torch.no_grad()
    def update_x(self, x: torch.Tensor) -> None:
        x_abs = x.detach().float().abs()
        flat = x_abs.reshape(-1, x_abs.shape[-1])
        cur_max = flat.max(dim=0).values.cpu()
        cur_sum = flat.sum(dim=0).cpu()
        self.x_max = cur_max if self.x_max is None else torch.maximum(self.x_max, cur_max)
        self.x_sum = cur_sum if self.x_sum is None else self.x_sum + cur_sum
        self.x_count += flat.shape[0]
        if self.capture_online:
            self.online["x"] = x.detach().float().cpu()[0]

    @torch.no_grad()
    def update_y(self, y: torch.Tensor) -> None:
        if self.capture_online:
            self.online["y"] = y.detach().float().cpu()[0]

    @torch.no_grad()
    def update_bc(self, name: str, x: torch.Tensor) -> None:
        x_abs = x.detach().float().abs()
        bsz, seqlen, dim = x_abs.shape
        dstate = dim // self.ngroups
        grouped = x_abs.reshape(bsz, seqlen, self.ngroups, dstate)
        group_max = grouped.amax(dim=(0, 1, 3)).cpu()
        if name == "B":
            self.B_group_max_rows.append(group_max)
        else:
            self.C_group_max_rows.append(group_max)
        if self.capture_online:
            self.online[name] = x.detach().float().cpu()[0]


def make_hooks(model, layer_idx: int, capture: ActivationCapture):
    mixer = model.backbone.layers[layer_idx].mixer
    handles = []
    handles.append(mixer.x_conv_out.register_forward_hook(lambda _m, inputs, _out: capture.update_x(inputs[0])))
    handles.append(mixer.B_conv_out.register_forward_hook(lambda _m, inputs, _out: capture.update_bc("B", inputs[0])))
    handles.append(mixer.C_conv_out.register_forward_hook(lambda _m, inputs, _out: capture.update_bc("C", inputs[0])))
    handles.append(mixer.out_proj.register_forward_pre_hook(lambda _m, inputs: capture.update_y(inputs[0])))
    return handles


def plot_profile(output_root: Path, capture: ActivationCapture, *, headdim: int, top_layer: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_max = capture.x_max
    if x_max is None:
        raise RuntimeError("Missing x channel calibration max.")
    sorted_idx = torch.argsort(x_max, descending=True)
    x = capture.online["x"].abs()
    y = capture.online["y"].abs()
    B = capture.online["B"].abs()
    C = capture.online["C"].abs()
    x_sorted = x[:, sorted_idx]
    y_sorted = y[:, sorted_idx]

    B_group_rows = torch.stack(capture.B_group_max_rows)
    C_group_rows = torch.stack(capture.C_group_max_rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    heatmaps = [
        (axes[0, 0], x_sorted, "(a) sorted x activation", "dims sorted by calibrated max"),
        (axes[0, 1], y_sorted, "(b) y activation", "same sorted dims"),
        (axes[0, 2], B, "(c) B activation", "state groups"),
        (axes[1, 0], C, "(d) C activation", "state groups"),
    ]
    for ax, data, title, xlabel in heatmaps:
        im = ax.imshow(data.numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("token")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[1, 1].boxplot([B_group_rows[:, i].numpy() for i in range(capture.ngroups)], showfliers=True)
    axes[1, 1].set_title("(e) B group maximum")
    axes[1, 1].set_xlabel("group")
    axes[1, 1].set_ylabel("Abs. Values")

    axes[1, 2].boxplot([C_group_rows[:, i].numpy() for i in range(capture.ngroups)], showfliers=True)
    axes[1, 2].set_title("(f) C group maximum")
    axes[1, 2].set_xlabel("group")
    axes[1, 2].set_ylabel("Abs. Values")

    fig.suptitle(f"Nemotron-H-8B Mamba layer {top_layer}: Quamba2-style activation persistence profile")
    fig.savefig(output_root / "figure3_activation_persistence.png", dpi=220)
    plt.close(fig)

    nheads = x_max.numel() // headdim
    unsorted = x_max.reshape(nheads, headdim)
    sorted_each_head = torch.stack([row[torch.argsort(row, descending=True)] for row in unsorted])
    head_order = torch.argsort(unsorted.max(dim=1).values, descending=True)
    sorted_heads = sorted_each_head[head_order]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, data, title in [
        (axes[0], unsorted, "(a) calibrated channel max"),
        (axes[1], sorted_each_head, "(b) sort each head"),
        (axes[2], sorted_heads, "(c) sort heads by head max"),
    ]:
        im = ax.imshow(data.numpy(), aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("#Channels")
        ax.set_ylabel("#Heads")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Nemotron-H-8B Mamba layer {top_layer}: channel outlier sorting")
    fig.savefig(output_root / "figure4_sort_and_cluster_proxy.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Quamba2-style outlier profiles for Nemotron-H Mamba layers.")
    parser.add_argument("--model", default="/scratch2/wl730/models/nemotron-h-8b")
    parser.add_argument("--output-root", default="results/quamba_compat/round19_quamba_outlier_profile")
    parser.add_argument("--calib-source", default="quamba_pile", choices=["local_prompts", "quamba_pile", "wikitext"])
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--online-sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=-1, help="-1 means the last Nemotron Mamba layer.")
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda().eval()
    patch_attention_cache_in_blocks(model)
    configure_nemotron_mamba_layers(model, mamba_layers, use_had_transform=False)

    mixer = model.backbone.layers[layer_idx].mixer
    capture = ActivationCapture(layer_idx, ngroups=mixer.ngroups)
    handles = make_hooks(model, layer_idx, capture)

    calibration_dataset = load_calibration_dataset(args.calib_source)
    device = next(model.parameters()).device
    for sample_idx in tqdm(range(args.num_samples), desc="profile calibration"):
        capture.capture_online = sample_idx == args.online_sample_idx
        input_ids = calibration_ids(tokenizer, sample_idx, args.seq_len, device, calibration_dataset)
        model(input_ids=input_ids, use_cache=False, return_dict=True)
    capture.capture_online = False

    for handle in handles:
        handle.remove()

    if set(capture.online) != {"x", "y", "B", "C"}:
        raise RuntimeError(f"Missing online tensors: {sorted(capture.online)}")

    raw_path = output_root / "profile_tensors.pt"
    torch.save(
        {
            "layer_idx": layer_idx,
            "x_channel_max": capture.x_max,
            "x_channel_mean": capture.x_sum / max(1, capture.x_count),
            "B_group_max": torch.stack(capture.B_group_max_rows),
            "C_group_max": torch.stack(capture.C_group_max_rows),
            "online": capture.online,
            "mamba_layers": mamba_layers,
            "attention_layers": attention_layers,
            "mlp_layers": mlp_layers,
        },
        raw_path,
    )

    rows = [
        tensor_summary("online_x", capture.online["x"]),
        tensor_summary("online_y", capture.online["y"]),
        tensor_summary("online_B", capture.online["B"]),
        tensor_summary("online_C", capture.online["C"]),
    ]
    write_csv(output_root / "tensor_summary.csv", rows)
    write_csv(
        output_root / "group_max_summary.csv",
        [
            {
                "tensor": name,
                "group": group_idx,
                "mean_group_max": float(values[:, group_idx].mean()),
                "p90_group_max": float(torch.quantile(values[:, group_idx], 0.90)),
                "max_group_max": float(values[:, group_idx].max()),
            }
            for name, values in [("B", torch.stack(capture.B_group_max_rows)), ("C", torch.stack(capture.C_group_max_rows))]
            for group_idx in range(values.shape[1])
        ],
    )

    plot_error = None
    try:
        plot_profile(output_root, capture, headdim=mixer.headdim, top_layer=layer_idx)
    except Exception as exc:  # keep raw data even if plotting deps are unavailable
        plot_error = repr(exc)

    meta = {
        "model": args.model,
        "calib_source": args.calib_source,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "online_sample_idx": args.online_sample_idx,
        "layer_idx": layer_idx,
        "ngroups": mixer.ngroups,
        "headdim": mixer.headdim,
        "d_state": mixer.d_state,
        "raw_path": str(raw_path),
        "plot_error": plot_error,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "output_root": str(output_root), **meta}), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
