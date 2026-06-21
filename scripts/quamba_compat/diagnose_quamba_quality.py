from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.nemotron_8b_decode_eval import layer_groups, patch_attention_cache_in_blocks
from scripts.quamba_compat.eval_nemotron_quamba_w8a8_latency_ppl import (
    compute_quality,
    make_eval_ids,
    prepare_quamba_w8a8_model,
    run_decode,
)
from scripts.quamba_compat.run_nemotron_quamba_full_w8a8 import (
    CALIB_SOURCE_LOCAL,
    CALIB_SOURCE_QUAMBA_PILE,
    CALIB_SOURCE_WIKITEXT,
    apply_local_out_hadamard,
    build_reorder_params,
    calibrate_quamba2_scales,
    configure_nemotron_mamba_layers,
    ensure_quamba_on_path,
    get_channel_stats_for_reorder,
    load_calibration_dataset,
    quantize_all_mamba_layers,
    reorder_nemotron_mamba_layers,
)

MODEL_PATH = "/scratch2/wl730/models/nemotron-h-8b"
PRESET_WIKITEXT_LOCAL_OUT_HAD_PPL = "wikitext_local_out_had_ppl"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def apply_preset(args) -> None:
    if args.preset is None:
        return
    if args.preset == PRESET_WIKITEXT_LOCAL_OUT_HAD_PPL:
        args.modes = "baseline_fp16,w8a8_local_out_had_all"
        args.output_root = "results/quamba_compat/round15_wikitext2048_s8_w8a8_local_out_had_ppl"
        args.context_length = 2048
        args.batch_size = 2
        args.decode_steps = 64
        args.num_calib_samples = 8
        args.calib_seq_len = 2048
        args.calib_source = CALIB_SOURCE_WIKITEXT
        args.no_group_heads = False
        return
    raise ValueError(f"Unsupported preset: {args.preset}")


def load_fp16_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda().eval()
    patch_attention_cache_in_blocks(model)
    return model


def cleanup(model) -> None:
    if model is not None:
        for layer in getattr(getattr(model, "backbone", None), "layers", []):
            mixer = getattr(layer, "mixer", None)
            if hasattr(mixer, "_nemotron_cache_by_owner"):
                mixer._nemotron_cache_by_owner.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def fuse_had_matrices_mamba_only(model, mamba_layers: list[int]) -> list[int]:
    from quamba.qLinearLayer import HadLinear

    fused_layers = []
    for layer_idx in mamba_layers:
        mixer = model.backbone.layers[layer_idx].mixer
        fused = False
        in_proj = getattr(mixer, "in_proj", None)
        out_proj = getattr(mixer, "out_proj", None)
        if isinstance(in_proj, HadLinear):
            in_proj.fuse_hadamard()
            fused = True
        if isinstance(out_proj, HadLinear):
            out_proj.fuse_hadamard()
            fused = True
        if fused:
            fused_layers.append(layer_idx)
    return fused_layers


class FakeQuantOutProj(torch.nn.Module):
    """Diagnostic out_proj replacement.

    This module is intentionally not a performance kernel. It uses PyTorch
    fake quantization to isolate whether out_proj quality loss comes from
    activation scale, rounding, or weight quantization.
    """

    def __init__(
        self,
        linear: torch.nn.Linear,
        input_scale: torch.Tensor,
        *,
        activation_mode: str,
        rounding: str,
        weight_mode: str,
    ) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("FakeQuantOutProj only supports bias-free out_proj.")
        if activation_mode != "static":
            raise ValueError(f"Unsupported activation_mode={activation_mode!r}")
        if rounding not in {"round", "truncate"}:
            raise ValueError(f"Unsupported rounding={rounding!r}")
        if weight_mode not in {"fp16", "w8"}:
            raise ValueError(f"Unsupported weight_mode={weight_mode!r}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.activation_mode = activation_mode
        self.rounding = rounding
        self.weight_mode = weight_mode
        self.register_buffer("input_scale", input_scale.detach().float().reshape(()))

        weight = linear.weight.detach()
        if weight_mode == "fp16":
            self.register_buffer("weight", weight.to(torch.float16).contiguous())
        else:
            weight_scale = weight.float().abs().amax() / 127.0
            weight_scale = torch.clamp(weight_scale, min=torch.tensor(1.0e-12, device=weight.device))
            q_weight = torch.round(weight.float() / weight_scale).clamp(-128, 127)
            deq_weight = (q_weight * weight_scale).to(torch.float16).contiguous()
            self.register_buffer("weight", deq_weight)
            self.register_buffer("weight_scale", weight_scale.float().reshape(()))

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_scale.to(device=x.device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._scale(x)
        q = (x.float() / scale).clamp(-128, 127)
        if self.rounding == "round":
            q = torch.round(q)
        else:
            q = q.to(torch.int8).to(torch.float32)
        deq = (q * scale).to(torch.float16)
        return torch.nn.functional.linear(deq, self.weight)


@torch.no_grad()
def prepare_w8a8_local_out_had_model(model, tokenizer, mamba_layers: list[int], args) -> dict[str, Any]:
    configured_layers = configure_nemotron_mamba_layers(
        model,
        mamba_layers,
        use_had_transform=False,
    )
    calibration_dataset = load_calibration_dataset(args.calib_source)
    reorder_params = None
    grouped_available = False
    if not args.no_group_heads:
        channel_stats = get_channel_stats_for_reorder(
            model,
            tokenizer,
            mamba_layers,
            args.num_calib_samples,
            args.calib_seq_len,
            calibration_dataset,
        )
        reorder_params = build_reorder_params(model, mamba_layers, channel_stats)
        reorder_nemotron_mamba_layers(model, mamba_layers, reorder_params)
        grouped_available = True

    local_out_had_layers = apply_local_out_hadamard(model, mamba_layers)
    act_scales = calibrate_quamba2_scales(
        model,
        tokenizer,
        mamba_layers,
        reorder_params,
        args.num_calib_samples,
        args.calib_seq_len,
        calibration_dataset,
    )
    layer_summaries = quantize_all_mamba_layers(
        model,
        mamba_layers,
        act_scales,
        use_had_transform=True,
        require_grouped=not args.no_group_heads,
    )
    replaced_mamba_layers = [
        idx
        for idx in mamba_layers
        if model.backbone.layers[idx].mixer.__class__.__name__ == "W8A8QMamba2"
    ]
    if replaced_mamba_layers != mamba_layers:
        raise RuntimeError(
            f"Not all Mamba layers were replaced: expected {mamba_layers}, got {replaced_mamba_layers}"
        )
    return {
        "configured_layers": configured_layers,
        "local_out_had_layers": local_out_had_layers,
        "replaced_mamba_layers": replaced_mamba_layers,
        "layer_summaries": layer_summaries,
        "num_calib_samples": args.num_calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "calib_source": args.calib_source,
        "use_group_heads": not args.no_group_heads,
        "grouped_available": grouped_available,
        "use_had_transform": "local_out_only",
    }


@torch.no_grad()
def prepare_w8a8_fp_outproj_model(model, tokenizer, mamba_layers: list[int], args) -> dict[str, Any]:
    configured_layers = configure_nemotron_mamba_layers(
        model,
        mamba_layers,
        use_had_transform=False,
    )
    calibration_dataset = load_calibration_dataset(args.calib_source)
    reorder_params = None
    grouped_available = False
    if not args.no_group_heads:
        channel_stats = get_channel_stats_for_reorder(
            model,
            tokenizer,
            mamba_layers,
            args.num_calib_samples,
            args.calib_seq_len,
            calibration_dataset,
        )
        reorder_params = build_reorder_params(model, mamba_layers, channel_stats)
        reorder_nemotron_mamba_layers(model, mamba_layers, reorder_params)
        grouped_available = True

    fp_out_projs = {
        layer_idx: copy.deepcopy(model.backbone.layers[layer_idx].mixer.out_proj).eval()
        for layer_idx in mamba_layers
    }
    act_scales = calibrate_quamba2_scales(
        model,
        tokenizer,
        mamba_layers,
        reorder_params,
        args.num_calib_samples,
        args.calib_seq_len,
        calibration_dataset,
    )
    layer_summaries = quantize_all_mamba_layers(
        model,
        mamba_layers,
        act_scales,
        use_had_transform=False,
        require_grouped=not args.no_group_heads,
    )
    restored_layers = []
    device = next(model.parameters()).device
    for layer_idx in mamba_layers:
        mixer = model.backbone.layers[layer_idx].mixer
        if mixer.__class__.__name__ != "W8A8QMamba2":
            raise RuntimeError(f"Layer {layer_idx} is not W8A8QMamba2 after quantization.")
        mixer.had = torch.nn.Identity()
        mixer.out_proj = fp_out_projs[layer_idx].to(device).eval()
        restored_layers.append(layer_idx)
    return {
        "configured_layers": configured_layers,
        "restored_fp_outproj_layers": restored_layers,
        "layer_summaries": layer_summaries,
        "num_calib_samples": args.num_calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "calib_source": args.calib_source,
        "use_group_heads": not args.no_group_heads,
        "grouped_available": grouped_available,
        "use_had_transform": False,
        "ablation": "restore_fp_outproj_after_w8a8",
    }


@torch.no_grad()
def prepare_w8a8_fake_outproj_model(
    model,
    tokenizer,
    mamba_layers: list[int],
    args,
    *,
    activation_mode: str,
    rounding: str,
    weight_mode: str,
) -> dict[str, Any]:
    configured_layers = configure_nemotron_mamba_layers(
        model,
        mamba_layers,
        use_had_transform=False,
    )
    calibration_dataset = load_calibration_dataset(args.calib_source)
    reorder_params = None
    grouped_available = False
    if not args.no_group_heads:
        channel_stats = get_channel_stats_for_reorder(
            model,
            tokenizer,
            mamba_layers,
            args.num_calib_samples,
            args.calib_seq_len,
            calibration_dataset,
        )
        reorder_params = build_reorder_params(model, mamba_layers, channel_stats)
        reorder_nemotron_mamba_layers(model, mamba_layers, reorder_params)
        grouped_available = True

    fp_out_projs = {
        layer_idx: copy.deepcopy(model.backbone.layers[layer_idx].mixer.out_proj).eval()
        for layer_idx in mamba_layers
    }
    act_scales = calibrate_quamba2_scales(
        model,
        tokenizer,
        mamba_layers,
        reorder_params,
        args.num_calib_samples,
        args.calib_seq_len,
        calibration_dataset,
    )
    layer_summaries = quantize_all_mamba_layers(
        model,
        mamba_layers,
        act_scales,
        use_had_transform=False,
        require_grouped=not args.no_group_heads,
    )
    replaced_layers = []
    device = next(model.parameters()).device
    for layer_idx in mamba_layers:
        mixer = model.backbone.layers[layer_idx].mixer
        if mixer.__class__.__name__ != "W8A8QMamba2":
            raise RuntimeError(f"Layer {layer_idx} is not W8A8QMamba2 after quantization.")
        mixer.had = torch.nn.Identity()
        mixer.out_proj = FakeQuantOutProj(
            fp_out_projs[layer_idx].to(device).eval(),
            act_scales[layer_idx]["out_proj:input"],
            activation_mode=activation_mode,
            rounding=rounding,
            weight_mode=weight_mode,
        ).to(device).eval()
        replaced_layers.append(layer_idx)
    return {
        "configured_layers": configured_layers,
        "fake_outproj_layers": replaced_layers,
        "layer_summaries": layer_summaries,
        "num_calib_samples": args.num_calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "calib_source": args.calib_source,
        "use_group_heads": not args.no_group_heads,
        "grouped_available": grouped_available,
        "use_had_transform": False,
        "ablation": "fake_quant_outproj_after_w8a8",
        "outproj_activation_mode": activation_mode,
        "outproj_rounding": rounding,
        "outproj_weight_mode": weight_mode,
    }


@torch.no_grad()
def prepare_mode(model, tokenizer, mode: str, args) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    mamba_layers, attention_layers, mlp_layers = layer_groups(config)
    meta: dict[str, Any] = {
        "mode": mode,
        "mamba_layers": mamba_layers,
        "attention_layers": attention_layers,
        "mlp_layers": mlp_layers,
    }
    if mode == "baseline_fp16":
        return meta
    if mode == "simple_no_had_all":
        meta["configured_layers"] = configure_nemotron_mamba_layers(
            model,
            mamba_layers,
            use_had_transform=False,
        )
        return meta
    if mode == "simple_no_had_reorder_all":
        if args.no_group_heads:
            raise ValueError("simple_no_had_reorder_all requires CAWR grouped reorder; do not pass --no-group-heads.")
        meta["configured_layers"] = configure_nemotron_mamba_layers(
            model,
            mamba_layers,
            use_had_transform=False,
        )
        calibration_dataset = load_calibration_dataset(args.calib_source)
        channel_stats = get_channel_stats_for_reorder(
            model,
            tokenizer,
            mamba_layers,
            args.num_calib_samples,
            args.calib_seq_len,
            calibration_dataset,
        )
        reorder_params = build_reorder_params(model, mamba_layers, channel_stats)
        reorder_nemotron_mamba_layers(model, mamba_layers, reorder_params)
        meta["reorder_only_layers"] = mamba_layers
        meta["calib_source"] = args.calib_source
        meta["num_calib_samples"] = args.num_calib_samples
        meta["calib_seq_len"] = args.calib_seq_len
        meta["use_group_heads"] = True
        return meta
    if mode == "simple_had_all":
        meta["configured_layers"] = configure_nemotron_mamba_layers(
            model,
            mamba_layers,
            use_had_transform=True,
        )
        return meta
    if mode == "simple_local_out_had_all":
        meta["configured_layers"] = configure_nemotron_mamba_layers(
            model,
            mamba_layers,
            use_had_transform=False,
        )
        meta["local_out_had_layers"] = apply_local_out_hadamard(model, mamba_layers)
        return meta
    if mode == "simple_no_had_layer0":
        meta["configured_layers"] = configure_nemotron_mamba_layers(
            model,
            [mamba_layers[0]],
            use_had_transform=False,
        )
        return meta
    if mode == "w8a8_no_had_all":
        meta.update(
            prepare_quamba_w8a8_model(
                model,
                tokenizer,
                num_calib_samples=args.num_calib_samples,
                calib_seq_len=args.calib_seq_len,
                use_group_heads=not args.no_group_heads,
                use_had_transform=False,
                calib_source=args.calib_source,
            )
        )
        return meta
    if mode == "w8a8_no_had_fp_outproj_all":
        meta.update(prepare_w8a8_fp_outproj_model(model, tokenizer, mamba_layers, args))
        return meta
    if mode == "w8a8_no_had_outproj_fpweight_static_round_all":
        meta.update(
            prepare_w8a8_fake_outproj_model(
                model,
                tokenizer,
                mamba_layers,
                args,
                activation_mode="static",
                rounding="round",
                weight_mode="fp16",
            )
        )
        return meta
    if mode == "w8a8_no_had_outproj_w8_static_round_all":
        meta.update(
            prepare_w8a8_fake_outproj_model(
                model,
                tokenizer,
                mamba_layers,
                args,
                activation_mode="static",
                rounding="round",
                weight_mode="w8",
            )
        )
        return meta
    if mode == "w8a8_local_out_had_all":
        meta.update(prepare_w8a8_local_out_had_model(model, tokenizer, mamba_layers, args))
        return meta
    if mode == "w8a8_had_fused_all":
        meta["configured_layers"] = configure_nemotron_mamba_layers(
            model,
            mamba_layers,
            use_had_transform=True,
        )
        channel_stats = get_channel_stats_for_reorder(
            model,
            tokenizer,
            mamba_layers,
            args.num_calib_samples,
            args.calib_seq_len,
            load_calibration_dataset(args.calib_source),
        )
        reorder_params = build_reorder_params(model, mamba_layers, channel_stats)
        reorder_nemotron_mamba_layers(model, mamba_layers, reorder_params)
        calibration_dataset = load_calibration_dataset(args.calib_source)
        act_scales = calibrate_quamba2_scales(
            model,
            tokenizer,
            mamba_layers,
            reorder_params,
            args.num_calib_samples,
            args.calib_seq_len,
            calibration_dataset,
        )
        meta["had_fused_layers"] = fuse_had_matrices_mamba_only(model, mamba_layers)
        meta["layer_summaries"] = quantize_all_mamba_layers(
            model,
            mamba_layers,
            act_scales,
            use_had_transform=True,
            require_grouped=True,
        )
        return meta
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Quamba quality collapse on Nemotron-H.")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--output-root", default="results/quamba_compat/round6_quality_diagnosis")
    parser.add_argument(
        "--preset",
        choices=[PRESET_WIKITEXT_LOCAL_OUT_HAD_PPL],
        default=None,
        help="Apply a named experiment setup. Explicit CLI values for preset-controlled fields are overwritten.",
    )
    parser.add_argument(
        "--modes",
        default="baseline_fp16,simple_no_had_layer0,simple_no_had_all,simple_had_all,w8a8_no_had_all,w8a8_had_fused_all",
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--num-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seq-len", type=int, default=128)
    parser.add_argument(
        "--calib-source",
        choices=[CALIB_SOURCE_LOCAL, CALIB_SOURCE_QUAMBA_PILE, CALIB_SOURCE_WIKITEXT],
        default=CALIB_SOURCE_LOCAL,
        help="Calibration source. quamba_pile matches Quamba's default monology/pile-uncopyrighted val split.",
    )
    parser.add_argument("--no-group-heads", action="store_true")
    args = parser.parse_args()
    apply_preset(args)

    ensure_quamba_on_path()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = make_eval_ids(
        tokenizer,
        args.dataset,
        args.batch_size,
        args.context_length,
        0,
        args.decode_steps,
    )

    rows = []
    baseline_logits = None
    baseline_targets = None
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        print(json.dumps({"event": "start_mode", "mode": mode}), flush=True)
        model = None
        try:
            model = load_fp16_model(args.model)
            meta = prepare_mode(model, tokenizer, mode, args)
            (output_root / f"{mode}_prepare.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
            )
            decode_row, logits, targets = run_decode(
                model,
                ids,
                context_length=args.context_length,
                warmup_steps=0,
                decode_steps=args.decode_steps,
                collect_logits=True,
            )
            if mode == "baseline_fp16":
                baseline_logits = logits
                baseline_targets = targets
                quality = compute_quality(logits, targets, None)
            else:
                if baseline_logits is None or baseline_targets is None:
                    raise RuntimeError("baseline_fp16 must run first.")
                quality = compute_quality(logits, targets, baseline_logits)
            row = {
                "mode": mode,
                "ce": quality["ce"],
                "ppl": quality["ppl"],
                "kl_vs_baseline": quality["kl_vs_baseline"],
                "top1_match_vs_baseline": quality["top1_match_vs_baseline"],
                "decode_ms_per_step": decode_row["decode_ms_per_step"],
                "prefill_latency_s": decode_row["prefill_latency_s"],
                "peak_memory_gib": decode_row["peak_memory_gib"],
            }
            rows.append(row)
            print(json.dumps({"event": "mode_done", **row}), flush=True)
        finally:
            cleanup(model)

    write_csv(output_root / "quality_diagnosis.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps({"args": vars(args), "quality": rows}, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"event": "done", "output_root": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
