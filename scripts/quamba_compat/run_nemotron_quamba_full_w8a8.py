from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.nemotron_8b_decode_eval import layer_groups

MODEL_PATH = "/scratch2/wl730/models/nemotron-h-8b"

CALIBRATION_TEXTS = [
    "Hybrid Mamba attention models combine recurrent state updates with sparse global attention.",
    "A quantized state space model must preserve long range information while reducing memory traffic.",
    "The researcher compared baseline decoding latency, kernel time, and perplexity across several settings.",
    "In this experiment, every Mamba block is calibrated before converting the model to a W8A8 kernel.",
    "Efficient sequence models need careful co-design between cache layout, quantization scales, and kernels.",
    "The final answer should report which parts are exactly aligned with the reference implementation.",
    "A robust implementation should fail loudly when grouped activation scales are unavailable.",
    "This prompt is used only for calibration and smoke testing, not for claiming final model quality.",
]

CALIB_SOURCE_LOCAL = "local_prompts"
CALIB_SOURCE_QUAMBA_PILE = "quamba_pile"
CALIB_SOURCE_WIKITEXT = "wikitext"


def ensure_quamba_on_path() -> None:
    quamba_root = os.environ.get("QUAMBA_ROOT")
    if quamba_root and quamba_root not in sys.path:
        sys.path.insert(0, quamba_root)


def patch_norm_api(norm: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(norm, "eps"):
        setattr(norm, "eps", float(getattr(norm, "variance_epsilon")))
    if not hasattr(norm, "bias"):
        setattr(norm, "bias", None)
    return norm


def patch_gated_norm_api(norm: torch.nn.Module) -> torch.nn.Module:
    patch_norm_api(norm)
    if not hasattr(norm, "norm_before_gate"):
        setattr(norm, "norm_before_gate", False)
    return norm


def make_mamba2_view(mixer: torch.nn.Module) -> SimpleNamespace:
    hidden_size = int(mixer.hidden_size)
    intermediate_size = int(mixer.intermediate_size)
    if intermediate_size % hidden_size != 0:
        raise ValueError(f"Cannot infer Mamba2 expand: {intermediate_size=} {hidden_size=}")
    patch_gated_norm_api(mixer.norm)
    return SimpleNamespace(
        d_model=hidden_size,
        d_state=int(mixer.ssm_state_size),
        d_conv=int(mixer.conv_kernel_size),
        conv_init=None,
        expand=intermediate_size // hidden_size,
        process_group=None,
        sequence_parallel=True,
        headdim=int(mixer.head_dim),
        d_ssm=intermediate_size,
        ngroups=int(mixer.n_groups),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_limit=tuple(mixer.time_step_limit),
        chunk_size=int(mixer.chunk_size),
        layer_idx=int(mixer.layer_idx),
        in_proj=mixer.in_proj,
        conv1d=mixer.conv1d,
        dt_bias=mixer.dt_bias,
        A_log=mixer.A_log,
        D=mixer.D,
        norm=mixer.norm,
        out_proj=mixer.out_proj,
    )


def patch_quamba_mixer_forward(mixer: torch.nn.Module) -> torch.nn.Module:
    if getattr(mixer, "_nemotron_cache_patched", False):
        return mixer
    original_forward = mixer.forward
    mixer._nemotron_orig_forward = original_forward
    mixer._nemotron_cache_by_owner = {}

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, **kwargs):
        if cache_params is None:
            return self._nemotron_orig_forward(hidden_states)
        cache_id = id(cache_params)
        inference_params = self._nemotron_cache_by_owner.get(cache_id)
        if inference_params is None:
            inference_params = SimpleNamespace(key_value_memory_dict={}, seqlen_offset=0)
            self._nemotron_cache_by_owner[cache_id] = inference_params
        else:
            # Quamba only checks whether seqlen_offset > 0 to select step()
            # during decode. Avoid cache_position.item() here because it can
            # synchronize the CPU with the GPU once per Mamba layer per token.
            inference_params.seqlen_offset = 1 if hidden_states.shape[1] == 1 else 0
        return self._nemotron_orig_forward(hidden_states, inference_params=inference_params)

    mixer.forward = MethodType(forward, mixer)
    mixer._nemotron_cache_patched = True
    return mixer


def load_calibration_dataset(calib_source: str):
    if calib_source == CALIB_SOURCE_LOCAL:
        return None
    if calib_source == CALIB_SOURCE_QUAMBA_PILE:
        from datasets import load_dataset

        dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
        return dataset.shuffle(seed=42)
    if calib_source == CALIB_SOURCE_WIKITEXT:
        texts = load_wikitext_texts("wikitext", split="train")
        joined = "\n\n".join(text for text in texts if text.strip())
        chars_per_sample = 16384
        if len(joined) < chars_per_sample:
            raise RuntimeError("WikiText calibration corpus is too small.")
        samples = []
        for sample_idx in range(1024):
            start = (sample_idx * chars_per_sample) % max(1, len(joined) - chars_per_sample)
            samples.append({"text": joined[start : start + chars_per_sample]})
        return samples
    raise ValueError(f"Unsupported calibration source: {calib_source!r}")


def calibration_ids(
    tokenizer,
    sample_idx: int,
    seq_len: int,
    device: torch.device,
    calibration_dataset=None,
) -> torch.Tensor:
    if calibration_dataset is None:
        text = CALIBRATION_TEXTS[sample_idx % len(CALIBRATION_TEXTS)]
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :seq_len]
    else:
        text = calibration_dataset[sample_idx]["text"]
        encoded = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids
    if encoded.shape[1] == 0:
        raise ValueError("Tokenizer returned an empty calibration sample.")
    if calibration_dataset is None and encoded.shape[1] < seq_len:
        pad = encoded[:, -1:].expand(encoded.shape[0], seq_len - encoded.shape[1])
        encoded = torch.cat([encoded, pad], dim=1)
    return encoded.to(device)


def make_nemotron_cache(model, batch_size: int, device: torch.device):
    module = sys.modules[model.__class__.__module__]
    cache_cls = getattr(module, "HybridMambaAttentionDynamicCache")
    dtype = next(model.parameters()).dtype
    return cache_cls(model.config, batch_size=batch_size, dtype=dtype, device=device)


@torch.no_grad()
def configure_nemotron_mamba_layers(model, mamba_layers: list[int], *, use_had_transform: bool) -> list[int]:
    from quamba.modelutils_mamba import fuse_ln_linear
    from quamba.qMamba2 import Mamba2Simple

    device = next(model.parameters()).device
    converted = []
    for layer_idx in tqdm(mamba_layers, desc="configure Mamba2Simple"):
        block = model.backbone.layers[layer_idx]
        if getattr(block, "block_type", None) != "mamba":
            raise ValueError(f"Layer {layer_idx} is not a Nemotron mamba block.")
        patch_norm_api(block.norm)
        patch_gated_norm_api(block.mixer.norm)
        fuse_ln_linear(block.norm, block.mixer.in_proj)
        simple = Mamba2Simple(make_mamba2_view(block.mixer), use_had_transform=use_had_transform).to(device).eval()
        patch_quamba_mixer_forward(simple)
        block.mixer = simple
        converted.append(layer_idx)
        gc.collect()
        torch.cuda.empty_cache()
    return converted


@torch.no_grad()
def apply_local_out_hadamard(model, mamba_layers: list[int]) -> list[int]:
    """Insert a block-local Hadamard before out_proj without changing residual coordinates.

    Quamba's paper path rotates the model more globally. That is unsafe when only
    Nemotron-H Mamba blocks are replaced inside a hybrid model, because attention
    and MLP blocks still expect the original residual basis. This helper only
    rotates the activation immediately before out_proj and fuses the inverse
    rotation into out_proj's input dimension, so the FP block remains equivalent:

        out_proj(H y; W H) == out_proj(y; W)
    """
    from quamba.qHadamard import Hadamard
    from quamba.qLinearLayer import HadLinear

    device = next(model.parameters()).device
    converted = []
    for layer_idx in tqdm(mamba_layers, desc="local out Hadamard"):
        mixer = model.backbone.layers[layer_idx].mixer
        mixer.had = Hadamard(mixer.out_proj.in_features).to(device)
        had_out_proj = HadLinear(
            mixer.out_proj,
            input_transform=True,
            output_transform=False,
        ).to(device)
        had_out_proj.fuse_hadamard()
        mixer.out_proj = had_out_proj
        converted.append(layer_idx)
    return converted


@torch.no_grad()
def get_channel_stats_for_reorder(
    model,
    tokenizer,
    mamba_layers: list[int],
    num_samples: int,
    seq_len: int,
    calibration_dataset=None,
) -> dict[int, torch.Tensor]:
    from quamba.qActLayer import ActIdentity

    device = next(model.parameters()).device
    stats: dict[int, torch.Tensor] = {}
    hooks = []

    def stat_hook(_module, inputs, _outputs, layer_idx: int):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        hidden_dim = x.shape[-1]
        x = x.reshape(-1, hidden_dim)
        current = torch.mean(x.detach().abs(), dim=0).float().cpu()
        if layer_idx in stats:
            stats[layer_idx] = torch.maximum(stats[layer_idx], current)
        else:
            stats[layer_idx] = current

    for layer_idx in mamba_layers:
        act = model.backbone.layers[layer_idx].mixer.x_conv_out
        if not isinstance(act, ActIdentity):
            raise TypeError(f"Layer {layer_idx} x_conv_out is not Quamba ActIdentity.")
        hooks.append(act.register_forward_hook(partial(stat_hook, layer_idx=layer_idx)))

    for sample_idx in tqdm(range(num_samples), desc="reorder stats"):
        input_ids = calibration_ids(tokenizer, sample_idx, seq_len, device, calibration_dataset)
        model(input_ids=input_ids, use_cache=False, return_dict=True)

    for hook in hooks:
        hook.remove()

    missing = sorted(set(mamba_layers) - set(stats))
    if missing:
        raise RuntimeError(f"Missing reorder x_conv_out stats for Mamba layers: {missing}")
    return stats


@torch.no_grad()
def build_reorder_params(model, mamba_layers: list[int], channel_stats: dict[int, torch.Tensor]) -> dict[str, list[Any]]:
    from quamba.reorder_utils import group_wise_sort_indices

    num_layers = len(model.backbone.layers)
    params: dict[str, list[Any]] = {
        "head_groups": [None for _ in range(num_layers)],
        "head_index": [None for _ in range(num_layers)],
        "channel_group": [None for _ in range(num_layers)],
        "channel_index": [None for _ in range(num_layers)],
    }
    for layer_idx in mamba_layers:
        mixer = model.backbone.layers[layer_idx].mixer
        # Quamba's group_wise_sort_indices builds some index tensors on CPU, so keep
        # the channel statistics on CPU to follow the official path.
        scales = channel_stats[layer_idx].cpu()
        channel_index, head_groups, head_index, dim_groups = group_wise_sort_indices(
            scales, mixer.headdim, mixer.ngroups
        )
        params["head_groups"][layer_idx] = head_groups
        params["head_index"][layer_idx] = head_index
        params["channel_group"][layer_idx] = dim_groups
        params["channel_index"][layer_idx] = channel_index
    return params


@torch.no_grad()
def reorder_nemotron_mamba_layers(model, mamba_layers: list[int], reorder_params: dict[str, list[Any]]) -> None:
    from quamba.reorder_utils import reorder_conv, reorder_linear, reorder_norm

    for layer_idx in tqdm(mamba_layers, desc="reorder Mamba2"):
        mixer = model.backbone.layers[layer_idx].mixer
        head_idx = reorder_params["head_index"][layer_idx]
        ch_idx = reorder_params["channel_index"][layer_idx]
        if head_idx is None or ch_idx is None:
            raise RuntimeError(f"Missing reorder params for Mamba layer {layer_idx}.")

        in_proj_output_index = torch.arange(mixer.in_proj.out_features, device=mixer.in_proj.weight.device)
        in_proj_output_index[0 : mixer.d_ssm] = ch_idx
        in_proj_output_index[mixer.d_ssm : 2 * mixer.d_ssm] = ch_idx + mixer.d_ssm
        dt_offset = 2 * mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state
        in_proj_output_index[-mixer.nheads :] = head_idx + dt_offset
        reorder_linear(mixer.in_proj, out_reorder_index=in_proj_output_index)

        conv1d_indices = torch.arange(mixer.conv1d.in_channels, device=mixer.conv1d.weight.device)
        conv1d_indices[0 : mixer.d_ssm] = ch_idx
        reorder_conv(mixer.conv1d, reorder_index=conv1d_indices)

        mixer.A_log.data = mixer.A_log[head_idx].data
        mixer.D.data = mixer.D[head_idx].data
        mixer.dt_bias.data = mixer.dt_bias[head_idx].data
        reorder_norm(mixer.norm, reorder_index=ch_idx)
        reorder_linear(mixer.out_proj, in_reorder_index=ch_idx)


@torch.no_grad()
def calibrate_quamba2_scales(
    model,
    tokenizer,
    mamba_layers: list[int],
    reorder_params: dict[str, list[Any]] | None,
    num_samples: int,
    seq_len: int,
    calibration_dataset=None,
) -> list[dict[str, Any]]:
    from quamba.observer import (
        CachedStatesCrossHeadMinmaxObserver,
        CrossHeadMinmaxObserver,
        PerSSDGroupObserver,
        PerTensorMinmaxObserver,
        PerTensorPercentileObserver,
    )
    from quamba.qActLayer import ActIdentity

    device = next(model.parameters()).device
    layers = model.backbone.layers
    observers: list[dict[str, Any]] = [{} for _ in range(len(layers))]
    hooks = []
    use_grouped = reorder_params is not None

    def stat_hook(_module, inputs, outputs, op: str, layer_idx: int):
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        observers[layer_idx][op + ":input"].update(x_in.detach())
        x_out = outputs[0] if isinstance(outputs, tuple) else outputs
        observers[layer_idx][op + ":output"].update(x_out.detach())

    for layer_idx in mamba_layers:
        mixer = layers[layer_idx].mixer
        head_groups = reorder_params["head_groups"][layer_idx] if use_grouped else None
        channel_group = reorder_params["channel_group"][layer_idx] if use_grouped else None
        for name, module in mixer.named_modules():
            if not isinstance(module, (torch.nn.Linear, ActIdentity)):
                continue
            op = name.split(".")[0]
            if use_grouped and op == "x_conv_out":
                observers[layer_idx][op + ":input"] = CrossHeadMinmaxObserver(
                    n_bits=8,
                    clip_ratio=1.0,
                    sym=True,
                    ngroups=mixer.ngroups,
                    headdim=mixer.headdim,
                    head_groups=head_groups,
                    channel_group=channel_group,
                )
            elif use_grouped and op in {"B_conv_out", "C_conv_out"}:
                observers[layer_idx][op + ":input"] = PerSSDGroupObserver(
                    n_bits=8,
                    clip_ratio=1.0,
                    sym=True,
                    dstate=mixer.d_state,
                )
            elif use_grouped and op == "ssm_state_act":
                observers[layer_idx][op + ":input"] = CachedStatesCrossHeadMinmaxObserver(
                    n_bits=8,
                    clip_ratio=1.0,
                    sym=True,
                    ngroups=mixer.ngroups,
                    headdim=mixer.headdim,
                    dstate=mixer.d_state,
                    head_groups=head_groups,
                    channel_group=channel_group,
                )
            elif (not use_grouped) and op in {"x_conv_out", "ssm_state_act"}:
                observers[layer_idx][op + ":input"] = PerTensorPercentileObserver(
                    n_bits=8,
                    clip_ratio=1.0,
                    sym=True,
                    percentile_alpha=0.9995,
                )
            else:
                observers[layer_idx][op + ":input"] = PerTensorMinmaxObserver(
                    n_bits=8,
                    clip_ratio=1.0,
                    sym=True,
                )
            observers[layer_idx][op + ":output"] = PerTensorMinmaxObserver(
                n_bits=8,
                clip_ratio=1.0,
                sym=True,
            )
            hooks.append(module.register_forward_hook(partial(stat_hook, op=op, layer_idx=layer_idx)))

    for sample_idx in tqdm(range(num_samples), desc="activation calibration"):
        input_ids = calibration_ids(tokenizer, sample_idx, seq_len, device, calibration_dataset)
        cache_position = torch.arange(input_ids.shape[1], device=device)
        cache_params = make_nemotron_cache(model, batch_size=input_ids.shape[0], device=device)
        model(
            input_ids=input_ids,
            cache_params=cache_params,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

    for hook in hooks:
        hook.remove()

    act_scales: list[dict[str, Any]] = [{} for _ in range(len(layers))]
    for layer_idx in mamba_layers:
        missing_stats = [name for name, observer in observers[layer_idx].items() if not observer.has_statistic]
        if missing_stats:
            raise RuntimeError(f"Layer {layer_idx} missing observer statistics: {missing_stats}")
        for name, observer in observers[layer_idx].items():
            scale, _base = observer.get_quantization_parameters()
            act_scales[layer_idx][name] = scale
    return act_scales


def is_grouped_scale(scale: Any) -> bool:
    return isinstance(scale, list)


def scale_summary(scale: Any) -> Any:
    if torch.is_tensor(scale):
        return {"kind": "tensor", "shape": list(scale.shape), "dtype": str(scale.dtype)}
    if isinstance(scale, list):
        return {"kind": "grouped_list", "ssd_groups": len(scale)}
    return {"kind": type(scale).__name__}


@torch.no_grad()
def quantize_all_mamba_layers(
    model,
    mamba_layers: list[int],
    act_scales: list[dict[str, Any]],
    *,
    use_had_transform: bool,
    require_grouped: bool,
) -> list[dict[str, Any]]:
    from quamba.qMamba2 import W8A8QMamba2
    from quamba.qNorm import QRMSNorm

    device = next(model.parameters()).device
    summaries = []
    for layer_idx in tqdm(mamba_layers, desc="quantize W8A8"):
        block = model.backbone.layers[layer_idx]
        scales = act_scales[layer_idx]
        required_keys = [
            "in_proj:input",
            "z_act:input",
            "z_act:output",
            "x_conv_in:input",
            "x_conv_in:output",
            "B_conv_in:input",
            "B_conv_in:output",
            "C_conv_in:input",
            "C_conv_in:output",
            "x_conv_out:input",
            "B_conv_out:input",
            "C_conv_out:input",
            "dt_act:input",
            "dt_act:output",
            "ssm_state_act:input",
            "out_proj:input",
        ]
        missing = [key for key in required_keys if key not in scales]
        if missing:
            raise RuntimeError(f"Layer {layer_idx} missing Quamba act scales: {missing}")
        if require_grouped and not is_grouped_scale(scales["x_conv_out:input"]):
            raise RuntimeError(
                f"Layer {layer_idx} expected grouped x_conv_out scale but got {scale_summary(scales['x_conv_out:input'])}"
            )

        block.norm = QRMSNorm.from_fp16(
            patch_norm_api(block.norm),
            output_scale=scales["in_proj:input"].item(),
        ).to(device)
        q_mixer = W8A8QMamba2.from_fp16(
            originalLayer=block.mixer,
            act_scales=scales,
            use_had_transform=use_had_transform,
        ).to(device).eval()
        patch_quamba_mixer_forward(q_mixer)
        block.mixer = q_mixer
        summaries.append(
            {
                "layer": layer_idx,
                "norm_class": block.norm.__class__.__name__,
                "mixer_class": block.mixer.__class__.__name__,
                "x_conv_out_scale": scale_summary(scales["x_conv_out:input"]),
                "ssm_state_scale": scale_summary(scales["ssm_state_act:input"]),
                "conv_weight_dtype": str(block.mixer.conv1d.weight.dtype),
                "in_proj_weight_dtype": str(block.mixer.in_proj.weight.dtype),
                "out_proj_weight_dtype": str(block.mixer.out_proj.weight.dtype),
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
    return summaries


@torch.no_grad()
def run_smoke(model, tokenizer, seq_len: int, max_new_tokens: int, calibration_dataset=None) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_ids = calibration_ids(tokenizer, 0, seq_len, device, calibration_dataset)
    cache_position = torch.arange(input_ids.shape[1], device=device)
    cache_params = make_nemotron_cache(model, batch_size=input_ids.shape[0], device=device)
    outputs = model(
        input_ids=input_ids,
        cache_params=cache_params,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_outputs = model(
        input_ids=next_token,
        cache_params=cache_params,
        cache_position=torch.tensor([input_ids.shape[1]], device=device, dtype=torch.long),
        use_cache=True,
        return_dict=True,
    )
    generated = model.generate(
        input_ids=input_ids[:, : min(16, input_ids.shape[1])],
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    return {
        "prefill_logits_shape": list(outputs.logits.shape),
        "manual_decode_logits_shape": list(decode_outputs.logits.shape),
        "generated_shape": list(generated.shape),
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quamba-aligned W8A8 replacement for all Nemotron-H Mamba layers.")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output", default="results/quamba_compat/round4_full_w8a8/result.json")
    parser.add_argument("--num-calib-samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--calib-source",
        choices=[CALIB_SOURCE_LOCAL, CALIB_SOURCE_QUAMBA_PILE, CALIB_SOURCE_WIKITEXT],
        default=CALIB_SOURCE_LOCAL,
        help="Calibration source. quamba_pile matches Quamba's default monology/pile-uncopyrighted val split.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--no-group-heads", action="store_true", help="Disable Quamba2 grouped/reordered activation scales.")
    parser.add_argument("--no-had", action="store_true", help="Disable Quamba Hadamard path. This is not paper-aligned.")
    args = parser.parse_args()

    ensure_quamba_on_path()
    use_group_heads = not args.no_group_heads
    use_had_transform = not args.no_had

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    mamba_layers, attention_layers, mlp_layers = layer_groups(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda().eval()

    configured_layers = configure_nemotron_mamba_layers(
        model,
        mamba_layers,
        use_had_transform=use_had_transform,
    )
    calibration_dataset = load_calibration_dataset(args.calib_source)

    reorder_params = None
    grouped_available = False
    if use_group_heads:
        channel_stats = get_channel_stats_for_reorder(
            model,
            tokenizer,
            mamba_layers,
            args.num_calib_samples,
            args.seq_len,
            calibration_dataset,
        )
        reorder_params = build_reorder_params(model, mamba_layers, channel_stats)
        reorder_nemotron_mamba_layers(model, mamba_layers, reorder_params)
        grouped_available = True

    act_scales = calibrate_quamba2_scales(
        model,
        tokenizer,
        mamba_layers,
        reorder_params,
        args.num_calib_samples,
        args.seq_len,
        calibration_dataset,
    )
    layer_summaries = quantize_all_mamba_layers(
        model,
        mamba_layers,
        act_scales,
        use_had_transform=use_had_transform,
        require_grouped=use_group_heads,
    )
    smoke = run_smoke(model, tokenizer, args.seq_len, args.max_new_tokens, calibration_dataset)

    replaced_mamba_layers = [
        idx
        for idx in mamba_layers
        if model.backbone.layers[idx].mixer.__class__.__name__ == "W8A8QMamba2"
    ]
    if replaced_mamba_layers != mamba_layers:
        raise RuntimeError(
            f"Not all Mamba layers were replaced: expected {mamba_layers}, got {replaced_mamba_layers}"
        )

    result = {
        "mode": "quamba_aligned_w8a8_all_mamba",
        "model": args.model,
        "num_layers": len(model.backbone.layers),
        "mamba_layers": mamba_layers,
        "attention_layers": attention_layers,
        "mlp_layers": mlp_layers,
        "configured_layers": configured_layers,
        "replaced_mamba_layers": replaced_mamba_layers,
        "num_calib_samples": args.num_calib_samples,
        "seq_len": args.seq_len,
        "calib_source": args.calib_source,
        "use_had_transform": use_had_transform,
        "use_group_heads": use_group_heads,
        "grouped_available": grouped_available,
        "layer_summaries": layer_summaries,
        "smoke": smoke,
        "known_differences_from_quamba_official": [
            "Only Nemotron-H Mamba blocks are quantized; attention, MLP, embedding, final norm, and lm_head stay in the original dtype.",
            "Nemotron-H uses cache_params/cache_position, so W8A8QMamba2.forward is monkey-patched only to adapt the call signature to Quamba inference_params.",
            "Calibration can use Quamba's default monology/pile-uncopyrighted dataset via --calib-source quamba_pile; local prompt strings remain available for quick smoke tests.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
