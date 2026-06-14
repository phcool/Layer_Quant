from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_quant.int4_8b_attention import patch_nemotron_h_attention_int4_kv
from hybrid_quant.mamba_state_8b_kernel import (
    patch_nemotron_h_mamba_decode_state_kernel,
    register_mamba_state_kernel_caches,
)
from hybrid_quant.nemotron_8b_decode_eval import layer_groups, make_hybrid_cache, patch_attention_cache_in_blocks
from run_nemotron_8b_decode_degradation import MODEL_PATH
from scripts.latency.run_nemotron_8b_latency import make_batch_ids


QUANTIZATION_MODES = {
    "none": ("none", "none"),
    "kv_int4": ("int4", "none"),
    "state_mx8": ("none", "mx8"),
    "kv_int4_state_mx8": ("int4", "mx8"),
}


def cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=MODEL_PATH)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--mode", choices=sorted(QUANTIZATION_MODES), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    kv_quantization, state_quantization = QUANTIZATION_MODES[args.mode]
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.repo,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
        patch_attention_cache_in_blocks(model)
        if kv_quantization == "int4":
            patch_nemotron_h_attention_int4_kv(model, group_size=args.kv_group_size)
        if state_quantization == "mx8":
            patch_nemotron_h_mamba_decode_state_kernel(model, group_size=16, stochastic=True, seed=args.seed)

        config = AutoConfig.from_pretrained(args.repo, trust_remote_code=True)
        mamba_layers, attention_layers, _ = layer_groups(config)
        mode_by_layer = {idx: "mx8" for idx in mamba_layers} if state_quantization == "mx8" else {}
        ids = make_batch_ids(
            args.repo,
            args.dataset,
            args.batch_size,
            args.sequence_length,
            args.warmup_steps + args.decode_steps,
        ).to(device)
        positions = torch.arange(
            args.sequence_length + args.warmup_steps + args.decode_steps + 1,
            device=device,
            dtype=torch.long,
        )
        cache = make_hybrid_cache(model, batch_size=args.batch_size)

        with torch.inference_mode():
            model(
                input_ids=ids[:, : args.sequence_length],
                cache_params=cache,
                cache_position=positions[: args.sequence_length],
                use_cache=True,
                return_dict=True,
            )
            if state_quantization == "mx8":
                register_mamba_state_kernel_caches(model, cache, mode_by_layer)

            next_pos = args.sequence_length
            for _ in range(args.warmup_steps):
                model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=positions[next_pos : next_pos + 1],
                    use_cache=True,
                    return_dict=True,
                )
                next_pos += 1
            torch.cuda.synchronize(device)

            print(
                json.dumps(
                    {
                        "event": "profile_start",
                        "mode": args.mode,
                        "kv_quantization": kv_quantization,
                        "state_quantization": state_quantization,
                        "batch_size": args.batch_size,
                        "sequence_length": args.sequence_length,
                        "decode_steps": args.decode_steps,
                        "warmup_steps": args.warmup_steps,
                    }
                ),
                flush=True,
            )
            cuda_profiler_start()
            for step_idx in range(args.decode_steps):
                torch.cuda.nvtx.range_push(f"decode_step_{step_idx}")
                model(
                    input_ids=ids[:, next_pos : next_pos + 1],
                    cache_params=cache,
                    cache_position=positions[next_pos : next_pos + 1],
                    use_cache=True,
                    return_dict=True,
                )
                torch.cuda.nvtx.range_pop()
                next_pos += 1
            cuda_profiler_stop()
            torch.cuda.synchronize(device)
            print(json.dumps({"event": "profile_stop", "mode": args.mode}), flush=True)
    finally:
        if model is not None:
            module = __import__(model.__class__.__module__, fromlist=["selective_state_update"])
            if hasattr(module, "_mamba_decode_state_kernel_caches"):
                module._mamba_decode_state_kernel_caches = {}
            if hasattr(model, "_mamba_decode_state_kernel_caches"):
                model._mamba_decode_state_kernel_caches = {}
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
