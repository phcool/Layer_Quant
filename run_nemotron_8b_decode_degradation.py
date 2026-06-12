from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.int4_8b_attention import patch_nemotron_h_attention_int4_kv
from hybrid_quant.nemotron_8b_decode_eval import (
    adjacent_mamba_layers,
    layer_groups,
    make_hybrid_cache,
    patch_attention_cache_in_blocks,
)
from hybrid_quant.mamba_state_8b_kernel import (
    patch_nemotron_h_mamba_decode_state_kernel,
    register_mamba_state_kernel_caches,
)


MODEL_PATH = "/scratch2/wl730/models/nemotron-h-8b"


def parse_steps(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def experiment_plan(name: str, mamba_layers: list[int], attention_layers: list[int]) -> tuple[str, set[int], dict[int, str]]:
    all_mamba = set(mamba_layers)
    attention_adjacent_mamba = adjacent_mamba_layers(mamba_layers, attention_layers)
    non_adjacent_mamba = all_mamba - attention_adjacent_mamba
    if name == "baseline":
        return "normal", set(), {}
    if name == "kv_int4":
        return "int4", set(), {}
    if name == "ssm_mx8":
        return "normal", all_mamba, {idx: "mx8" for idx in all_mamba}
    if name == "ssm_mx8_skip_attn_adjacent":
        return "normal", non_adjacent_mamba, {idx: "mx8" for idx in non_adjacent_mamba}
    if name == "ssm_mx4":
        return "normal", all_mamba, {idx: "mx4" for idx in all_mamba}
    if name == "both_int4_mx8":
        return "int4", all_mamba, {idx: "mx8" for idx in all_mamba}
    if name == "both_int4_mx4":
        return "int4", all_mamba, {idx: "mx4" for idx in all_mamba}
    raise ValueError(f"Unknown experiment {name!r}")


def effective_state_group_size(state_quant: str, configured_group_size: int) -> int | None:
    if state_quant == "none":
        return None
    if state_quant in {"mx8", "mx4"}:
        return 16
    return configured_group_size


def figure_path_for_data(path: Path) -> Path:
    parts = list(path.parts)
    if "data" in parts:
        parts[parts.index("data")] = "figures"
        return Path(*parts).with_suffix(".png")
    return path.with_suffix(".png")


def run_decode_ppl(
    model,
    ids: torch.Tensor,
    context_length: int,
    decode_steps: int,
    checkpoint_steps: list[int],
    quant_layers: set[int],
    mode_by_layer: dict[int, str],
    group_size: int,
    stochastic: bool,
    seed: int,
    progress_every: int,
) -> tuple[list[dict], dict]:
    cache = make_hybrid_cache(model, batch_size=ids.shape[0])
    started = time.perf_counter()
    checkpoint_set = set(checkpoint_steps)
    checkpoint_rows: list[dict] = []
    total_nll = 0.0
    total_tokens = 0

    with torch.inference_mode():
        prefill_pos = torch.arange(context_length, device=ids.device, dtype=torch.long)
        model(
            input_ids=ids[:, :context_length],
            cache_params=cache,
            cache_position=prefill_pos,
            use_cache=True,
            return_dict=True,
        )
        if quant_layers:
            register_mamba_state_kernel_caches(model, cache, mode_by_layer)

        for step in range(decode_steps):
            pos = context_length + step
            cache_pos = torch.tensor([pos], device=ids.device, dtype=torch.long)
            outputs = model(
                input_ids=ids[:, pos : pos + 1],
                cache_params=cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
            )
            loss = F.cross_entropy(outputs.logits[:, -1, :], ids[:, pos + 1], reduction="sum")
            total_nll += float(loss.item())
            total_tokens += ids.shape[0]

            completed = step + 1
            if completed in checkpoint_set:
                elapsed = time.perf_counter() - started
                checkpoint_rows.append(
                    {
                        "decode_steps": completed,
                        "tokens": total_tokens,
                        "nll": total_nll,
                        "ppl": math.exp(total_nll / total_tokens),
                        "elapsed_s": elapsed,
                        "tokens_per_s": total_tokens / elapsed if elapsed else None,
                    }
                )
            if progress_every and completed % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "decode_steps": completed,
                            "ppl": math.exp(total_nll / total_tokens),
                            "elapsed_s": elapsed,
                        }
                    ),
                    flush=True,
                )

    torch.cuda.synchronize(ids.device)
    total_elapsed = time.perf_counter() - started
    return checkpoint_rows, {
        "decode_steps": decode_steps,
        "tokens": total_tokens,
        "nll": total_nll,
        "ppl": math.exp(total_nll / total_tokens),
        "elapsed_s": total_elapsed,
        "tokens_per_s": total_tokens / total_elapsed if total_elapsed else None,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    columns = [
        "experiment",
        "kv_quantization",
        "state_quantization",
        "state_backend",
        "context_length",
        "decode_steps",
        "ppl",
        "baseline_ppl",
        "ppl_degradation_pct",
        "elapsed_s",
        "tokens_per_s",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join("" if row.get(col) is None else str(row.get(col)) for col in columns) + "\n")


def plot_degradation(rows: list[dict], checkpoint_steps: list[int], out_path: Path) -> None:
    styles = {
        "kv_int4": ("KV Cache INT4", "tab:orange", "o"),
        "ssm_mx8": ("SSM State MX8", "tab:green", "s"),
        "ssm_mx8_skip_attn_adjacent": ("SSM MX8 Skip Attn-Adjacent", "tab:cyan", "P"),
        "ssm_mx4": ("SSM State MX4", "tab:red", "^"),
        "both_int4_mx8": ("Both INT4+MX8", "dodgerblue", "D"),
        "both_int4_mx4": ("Both INT4+MX4", "tab:purple", "v"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for ax in axes:
        for exp, (label, color, marker) in styles.items():
            exp_rows = [row for row in rows if row["experiment"] == exp]
            if not exp_rows:
                continue
            exp_rows.sort(key=lambda row: row["decode_steps"])
            ax.plot(
                [row["decode_steps"] for row in exp_rows],
                [row["ppl_degradation_pct"] for row in exp_rows],
                label=label,
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=4,
            )
        ax.axhline(5.0, color="0.7", linestyle=":", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(checkpoint_steps)
        ax.set_xticklabels([str(x) if x < 1024 else f"{x // 1024}K" for x in checkpoint_steps])
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Decode Steps")
        ax.set_ylabel("PPL Degradation (%)")
    axes[0].set_title("(a) Cumulative Quantization Error")
    axes[1].set_title("(b) Detail: Near-Lossless Configs")
    axes[1].set_ylim(-1, 12)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Cumulative Quantization Error in Hybrid Model Decode\n(Nemotron-H-8B)")
    fig.tight_layout()
    fig.savefig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=MODEL_PATH)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--decode-steps", type=int, default=2048)
    parser.add_argument("--checkpoint-steps", default="128,256,512,1024,2048")
    parser.add_argument("--experiments", default="baseline,kv_int4,ssm_mx8,both_int4_mx8")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--output", default="results/ppl/data/nemotron_8b_decode_degradation_ctx1024.jsonl")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint_steps = parse_steps(args.checkpoint_steps)
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    texts = load_wikitext_texts(args.dataset, split="test")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    png_path = figure_path_for_data(out_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    baseline_by_step: dict[int, float] = {}
    header_written = False

    for exp_name in experiments:
        model = AutoModelForCausalLM.from_pretrained(
            args.repo,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
        patch_attention_cache_in_blocks(model)
        tokenizer = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True)
        ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[
            :, : args.context_length + args.decode_steps + 1
        ].cuda()

        mamba_layers, attention_layers, mlp_layers = layer_groups(model.config)
        adjacent = adjacent_mamba_layers(mamba_layers, attention_layers)
        kv_mode, quant_layers, mode_by_layer = experiment_plan(exp_name, mamba_layers, attention_layers)
        if kv_mode == "int4":
            patch_nemotron_h_attention_int4_kv(model, group_size=args.kv_group_size)
        if quant_layers:
            patch_nemotron_h_mamba_decode_state_kernel(
                model,
                group_size=16,
                stochastic=not args.deterministic,
                seed=args.seed,
            )

        if not header_written:
            header = {
                "event": "layer_plan",
                "context_length": args.context_length,
                "decode_steps": args.decode_steps,
                "checkpoint_steps": checkpoint_steps,
                "mamba_layers": mamba_layers,
                "attention_layers": attention_layers,
                "mlp_layers": mlp_layers,
                "attention_adjacent_mamba_layers": sorted(adjacent),
                "non_attention_adjacent_mamba_layers": sorted(set(mamba_layers) - adjacent),
            }
            print(json.dumps(header), flush=True)
            with out_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")
            header_written = True

        print(json.dumps({"event": "start_experiment", "experiment": exp_name}), flush=True)
        checkpoint_rows, final_row = run_decode_ppl(
            model=model,
            ids=ids,
            context_length=args.context_length,
            decode_steps=args.decode_steps,
            checkpoint_steps=checkpoint_steps,
            quant_layers=quant_layers,
            mode_by_layer=mode_by_layer,
            group_size=args.group_size,
            stochastic=not args.deterministic,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        state_quant = "none"
        if quant_layers:
            modes = sorted(set(mode_by_layer.values()))
            state_quant = "+".join(modes)
        rows_to_write = []
        for row in checkpoint_rows:
            baseline_ppl = row["ppl"] if exp_name == "baseline" else baseline_by_step.get(row["decode_steps"])
            if exp_name == "baseline":
                baseline_by_step[row["decode_steps"]] = row["ppl"]
            result = {
                "event": "checkpoint",
                "experiment": exp_name,
                "context_length": args.context_length,
                "kv_quantization": kv_mode,
                "kv_group_size": args.kv_group_size if kv_mode == "int4" else None,
                "state_quantization": state_quant,
                "state_group_size": effective_state_group_size(state_quant, args.group_size),
                "stochastic_rounding": bool(quant_layers) and not args.deterministic,
                "state_backend": f"{state_quant}_kernel" if quant_layers else None,
                "quantized_mamba_layers": sorted(quant_layers),
                "unquantized_mamba_layers": sorted(set(mamba_layers) - quant_layers),
                "baseline_ppl": baseline_ppl,
                "ppl_degradation_pct": None
                if baseline_ppl is None
                else (row["ppl"] / baseline_ppl - 1.0) * 100.0,
                **row,
            }
            rows_to_write.append(result)
            all_rows.append(result)
            print(json.dumps(result), flush=True)

        summary = {
            "event": "final",
            "experiment": exp_name,
            "context_length": args.context_length,
            "kv_quantization": kv_mode,
            "state_quantization": state_quant,
            "state_backend": f"{state_quant}_kernel" if quant_layers else None,
            "quantized_mamba_layers": sorted(quant_layers),
            **final_row,
        }
        print(json.dumps(summary), flush=True)
        with out_path.open("a", encoding="utf-8") as f:
            for row in rows_to_write:
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps(summary) + "\n")

        del model
        torch.cuda.empty_cache()

    write_csv([row for row in all_rows if row["experiment"] != "baseline"], csv_path)
    plot_degradation([row for row in all_rows if row["experiment"] != "baseline"], checkpoint_steps, png_path)
    print(json.dumps({"event": "outputs", "jsonl": str(out_path), "csv": str(csv_path), "plot": str(png_path)}), flush=True)


if __name__ == "__main__":
    main()
