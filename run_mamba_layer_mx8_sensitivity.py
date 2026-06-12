from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hybrid_quant.data import load_wikitext_texts
from hybrid_quant.nemotron_8b_decode_eval import layer_groups, patch_attention_cache_in_blocks
from hybrid_quant.mamba_state_8b_kernel import (
    patch_nemotron_h_mamba_decode_state_kernel,
)
from run_nemotron_8b_decode_degradation import MODEL_PATH, parse_steps, run_decode_ppl


def parse_layers(value: str, mamba_layers: list[int]) -> list[int]:
    if value == "all":
        return mamba_layers
    selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(mamba_layers))
    if invalid:
        raise ValueError(f"Non-Mamba layers requested: {invalid}; Mamba layers are {mamba_layers}")
    return selected


def make_ids(repo: str, dataset: str, context_length: int, decode_steps: int, device: torch.device) -> torch.Tensor:
    texts = load_wikitext_texts(dataset, split="test")
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[
        :, : context_length + decode_steps + 1
    ]
    return ids.to(device)


def evaluate(
    *,
    repo: str,
    ids: torch.Tensor,
    context_length: int,
    decode_steps: int,
    layer_idx: int | None,
    seed: int,
    deterministic: bool,
    progress_every: int,
) -> dict:
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    patch_attention_cache_in_blocks(model)

    quant_layers: set[int] = set()
    mode_by_layer: dict[int, str] = {}
    if layer_idx is not None:
        quant_layers = {layer_idx}
        mode_by_layer = {layer_idx: "mx8"}
        patch_nemotron_h_mamba_decode_state_kernel(
            model,
            group_size=16,
            stochastic=not deterministic,
            seed=seed,
        )

    checkpoint_rows, final_row = run_decode_ppl(
        model=model,
        ids=ids,
        context_length=context_length,
        decode_steps=decode_steps,
        checkpoint_steps=[decode_steps],
        quant_layers=quant_layers,
        mode_by_layer=mode_by_layer,
        group_size=16,
        stochastic=not deterministic,
        seed=seed,
        progress_every=progress_every,
    )
    del model
    torch.cuda.empty_cache()
    return checkpoint_rows[-1] if checkpoint_rows else final_row


def write_rows(path: Path, rows: list[dict]) -> None:
    columns = [
        "layer_idx",
        "ppl",
        "baseline_ppl",
        "delta_ppl",
        "ppl_degradation_pct",
        "elapsed_s",
        "tokens_per_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[int] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                layer_idx = int(row["layer_idx"])
                if layer_idx in seen:
                    continue
                seen.add(layer_idx)
                rows.append(
                    {
                        "layer_idx": layer_idx,
                        "ppl": float(row["ppl"]),
                        "baseline_ppl": float(row["baseline_ppl"]),
                        "delta_ppl": float(row["delta_ppl"]),
                        "ppl_degradation_pct": float(row["ppl_degradation_pct"]),
                        "elapsed_s": float(row["elapsed_s"]),
                        "tokens_per_s": float(row["tokens_per_s"]),
                    }
                )
    rows.sort(key=lambda row: row["layer_idx"])
    return rows


def plot_rows(rows: list[dict], attention_layers: list[int], out_path: Path) -> None:
    xs = [row["layer_idx"] for row in rows]
    ys = [row["delta_ppl"] for row in rows]
    colors = ["#1f77b4" if y >= 0 else "#2ca02c" for y in ys]

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=170)
    ax.bar(xs, ys, width=0.72, color=colors, edgecolor="0.2", linewidth=0.4)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    for i, layer_idx in enumerate(attention_layers):
        ax.axvline(
            layer_idx,
            color="0.4",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label="Attention layer" if i == 0 else None,
        )
    ax.set_xticks(xs)
    ax.set_xlabel("Mamba Layer Index")
    ax.set_ylabel("Delta PPL vs Baseline")
    ax.set_title("Per-Mamba-Layer MX8 Sensitivity at Decode 256\n(Nemotron-H-8B, context=1024, WikiText)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=MODEL_PATH)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--baseline-ppl", type=float)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--output", default="results/nemotron_8b_mamba_layer_mx8_sensitivity_ctx1024_decode256.csv")
    parser.add_argument("--plot-inputs", default="")
    parser.add_argument("--plot-output", default="")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.repo, trust_remote_code=True)
    mamba_layers, attention_layers, mlp_layers = layer_groups(config)

    if args.plot_inputs:
        input_paths = [Path(item.strip()) for item in args.plot_inputs.split(",") if item.strip()]
        rows = read_rows(input_paths)
        out_path = Path(args.plot_output) if args.plot_output else Path(args.output).with_suffix(".png")
        write_rows(Path(args.output), rows)
        plot_rows(rows, attention_layers, out_path)
        print(json.dumps({"event": "plot", "csv": args.output, "plot": str(out_path)}), flush=True)
        return

    device = torch.device("cuda")
    ids = make_ids(args.repo, args.dataset, args.context_length, args.decode_steps, device)
    selected_layers = parse_layers(args.layers, mamba_layers)

    baseline_ppl = args.baseline_ppl
    if baseline_ppl is None:
        baseline = evaluate(
            repo=args.repo,
            ids=ids,
            context_length=args.context_length,
            decode_steps=args.decode_steps,
            layer_idx=None,
            seed=args.seed,
            deterministic=args.deterministic,
            progress_every=args.progress_every,
        )
        baseline_ppl = float(baseline["ppl"])
        print(json.dumps({"event": "baseline", "ppl": baseline_ppl}), flush=True)

    print(
        json.dumps(
            {
                "event": "layer_plan",
                "mamba_layers": mamba_layers,
                "attention_layers": attention_layers,
                "mlp_layers": mlp_layers,
                "selected_layers": selected_layers,
                "baseline_ppl": baseline_ppl,
            }
        ),
        flush=True,
    )

    rows: list[dict] = []
    for layer_idx in selected_layers:
        print(json.dumps({"event": "start_layer", "layer_idx": layer_idx}), flush=True)
        result = evaluate(
            repo=args.repo,
            ids=ids,
            context_length=args.context_length,
            decode_steps=args.decode_steps,
            layer_idx=layer_idx,
            seed=args.seed,
            deterministic=args.deterministic,
            progress_every=args.progress_every,
        )
        ppl = float(result["ppl"])
        row = {
            "layer_idx": layer_idx,
            "ppl": ppl,
            "baseline_ppl": baseline_ppl,
            "delta_ppl": ppl - baseline_ppl,
            "ppl_degradation_pct": (ppl / baseline_ppl - 1.0) * 100.0,
            "elapsed_s": result["elapsed_s"],
            "tokens_per_s": result["tokens_per_s"],
        }
        rows.append(row)
        write_rows(Path(args.output), rows)
        print(json.dumps({"event": "layer_result", **row}), flush=True)

    out_path = Path(args.output)
    plot_rows(rows, attention_layers, out_path.with_suffix(".png"))
    print(json.dumps({"event": "outputs", "csv": str(out_path), "plot": str(out_path.with_suffix(".png"))}), flush=True)


if __name__ == "__main__":
    main()
