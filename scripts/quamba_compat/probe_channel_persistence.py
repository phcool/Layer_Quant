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


def rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def spearman_to_global(vectors: list[torch.Tensor], global_strength: torch.Tensor) -> list[float]:
    global_rank = rankdata(global_strength.cpu())
    out = []
    for vector in vectors:
        out.append(pearson(rankdata(vector.cpu()), global_rank))
    return out


def topk_overlap_to_global(vectors: list[torch.Tensor], global_strength: torch.Tensor, k: int) -> list[float]:
    k = min(k, global_strength.numel())
    global_top = set(torch.topk(global_strength.cpu(), k=k).indices.tolist())
    out = []
    for vector in vectors:
        top = set(torch.topk(vector.cpu(), k=k).indices.tolist())
        out.append(len(global_top & top) / k)
    return out


def mean(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return sum(finite) / max(1, len(finite))


def percentile(values: list[float], q: float) -> float:
    finite = sorted(v for v in values if v == v)
    if not finite:
        return float("nan")
    idx = min(len(finite) - 1, max(0, round((len(finite) - 1) * q)))
    return finite[idx]


class PersistenceCollector:
    def __init__(self, mamba_layers: list[int], *, token_stride: int, max_token_vectors: int) -> None:
        self.token_stride = token_stride
        self.max_token_vectors = max_token_vectors
        self.sum_abs: dict[int, torch.Tensor] = {}
        self.count: dict[int, int] = {layer_idx: 0 for layer_idx in mamba_layers}
        self.sample_vectors: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in mamba_layers}
        self.token_vectors: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in mamba_layers}

    @torch.no_grad()
    def update(self, layer_idx: int, tensor: torch.Tensor) -> None:
        x = tensor.detach().float().abs()
        if x.dim() != 3:
            raise RuntimeError(f"Expected x_conv_out tensor [B,L,D], got {tuple(x.shape)}")
        batch, seqlen, dim = x.shape
        flat = x.reshape(batch * seqlen, dim)
        channel_sum = flat.sum(dim=0).cpu()
        if layer_idx not in self.sum_abs:
            self.sum_abs[layer_idx] = channel_sum
        else:
            self.sum_abs[layer_idx] += channel_sum
        self.count[layer_idx] += flat.shape[0]

        sample_mean = flat.mean(dim=0).cpu()
        self.sample_vectors[layer_idx].append(sample_mean)

        if len(self.token_vectors[layer_idx]) < self.max_token_vectors:
            sampled = x[:, :: self.token_stride, :].reshape(-1, dim)
            remaining = self.max_token_vectors - len(self.token_vectors[layer_idx])
            for row in sampled[:remaining]:
                self.token_vectors[layer_idx].append(row.cpu())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Quamba2 channel persistence on Nemotron-H Mamba layers.")
    parser.add_argument("--model", default="/scratch2/wl730/models/nemotron-h-8b")
    parser.add_argument("--output-root", default="results/quamba_compat/round18_channel_persistence")
    parser.add_argument("--calib-source", default="quamba_pile", choices=["local_prompts", "quamba_pile", "wikitext"])
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--token-stride", type=int, default=16)
    parser.add_argument("--max-token-vectors", type=int, default=256)
    parser.add_argument("--topk", type=int, default=512)
    args = parser.parse_args()

    ensure_quamba_on_path()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    mamba_layers, attention_layers, mlp_layers = layer_groups(config)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda().eval()
    patch_attention_cache_in_blocks(model)
    configure_nemotron_mamba_layers(model, mamba_layers, use_had_transform=False)

    calibration_dataset = load_calibration_dataset(args.calib_source)
    collector = PersistenceCollector(
        mamba_layers,
        token_stride=args.token_stride,
        max_token_vectors=args.max_token_vectors,
    )

    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, inputs, _outputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            collector.update(layer_idx, x)

        return hook

    for layer_idx in mamba_layers:
        mixer = model.backbone.layers[layer_idx].mixer
        handles.append(mixer.x_conv_out.register_forward_hook(make_hook(layer_idx)))

    device = next(model.parameters()).device
    for sample_idx in tqdm(range(args.num_samples), desc="channel persistence"):
        input_ids = calibration_ids(tokenizer, sample_idx, args.seq_len, device, calibration_dataset)
        model(input_ids=input_ids, use_cache=False, return_dict=True)

    for handle in handles:
        handle.remove()

    rows = []
    for layer_idx in mamba_layers:
        global_strength = collector.sum_abs[layer_idx] / max(1, collector.count[layer_idx])
        sample_spearman = spearman_to_global(collector.sample_vectors[layer_idx], global_strength)
        token_spearman = spearman_to_global(collector.token_vectors[layer_idx], global_strength)
        sample_overlap = topk_overlap_to_global(collector.sample_vectors[layer_idx], global_strength, args.topk)
        token_overlap = topk_overlap_to_global(collector.token_vectors[layer_idx], global_strength, args.topk)
        rows.append(
            {
                "layer": layer_idx,
                "channels": int(global_strength.numel()),
                "samples": len(collector.sample_vectors[layer_idx]),
                "token_vectors": len(collector.token_vectors[layer_idx]),
                "sample_spearman_mean": mean(sample_spearman),
                "sample_spearman_p10": percentile(sample_spearman, 0.10),
                "sample_spearman_p50": percentile(sample_spearman, 0.50),
                "token_spearman_mean": mean(token_spearman),
                "token_spearman_p10": percentile(token_spearman, 0.10),
                "token_spearman_p50": percentile(token_spearman, 0.50),
                "sample_topk_overlap_mean": mean(sample_overlap),
                "sample_topk_overlap_p10": percentile(sample_overlap, 0.10),
                "token_topk_overlap_mean": mean(token_overlap),
                "token_topk_overlap_p10": percentile(token_overlap, 0.10),
            }
        )

    write_csv(output_root / "channel_persistence.csv", rows)
    summary = {
        "model": args.model,
        "calib_source": args.calib_source,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "token_stride": args.token_stride,
        "max_token_vectors": args.max_token_vectors,
        "topk": args.topk,
        "mamba_layers": mamba_layers,
        "attention_layers": attention_layers,
        "mlp_layers": mlp_layers,
        "mean_sample_spearman": mean([row["sample_spearman_mean"] for row in rows]),
        "mean_token_spearman": mean([row["token_spearman_mean"] for row in rows]),
        "mean_sample_topk_overlap": mean([row["sample_topk_overlap_mean"] for row in rows]),
        "mean_token_topk_overlap": mean([row["token_topk_overlap_mean"] for row in rows]),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "output_root": str(output_root), **summary}), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
