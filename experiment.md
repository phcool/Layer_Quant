# Hybrid Low-Frequency Attention Experiments

## 2026-06-19 18:07 CST - Round 1: Latency Screening

### Motivation

The KV-load A/B experiment showed that accelerating KV load is not a large enough target: projected BF16 KV load is only about 3.85% of decode GPU kernel time at batch 8, context 8192. The next hypothesis is that hybrid models may benefit more from treating attention as a low-frequency global correction, while Mamba layers continue to run every decode step.

### Goal

Run a latency-only screening sweep to identify whether reducing attention execution frequency gives measurable decode speedup before spending time on quality/KL experiments.

### Fixed Setup

- Model: Nemotron-H-8B from `/scratch2/wl730/models/nemotron-h-8b`
- Server repo: `/scratch2/wl730/hybrid_codesign/han/chunk_update`
- Environment: project-local `.venv`, no server global environment changes
- Batch size: 8
- Context length: 8192
- Decode steps: 64
- Warmup steps: 8
- Dataset source: WikiText test text, same helper as existing latency scripts
- Output root: `results/low_freq_attention_round1_latency`

### Experiments

1. `periodic_refresh`
   - Intervals: 2, 4, 8
   - Semantics: all attention layers run every K decode steps; skipped attention blocks return residual only.

2. `correction_cache`
   - Intervals: 2, 4, 8
   - Decay: 1.0
   - Semantics: all attention layers run every K decode steps; skipped attention blocks reuse the last attention delta.

3. `correction_cache_decay`
   - Interval: 4
   - Decay: 0.95
   - Semantics: same as correction cache, but cached correction decays with age.

4. `layerwise_schedule`
   - Schedule A: `7:1,18:2,29:4,40:4`
   - Schedule B: `7:4,18:4,29:2,40:1`
   - Purpose: test whether early or late attention layers are more important to refresh frequently.

5. `calibrated_gating`
   - Keep fractions: 0.25, 0.5, 0.75
   - Max skip: 4
   - Purpose: estimate an upper-bound fixed replay schedule from hidden-state delta calibration.

### Metrics

- `decode_ms_per_step`
- `step_p50_ms`
- `step_p90_ms`
- `attention_runs`
- `attention_skips`
- `attention_run_fraction`
- Per-layer run/skip counts from `layer_summary.csv`

### Analysis Plan

- Compare each candidate against the existing nsys baseline at context 8192:
  - baseline `none` total GPU kernel time per step: about 27.835 ms
  - previous end-to-end event baseline may differ, so the round primarily compares candidates internally.
- Identify candidates with clear latency reduction proportional to attention run fraction.
- Select a small subset for Round 2 quality evaluation:
  - fastest periodic strategy
  - fastest correction-cache strategy
  - best layerwise strategy
  - one calibrated-gating strategy

### Command

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/run_sweep.py \
  --batch-size 8 \
  --sequence-length 8192 \
  --decode-steps 64 \
  --warmup-steps 8 \
  --output-root results/low_freq_attention_round1_latency
```

### Implementation Correction Before Running

The first attempted run was stopped after the first result because the layer run/skip counters included warmup steps. Before the official Round 1 run, the implementation is corrected as follows:

- Prefill or any `q_len > 1` attention call always runs full attention and does not enter the low-frequency skip/reuse path.
- After warmup, the low-frequency controller is reset so that `attention_runs + attention_skips` covers only measured decode steps.
- This makes the expected denominator equal to `num_attention_layers * decode_steps`, i.e. `4 * 64 = 256` for Nemotron-H-8B.

## 2026-06-19 18:20 CST - Round 1B: Same-Runner Full-Attention Baseline

### Motivation

Round 1 latency uses CUDA event timing around the full decode step inside the low-frequency attention runner. This is not directly comparable to the previous nsys GPU-kernel sum baseline. To interpret speedups correctly, a same-runner full-attention baseline is needed.

### Goal

Run `periodic_refresh` with `--interval 1`, which should execute every attention layer at every measured decode step. This gives the baseline for Round 1 candidates under identical runner, cache, event timing, batch, context, warmup, and decode-step settings.

### Fixed Setup

- Batch size: 8
- Context length: 8192
- Decode steps: 64
- Warmup steps: 8
- Output directory: `results/low_freq_attention_round1_latency/baseline/full_attention_interval1`

### Expected Sanity Check

- `attention_runs = 4 * 64 = 256`
- `attention_skips = 0`
- `attention_run_fraction = 1.0`

### Command

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/periodic_refresh/run.py \
  --batch-size 8 \
  --sequence-length 8192 \
  --decode-steps 64 \
  --warmup-steps 8 \
  --interval 1 \
  --output-dir results/low_freq_attention_round1_latency/baseline/full_attention_interval1
```

## 2026-06-19 18:31 CST - Round 2: Quality Screening for Promising Low-Frequency Schedules

### Motivation

Round 1 showed a clear latency/sparsity trend under the same-runner event timing baseline:

- Full attention baseline: about 38.99 ms/step
- Periodic interval 8: about 26.37 ms/step, 1.48x speedup, attention run fraction 0.125
- Periodic interval 4: about 28.17 ms/step, 1.38x speedup, attention run fraction 0.25
- Periodic interval 2: about 31.87 ms/step, 1.22x speedup, attention run fraction 0.5
- Correction-cache variants are slightly slower than residual-only periodic at the same interval, but may preserve quality better.
- Layerwise and calibrated gating did not look latency-efficient enough for the first quality pass.

The next question is whether any low-frequency schedule keeps acceptable next-token behavior.

### Goal

Run quality screening for the most relevant schedules and compare against a full-attention baseline using:

- Cross entropy and perplexity against the actual next token
- KL divergence from full-attention logits
- Top-1 match with full-attention logits

### Fixed Setup

- Context length: 8192
- Decode steps: 64
- Warmup steps: 8
- Batch size: 2 for quality screening, to avoid excessive CPU logits transfer
- Output root: `results/low_freq_attention_round2_quality`
- Quality mode enabled with `--quality`

### Experiments

1. `periodic_refresh`, intervals 2, 4, 8
2. `correction_cache`, intervals 2, 4, 8, decay 1.0
3. `correction_cache`, interval 4, decay 0.95

### Analysis Plan

- Treat interval 2 as the conservative candidate.
- Treat interval 4 as the likely latency/quality tradeoff candidate.
- Treat interval 8 as an upper-bound speed candidate that may fail quality.
- If correction-cache improves KL/top1 significantly relative to residual-only periodic at the same interval, keep it for Round 3.
- If all quality metrics collapse, the low-frequency attention idea is only useful with a more careful KV-update/correction design.

### Commands

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/periodic_refresh/run.py \
  --batch-size 2 --sequence-length 8192 --decode-steps 64 --warmup-steps 8 \
  --interval 2 --quality \
  --output-dir results/low_freq_attention_round2_quality/periodic_refresh/ctx8192_interval2

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/periodic_refresh/run.py \
  --batch-size 2 --sequence-length 8192 --decode-steps 64 --warmup-steps 8 \
  --interval 4 --quality \
  --output-dir results/low_freq_attention_round2_quality/periodic_refresh/ctx8192_interval4

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/periodic_refresh/run.py \
  --batch-size 2 --sequence-length 8192 --decode-steps 64 --warmup-steps 8 \
  --interval 8 --quality \
  --output-dir results/low_freq_attention_round2_quality/periodic_refresh/ctx8192_interval8

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/correction_cache/run.py \
  --batch-size 2 --sequence-length 8192 --decode-steps 64 --warmup-steps 8 \
  --interval 2 --quality \
  --output-dir results/low_freq_attention_round2_quality/correction_cache/ctx8192_interval2_reuse

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/correction_cache/run.py \
  --batch-size 2 --sequence-length 8192 --decode-steps 64 --warmup-steps 8 \
  --interval 4 --quality \
  --output-dir results/low_freq_attention_round2_quality/correction_cache/ctx8192_interval4_reuse

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/correction_cache/run.py \
  --batch-size 2 --sequence-length 8192 --decode-steps 64 --warmup-steps 8 \
  --interval 8 --quality \
  --output-dir results/low_freq_attention_round2_quality/correction_cache/ctx8192_interval8_reuse

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/correction_cache/run.py \
  --batch-size 2 --sequence-length 8192 --decode-steps 64 --warmup-steps 8 \
  --interval 4 --correction-decay 0.95 --quality \
  --output-dir results/low_freq_attention_round2_quality/correction_cache/ctx8192_interval4_decay095
```
