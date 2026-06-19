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

