# Calibrated Gating

First runs a full-attention calibration pass and records a hidden-state delta proxy
for each attention layer. It then builds a fixed replay schedule and measures that
schedule without per-step CPU synchronization.

This is an upper-bound proxy for Mamba-conditioned attention gating.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/calibrated_gating/run.py \
  --keep-fraction 0.5 \
  --max-skip 4 \
  --batch-size 8 \
  --sequence-length 8192 \
  --decode-steps 64 \
  --warmup-steps 8 \
  --quality \
  --output-dir results/low_freq_attention/calibrated_gating/ctx8192_keep50_maxskip4
```
