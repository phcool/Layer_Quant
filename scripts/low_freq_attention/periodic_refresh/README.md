# Periodic Refresh

Runs all attention layers every `--interval K` decode steps and skips attention on the other steps.

This measures the most aggressive version of low-frequency global correction:
Mamba layers still run every step, but skipped attention blocks return only the residual.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/periodic_refresh/run.py \
  --interval 2 \
  --batch-size 8 \
  --sequence-length 8192 \
  --decode-steps 64 \
  --warmup-steps 8 \
  --output-dir results/low_freq_attention/periodic_refresh/ctx8192_interval2
```
