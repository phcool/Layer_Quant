# Correction Cache

Runs attention every `--interval K` decode steps.
On skipped steps, the block reuses the last attention output delta as a cached correction.

Use `--correction-decay <float>` to decay older corrections, for example `0.9`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/correction_cache/run.py \
  --interval 4 \
  --correction-decay 0.95 \
  --batch-size 8 \
  --sequence-length 8192 \
  --decode-steps 64 \
  --warmup-steps 8 \
  --quality \
  --output-dir results/low_freq_attention/correction_cache/ctx8192_interval4_decay095
```
