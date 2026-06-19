# Low-Frequency Attention Experiments

This suite tests hybrid-specific attention co-design ideas:

- `periodic_refresh`: attention is a low-frequency global correction.
- `correction_cache`: skipped attention steps reuse the previous attention delta.
- `layerwise_schedule`: different attention layers refresh at different rates.
- `calibrated_gating`: a calibration pass builds a fixed replay schedule from hidden-state delta statistics.

Each experiment writes its own folder with:

- `latency.csv`
- `layer_summary.csv`
- `quality.csv` when `--quality` is enabled
- `summary.json`

Run a full sweep:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/run_sweep.py \
  --batch-size 8 \
  --sequence-length 8192 \
  --decode-steps 64 \
  --warmup-steps 8 \
  --quality \
  --output-root results/low_freq_attention
```

Quality pass is intentionally separate from latency pass so CPU logits transfers do not pollute latency.
