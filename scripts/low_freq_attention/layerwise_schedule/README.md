# Layerwise Schedule

Uses different refresh intervals for different attention layers.

`--layer-intervals` format is `layer:interval,layer:interval`.
Layers not listed use `--interval`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/low_freq_attention/layerwise_schedule/run.py \
  --interval 4 \
  --layer-intervals 7:1,18:2,29:4,40:4 \
  --batch-size 8 \
  --sequence-length 8192 \
  --decode-steps 64 \
  --warmup-steps 8 \
  --quality \
  --output-dir results/low_freq_attention/layerwise_schedule/ctx8192_7x1_18x2_29x4_40x4
```
