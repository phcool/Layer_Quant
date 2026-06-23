# Nemotron-H-8B Decode Quantization

This workspace keeps the current Nemotron-H-8B decode-path experiments only.

Supported quantization paths:

- KV cache: BF16 baseline or packed INT4 cache with fused Triton decode attention.
- Mamba recurrent state: BF16 baseline, MX8 kernel simulation, or MX4 kernel simulation.

The INT4 KV path stores K/V cache entries as two signed 4-bit values per byte
with one FP32 scale per group, then decodes packed K/V inside the attention
kernel for `q_len=1` generation steps.

The MX8 state path uses a 16-value block with one shared 8-bit exponent, a
1-bit micro-exponent per pair, signed int8 mantissas in `[-63, 63]`, and
stochastic rounding on writeback.

The MX4 state path follows the same block and micro-exponent structure, but
clamps mantissas to signed 4-bit levels in `[-7, 7]`.

## Run

```bash
cd /scratch2/wl730/hybrid_codesign/han/chunk_update
CUDA_VISIBLE_DEVICES=0 /scratch2/wl730/conda_envs/hybrid/bin/python \
  run_nemotron_8b_decode_degradation.py \
  --context-length 1024 \
  --decode-steps 2048 \
  --checkpoint-steps 128,256,512,1024,2048 \
  --experiments baseline,kv_int4,ssm_mx8,both_int4_mx8 \
  --output results/nemotron_8b_decode_mx8_kernel_ctx1024.jsonl
```

For MX4 state experiments, use:

```bash
CUDA_VISIBLE_DEVICES=0 /scratch2/wl730/conda_envs/hybrid/bin/python \
  run_nemotron_8b_decode_degradation.py \
  --experiments baseline,ssm_mx4,both_int4_mx4 \
  --output results/nemotron_8b_decode_mx4_kernel_ctx1024.jsonl
```

The script also writes a CSV and PNG next to the JSONL output.

## Plot Existing Results

```bash
/scratch2/wl730/conda_envs/hybrid/bin/python scripts/plots/plot_decode_degradation_like_reference.py
```

This produces `results/nemotron_8b_decode_mx8_like_reference.png`.
