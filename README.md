# Nemotron-H-8B Decode Quantization

This workspace keeps the current Nemotron-H-8B decode-path experiments only.

Supported quantization paths:

- KV cache: BF16 baseline or INT4 quant-dequant before attention cache update.
- Mamba recurrent state: BF16 baseline or MX8 kernel simulation.

The MX8 state path uses a 16-value block with one shared 8-bit exponent, a
1-bit micro-exponent per pair, signed int8 mantissas in `[-63, 63]`, and
stochastic rounding on writeback.

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

The script also writes a CSV and PNG next to the JSONL output.

## Plot Existing Results

```bash
/scratch2/wl730/conda_envs/hybrid/bin/python plot_decode_degradation_like_reference.py
```

This produces `results/nemotron_8b_decode_mx8_like_reference.png`.
