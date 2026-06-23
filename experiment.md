# Hybrid 低频 Attention 实验记录

本文档记录围绕 Nemotron-H-8B hybrid 架构做的低频 attention co-design 实验。实验目标是验证：在 hybrid 模型里，Mamba 层每步运行，而 attention 层是否可以作为低频 global correction，从而减少 decode 时 attention 的执行频率。

## Quamba Compat：Round 14 WikiText 长上下文 calibration + W8A8 Hadamard PPL 设计

### 背景

Round 13 证明，之前 local prompt calibration 对 Mamba state 的覆盖明显不足；换成 WikiText calibration 后，`ssm_state_act` 的 clipping 大幅下降：

- `local_prompts_128_s8`：state mean clip `4.04%`，worst layer `15.20%`
- `wikitext_2048_s8`：state mean clip `0.40%`，worst layer `0.61%`

因此需要重新跑真正的 W8A8 质量实验，检查更合理的 calibration 是否能改善 PPL。

### 实验目标

测试在开启 Quamba Hadamard path 的情况下，使用 WikiText 长上下文 calibration 后，W8A8 Mamba 替换的 PPL/KL/top1 是否恢复。

### 实验配置

脚本：

```text
scripts/quamba_compat/diagnose_quamba_quality.py
```

模式：

| mode | 说明 |
|---|---|
| `baseline_fp16` | FP16 baseline |
| `w8a8_had_fused_all` | 所有 Mamba layer 替换为 W8A8，开启 Quamba Hadamard；Hadamard fuse 只作用于 Mamba layer 的 `HadLinear`，跳过 hybrid 里的 MLP layer |

参数：

| 参数 | 值 |
|---|---:|
| context length | 2048 |
| batch size | 2 |
| decode steps | 64 |
| calibration source | WikiText |
| calibration seq len | 2048 |
| calibration samples | 8 |
| grouped CAWR | 开启 |

输出目录：

```text
results/quamba_compat/round14_wikitext2048_s8_w8a8_had_ppl/
```

### 运行修正

第一次运行时，直接调用 Quamba 官方 `fuse_had_matrices(model)` 失败：

```text
AttributeError: 'NemotronHMLP' object has no attribute 'in_proj'
```

原因是 Quamba 官方函数默认纯 Mamba block，每层都有 `mixer.in_proj/out_proj`；Nemotron-H 是 hybrid 架构，部分 block 是 MLP mixer。修正方式不是忽略 Hadamard，而是把等价的 `HadLinear.fuse_hadamard()` 限定到已配置的 Mamba layers，并记录实际 fuse 的 layer id。

### 判断标准

1. 如果 PPL 接近 baseline，说明主要问题是 calibration 覆盖不足。
2. 如果 PPL 仍明显偏高，则即使用更好的 state scale，Quamba W8A8 + Hadamard 直接迁移到 Nemotron-H hybrid 仍有结构性误差，下一步需要继续拆分：
   - Hadamard 融合是否破坏 hybrid residual basis；
   - W8A8 state update / qchunk_scan 与 Nemotron-H 的 Mamba state 分布是否仍不匹配；
   - out_proj 的 Hadamard 和 state scale 是否需要 hybrid-specific 设计。

### 运行结果

远端运行命令使用 GPU0，`context_length=2048`、`batch_size=2`、`decode_steps=64`、`num_calib_samples=8`、`calib_seq_len=2048`、`calib_source=wikitext`。

结果文件：

```text
results/quamba_compat/round14_wikitext2048_s8_w8a8_had_ppl/quality_diagnosis.csv
```

| mode | PPL | KL vs baseline | top1 match | decode ms/step | peak mem GiB |
|---|---:|---:|---:|---:|---:|
| `baseline_fp16` | 5.3261 | 0.0000 | 1.0000 | 31.1060 | 18.3819 |
| `w8a8_had_fused_all` | 119971.5401 | 10.1643 | 0.0078 | 23.3910 | 15.9359 |

`w8a8_had_fused_all_prepare.json` 确认：

- 24 个 Mamba layers 全部进入 `configured_layers`；
- 24 个 Mamba layers 全部进入 `had_fused_layers`；
- 24 个 Mamba mixers 全部替换为 `W8A8QMamba2`；
- `in_proj/out_proj/conv` weight dtype 均为 `torch.int8`；
- `x_conv_out_scale` 和 `ssm_state_scale` 都是 grouped scale，`ssd_groups=8`。

结论：这次不是 Hadamard 没打开、也不是 Mamba layer 漏 fuse。即使用 WikiText 长上下文 calibration，Quamba 的 fused Hadamard W8A8 path 直接迁移到 Nemotron-H hybrid 仍然严重失真。短 calibration 覆盖不足确实存在，但不是主要瓶颈；当前更像是 Quamba 的全模型 basis transform / CAWR / activation scale 假设和 Nemotron-H hybrid residual stream 不匹配。

## Quamba Compat：Round 13 长上下文 calibration 对 state scale 的影响设计

### 背景

Round 12 修正了 grouped state scale 的展开逻辑后，发现：

- `out_proj:input` 在 no-Had 下几乎不 clipping，但 `q_abs_p99` 很低，说明常见值只占用少量 int8 levels；
- local Hadamard 可以显著提高 `out_proj:input` 的 scale utilization；
- 更关键的是，`ssm_state_act:input` 在 2048 context prefill 后仍然有明显 clipping：
  - 平均 `clip_frac` 约 `4%`；
  - 最坏层约 `15%`；
  - 多个 layer 的 `q_abs_p99` 超过 127。

这说明当前质量退化不只是 `out_proj`，Mamba state 的 offline scale 也可能不适配实际长上下文状态分布。

### 实验目标

判断 `ssm_state_act` 的 scale mismatch 是否主要来自 calibration 序列太短、数据分布太窄。

### 实验方式

新增 WikiText 长文本 calibration source，仅用于 coverage probe：

```text
scripts/quamba_compat/probe_quamba_scale_coverage.py --calib-source wikitext
```

对比两组 no-Had coverage：

| 输出目录 | calibration 数据 | calibration seq_len | calibration samples | eval context | eval batch | decode steps |
|---|---|---:|---:|---:|---:|---:|
| `round13_scale_context_sweep/calib128` | WikiText 长文本切片 | 128 | 8 | 2048 | 2 | 32 |
| `round13_scale_context_sweep/calib2048` | WikiText 长文本切片 | 2048 | 2 | 2048 | 2 | 32 |
| `round13_scale_context_sweep/calib2048_s8` | WikiText 长文本切片 | 2048 | 8 | 2048 | 2 | 32 |

只跑 `no_had`，不引入 local Hadamard，避免 out_proj smoothing 干扰 state scale 归因。

补充说明：如果 `calib2048` 用 2 个样本反而更差，需要继续跑 `calib2048_s8` 控制样本数，避免把 sample coverage 误判成 seq_len 效应。

### 判断标准

1. 如果 `calib2048` 显著降低 `ssm_state_act:input` 的 `clip_frac/q_abs_p99`，说明当前 Quamba scale 对 hybrid 模型失败的主要原因之一是 calibration 没覆盖长上下文 state。
2. 如果 `calib2048` 仍然有明显 clipping，则问题不是简单增加 calibration 长度能解决，可能需要：
   - state scale 随 context length 分段；
   - per-layer/per-dstate 更稳健的 percentile 或 head-dim grouping；
   - 对 Mamba state 的更新路径做专门的 hybrid-aware quantization。

### 实验结果

输出目录：

```text
results/quamba_compat/round13_scale_context_sweep/
```

关键 op 的 prefill coverage 汇总：

| config | op | mean clip | max clip | mean q_abs p99 | max q_abs |
|---|---|---:|---:|---:|---:|
| local_prompts_128_s8 | ssm_state_act | 0.040368 | 0.152039 | 402.6 | 463554.1 |
| wikitext_128_s8 | ssm_state_act | 0.005355 | 0.007902 | 103.4 | 9216.0 |
| wikitext_2048_s2 | ssm_state_act | 0.021806 | 0.031374 | 180.0 | 28621.7 |
| wikitext_2048_s8 | ssm_state_act | 0.004046 | 0.006099 | 96.1 | 4342.1 |
| local_prompts_128_s8 | out_proj | 0.000002 | 0.000010 | 4.9 | 1346.8 |
| wikitext_128_s8 | out_proj | 0.000000 | 0.000001 | 4.2 | 579.7 |
| wikitext_2048_s8 | out_proj | 0.000000 | 0.000000 | 2.8 | 214.5 |
| wikitext_2048_s8 | x_conv_out | 0.000002 | 0.000005 | 19.6 | 1326.9 |
| wikitext_2048_s8 | B_conv_out | 0.000001 | 0.000001 | 46.5 | 263.2 |
| wikitext_2048_s8 | C_conv_out | 0.000001 | 0.000002 | 33.0 | 234.8 |

`ssm_state_act` 最坏层：

| config | worst layers by clip |
|---|---|
| local_prompts_128_s8 | L0 15.20%, L2 6.62%, L26 5.87%, L35 5.48%, L48 5.40% |
| wikitext_128_s8 | L48 0.79%, L44 0.75%, L50 0.73%, L39 0.69%, L0 0.68% |
| wikitext_2048_s2 | L33 3.14%, L39 3.05%, L35 2.90%, L48 2.81%, L4 2.62% |
| wikitext_2048_s8 | L33 0.61%, L35 0.56%, L39 0.54%, L4 0.53%, L31 0.50% |

### 结论

1. offline scale 的确存在适配问题，但不是“所有 offline scale 都不适用”。最糟糕的是之前的 `local_prompts` calibration，它的数据分布太窄，导致 `ssm_state_act` 在真实 WikiText 2048 context 上严重 clipping。
   - local prompts 128：state mean clip `4.04%`，worst layer `15.20%`
   - WikiText 128：state mean clip `0.54%`，worst layer `0.79%`
   - WikiText 2048/s8：state mean clip `0.40%`，worst layer `0.61%`

2. `calib2048_s2` 比 `calib128_s8` 更差，说明样本数太少会让 CAWR/reorder 和 state scale 不稳定。控制样本数后，`calib2048_s8` 是目前最好的 state coverage。

3. `out_proj` 的问题和 state 不一样。no-Had 下 `out_proj:input` 基本不 clipping，但 `q_abs_p99` 只有 `2.8-4.9`，说明常见值只用了很少 int8 levels；这解释了为什么 local Hadamard 能显著改善 out_proj：它不是简单减少 clipping，而是改善 scale utilization。

4. FP16 out_proj 仍然只有 PPL `10.66`，没有回到 baseline `6.75`，更合理的解释是 upstream Mamba W8A8 路径已经引入误差，尤其是 `ssm_state_act`。相比之下，`in_proj/x_conv/B/C/dt/z` 的 clipping 都很低，不是主要矛盾。

5. 下一步如果继续走 Quamba-compatible 路线，应先用更合理的 WikiText/Pile calibration 重新跑质量实验；如果 PPL 仍然明显差于 baseline，就说明需要针对 hybrid Mamba state 做新的 scale 策略，而不是只修 out_proj。

## Quamba Compat：Round 12 offline scale 覆盖与 local Hadamard 机制分析设计

### 背景

当前结果显示：

- `w8a8_no_had_all`：PPL `43.11`
- `w8a8_local_out_had_all`：PPL `16.71`
- `w8a8_no_had_fp_outproj_all`：PPL `10.66`
- BF16 baseline：PPL `6.75`

这说明两件事：

1. `out_proj` 是主要误差来源之一，因为恢复 FP16 `out_proj` 后质量大幅恢复。
2. `out_proj` 不是唯一误差来源，因为 FP16 `out_proj` 上界仍然只有 PPL `10.66`，没有回到 baseline `6.75`。

因此现在要判断：offline calibration 得到的 scale 是否不适用于真实 WikiText decode 分布，以及误差是否在多个 Mamba 子模块中都存在。

### 实验目标

本轮不再引入 dynamic per-token out_proj。只分析 Quamba 原始 static scale 和我们当前 local Hadamard 的机制。

要回答的问题：

1. offline calibration 的 scale 在真实 eval context/decode 上是否 clipping？
2. 如果没有明显 clipping，是否 scale utilization 很低，导致大部分值只用了很少 int8 levels？
3. 问题是否只集中在 `out_proj:input`，还是 `in_proj / conv / qchunk_scan / state / norm` 的 scale 也不匹配？
4. local Hadamard 改善 `out_proj` 的原因是否可以从 activation 分布上看到，例如：
   - `out_proj:input` 的 `q_abs_max` 降低；
   - `q_abs_p99` 更接近合理范围；
   - clipping ratio 降低；
   - channel/token outlier 被摊平。

### 实验方式

新脚本：

```text
scripts/quamba_compat/probe_quamba_scale_coverage.py
```

流程：

1. 加载 Nemotron-H-8B；
2. 将所有 Mamba layer 转成 Quamba `Mamba2Simple(use_had_transform=False)`；
3. 用 `local_prompts, 8 samples, seq_len=128` 做 CAWR reorder 和 activation calibration，得到 offline scales；
4. 不做 W8A8 quantization，保持 FP Mamba2Simple 路径；
5. 在真实 WikiText eval 输入上跑：
   - context length：`2048`
   - batch size：`2`
   - decode steps：`32`
6. hook Quamba calibration 中同一批 op，记录每个 layer/op/stage 的：
   - `clip_frac = mean(abs(x / scale) > 127)`
   - `q_abs_mean`
   - `q_abs_p99`
   - `q_abs_p999`
   - `q_abs_max`
   - `util_p99 = q_abs_p99 / 127`
   - `util_max = q_abs_max / 127`
7. 对比两个 mode：
   - `no_had`
   - `local_out_had`

### 判断标准

1. 如果某些 op 的 `clip_frac` 很高，说明 offline scale 没覆盖真实 eval 分布，质量损失来自 hard clipping。
2. 如果 `clip_frac` 很低但 `util_p99` 很低，说明 scale 被少数 outlier 拉大，常见值 int8 分辨率不足。
3. 如果 `local_out_had` 显著改善 `out_proj:input` 的 utilization / clipping，而内部 conv/scan/state 基本不变，说明 local Hadamard 的收益主要是 out_proj 前 channel smoothing。
4. 如果 `in_proj / conv / scan / state` 也有严重 mismatch，则 FP16 `out_proj` 仍不能回到 baseline 的原因可能来自这些 upstream W8A8 scale。

## Quamba Compat：Round 11 out_proj 量化误差归因设计

### 背景

Round 10 证明，在完整 Quamba W8A8 Mamba path 中，只把最后的 `QAct + W8A8 out_proj` 恢复成 FP16，就能把 PPL 从 `43.11` 降到 `10.66`，Top1 match 从 `43.75%` 提升到 `75.00%`。这说明 `out_proj` 是当前质量退化的主要来源之一。

进一步阅读 Quamba 实现后发现：

- `QAct` 是 per-tensor static scale；
- `QAct.forward` 使用 `(x / scale).clamp(...).to(torch.int8)`，即直接截断，不做 round；
- `W8A8B16O16Linear` 的 activation scale 和 weight scale 都是 per-tensor；
- 对 Nemotron-H 这种 hybrid 模型，Mamba block 的 `out_proj` 输出会回到 residual stream，并被后续 attention/MLP 使用，因此这一处的量化误差可能比纯 Mamba 模型更敏感。

### 核心问题

Round 11 要拆清楚 `out_proj` 的误差来自哪里：

1. 是 `QAct` 的截断而不是 round 导致？
2. 是 static per-tensor activation scale 被 outlier 拉大导致？
3. 是 W8 per-tensor weight quantization 本身导致？

### 实验方式

保持 `in_proj / conv1d / qchunk_scan / state / norm` 仍然走 Quamba W8A8 path，只替换 `out_proj` 处的 `had + out_proj` 组合。为了快速归因，本轮不写 CUDA kernel，而是用 PyTorch fake quant + FP matmul 做质量上界诊断；因此本轮 latency 只作为参考，不作为最终 kernel 性能结论。

新增 out_proj 诊断 mode：

| mode | out_proj 输入量化 | out_proj weight | 目的 |
|---|---|---|---|
| `w8a8_no_had_fp_outproj_all` | 不量化 | FP16 | Round 10 的质量上界 |
| `w8a8_no_had_outproj_fpweight_static_round_all` | static scale + round | FP16 | 只看 activation fixed-scale quantization 的影响 |
| `w8a8_no_had_outproj_w8_static_round_all` | static scale + round | W8 dequant | 看 weight W8 叠加后的影响 |

保留对照：

| mode | 目的 |
|---|---|
| `baseline_fp16` | 原始 BF16/FP16 baseline |
| `simple_no_had_reorder_all` | FP16 + CAWR reorder-only 等价性对照 |
| `w8a8_no_had_all` | Quamba 原生 W8A8 no-Had out_proj：static scale + truncate + W8 kernel |

### 实验配置

输出目录：

```text
results/quamba_compat/round11_outproj_quant_ablation
```

固定设置：

- model：`/scratch2/wl730/models/nemotron-h-8b`
- dataset for quality：WikiText test
- context length：`2048`
- batch size：`2`
- decode steps：`32`
- calibration source：`local_prompts`
- calibration samples：`8`
- calibration seq len：`128`
- CAWR grouped reorder：开启
- Hadamard：关闭
- Mamba layers：全部 24 个 Mamba layer

### 判断标准

1. 如果 `fpweight_static_round` 已经接近 FP16 `out_proj` 上界，说明 W8A8 原生退化主要来自 W8 weight 或 kernel/truncate 细节，而不是 activation static scale。
2. 如果 `fpweight_static_round` 仍明显差，说明 out_proj 输入 activation 的 fixed-scale quantization 本身就是主要问题。
3. 如果 `w8_static_round` 明显好于 `w8a8_no_had_all`，说明 Quamba 当前 `QAct` truncate 是重要误差源。
4. 如果 static round 仍远差于 FP16 `out_proj`，说明只靠 round 不能解决问题，下一步应优先分析 local Hadamard / local rotation 是否能把 static scale 下的 activation 分布变得更友好。

### 实验结果

结果文件：

```text
results/quamba_compat/round11_outproj_quant_ablation/quality_diagnosis.csv
```

| mode | CE | PPL | KL vs baseline | Top1 match | decode ms/step | peak memory GiB |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_fp16` | 1.908958 | 6.746056 | 0.000000 | 1.0000 | 33.1011 | 18.3819 |
| `simple_no_had_reorder_all` | 1.908764 | 6.744744 | 0.000015 | 1.0000 | 25.4357 | 18.3849 |
| `w8a8_no_had_all` | 3.763657 | 43.105756 | 1.829045 | 0.4375 | 24.1041 | 15.9359 |
| `w8a8_no_had_fp_outproj_all` | 2.366705 | 10.662200 | 0.348582 | 0.7500 | 22.6530 | 17.4359 |
| `w8a8_no_had_outproj_fpweight_static_round_all` | 3.127628 | 22.819792 | 1.039275 | 0.5469 | 24.1644 | 17.4359 |
| `w8a8_no_had_outproj_w8_static_round_all` | 3.326913 | 27.852227 | 1.263353 | 0.5469 | 24.4763 | 17.4359 |

### 结论

1. `out_proj` 输入 activation 的 static per-tensor scale 是当前最大问题。
   - 即使 `out_proj` weight 保持 FP16，只对输入 activation 使用 static scale + round，PPL 仍是 `22.82`，远差于 FP16 `out_proj` 上界 `10.66`。
   - 这说明问题不是单纯来自 W8 weight，而是 `out_proj` 前 activation 的量化尺度太粗。
2. W8 weight 叠加会进一步变差，但不是主因。
   - `static round + FP16 weight`：PPL `22.82`
   - `static round + W8 weight`：PPL `27.85`
   - 差距存在，但小于 static activation scale 相对 FP16 上界造成的差距。
3. Quamba 原生 `w8a8_no_had_all` 的 PPL `43.11` 比 fake `static round + W8` 的 `27.85` 更差，说明原生 `QAct` 的 truncate 和/或 CUTLASS int8 path 细节也有额外误差。
4. 本轮 fake-quant 使用 PyTorch matmul，不代表最终 latency。它的作用是确认：在不做 local Hadamard / local rotation 时，单纯把 `QAct` truncate 改成 round 仍然远远不够。

### 下一步

下一步回到已经验证有效的 local Hadamard / local rotation，分析它为什么能改善 static scale 下的 `out_proj` 量化：

1. 保持 `in_proj / conv / qchunk_scan / state / norm` 仍走 Quamba W8A8 path；
2. 对比 `w8a8_no_had_all`、`w8a8_local_out_had_all`、`w8a8_no_had_fp_outproj_all`；
3. 记录 local Hadamard 前后 `out_proj` 输入 activation 的 channel amax、token amax、scale utilization 和 clipping ratio；
4. 判断 local Hadamard 的收益来自 channel outlier smoothing，还是来自更适合 W8 weight / CUTLASS path 的输入分布。

## Quamba Compat：Round 10 W8A8 模块级 Ablation 第一批设计

### 背景

Round 9 已经证明 CAWR reorder-only 在 FP16 下等价，因此 W8A8 质量退化不来自 reorder 映射错误。下一步需要定位退化来自哪一段量化。

Quamba W8A8 Mamba2 内部不是普通 FP module 串联，而是带有 int8/scale 协议：

- `in_proj` 产生后续 conv / scan 使用的量化尺度；
- `Quamb2Conv1D` 输出 `x/B/C` 的量化表示和 grouped scale；
- `Quamba2ChunkScan` 使用 int8 state / x / B / C / dt / A / D scale；
- `QRMSNormGated` 输出 FP16；
- `QAct/QHadamard + W8A8 out_proj` 再把输出投影输入量化。

因此不能随便把 `conv` 或 `scan` 换回 FP16，否则可能会破坏真实 kernel 的输入协议。第一批先做最干净、物理意义明确的一项：

```text
w8a8_no_had_fp_outproj_all
```

它的流程：

1. 按完整 `w8a8_no_had_all` 流程配置 `Mamba2Simple`；
2. 执行 CAWR reorder；
3. 执行 activation calibration；
4. 转换成完整 Quamba `W8A8QMamba2`；
5. 只把最后的 `QAct + W8A8 out_proj` 换回 reorder 后的 FP16 `out_proj`；
6. 其余 `in_proj / conv1d / qchunk_scan / state / norm` 仍保持 W8A8 Quamba path。

这个实验回答的问题：

- 如果质量基本恢复，说明主要问题在 `out_proj` activation/weight quantization；
- 如果质量仍然差，说明主要问题在 `in_proj -> conv -> qchunk_scan/state -> norm` 这一侧。

### 实验配置

输出目录：

```text
results/quamba_compat/round10_module_ablation_fp_outproj
```

固定设置：

- model：`/scratch2/wl730/models/nemotron-h-8b`
- dataset for quality：WikiText test
- context length：`2048`
- batch size：`2`
- decode steps：`32`
- calibration source：`local_prompts`
- calibration samples：`8`
- calibration seq len：`128`
- CAWR grouped reorder：开启
- Hadamard：关闭，除已有 `w8a8_local_out_had_all` 对照外不使用全局 Hadamard
- Mamba layers：全部 24 个 Mamba layer

对比 mode：

| mode | 目的 |
|---|---|
| `baseline_fp16` | 原始 BF16 baseline |
| `simple_no_had_reorder_all` | FP16 + CAWR reorder-only 等价性对照 |
| `w8a8_no_had_all` | 完整 Quamba W8A8，无 Hadamard |
| `w8a8_no_had_fp_outproj_all` | 完整 W8A8，但最后 out_proj 恢复 FP16 |
| `w8a8_local_out_had_all` | 已知有效的 local out_proj Hadamard 对照 |

### 判断标准

1. 如果 `w8a8_no_had_fp_outproj_all` 接近 baseline，则 out_proj 量化是主因，下一步重点优化 hybrid-safe out_proj Hadamard / percentile scale / output projection quantization。
2. 如果它只小幅改善，则 out_proj 不是主因，下一步要深入 `conv + qchunk_scan/state`。
3. 如果它比 `w8a8_local_out_had_all` 还差，说明 local Hadamard 的收益不只是减少 out_proj 量化误差，也可能改善了前面 norm/output activation 分布。

### 实验结果

结果文件：

```text
results/quamba_compat/round10_module_ablation_fp_outproj/quality_diagnosis.csv
```

| mode | CE | PPL | KL vs baseline | Top1 match | decode ms/step | prefill s | peak memory GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_fp16` | 1.908958 | 6.746056 | 0.000000 | 1.0000 | 33.8363 | 6.1482 | 18.3819 |
| `simple_no_had_reorder_all` | 1.908764 | 6.744744 | 0.000015 | 1.0000 | 25.0126 | 0.3484 | 18.3849 |
| `w8a8_no_had_all` | 3.763657 | 43.105756 | 1.829045 | 0.4375 | 24.1508 | 6.2109 | 15.9359 |
| `w8a8_no_had_fp_outproj_all` | 2.366705 | 10.662200 | 0.348582 | 0.7500 | 22.5985 | 0.3026 | 17.4359 |
| `w8a8_local_out_had_all` | 2.815909 | 16.708355 | 0.747620 | 0.7031 | 23.3685 | 0.3283 | 15.9359 |

### 结论

1. `out_proj` 量化是当前 W8A8 质量退化的主要来源之一。把最后的 `QAct + W8A8 out_proj` 换回 FP16 后，PPL 从 `43.11` 降到 `10.66`，KL 从 `1.829` 降到 `0.349`，Top1 match 从 `43.75%` 提升到 `75.00%`。
2. `w8a8_no_had_fp_outproj_all` 的质量明显好于 `w8a8_local_out_had_all`，说明当前 local output Hadamard 方向是对的，但还没有充分解决 `out_proj` 输入 activation / weight quantization 的尺度和 outlier 问题。
3. 恢复 FP16 `out_proj` 后仍然没有回到 baseline PPL `6.746`，说明 `in_proj -> conv -> qchunk_scan/state -> norm` 这一侧仍有剩余误差，但优先级低于 `out_proj`。
4. `simple_no_had_reorder_all` 仍然和 baseline 基本等价，进一步确认 CAWR reorder 映射本身不是质量炸掉的原因。

### 下一步

下一步不应该先拆 `conv/scan`，因为这些模块有 Quamba 内部 int8/scale 协议，直接替换容易得到不物理的结果。更合理的路径是先做 `out_proj` 专项优化：

1. 保持其余 Quamba W8A8 path 不变，只研究 `out_proj` 前的 activation distribution；
2. 对比 `QAct` 当前 scale、percentile scale、校准集 scale、per-channel / per-group input scale；
3. 在 `out_proj` 前做 hybrid-safe local rotation，并确认它不需要全模型 Hadamard；
4. 以 `w8a8_no_had_fp_outproj_all` 作为质量上界，判断每个 `out_proj` quantization 改法离上界还有多远。

## Quamba Compat：Round 9 CAWR Reorder-only 等价性检查设计

### 背景

前面已经确认两件事：

1. `simple_no_had_all` 与 BF16 baseline 基本等价，说明 Nemotron-H 到 Quamba `Mamba2Simple` 的 adapter、norm fuse 和 cache adapter 大体正确。
2. W8A8 全 Mamba 替换质量仍然明显退化，且官方 Pile calibration 没有解决问题。

因此下一步需要先确认 Cluster-Aware Weight Reordering 本身是不是等价。如果只做 CAWR reorder、不做任何 W8A8 量化，模型输出就已经偏离 baseline，那么后续所有 W8A8 结果都不能被解释为“量化误差”，必须先修 reorder 映射。

### 本轮新增 mode

在 `scripts/quamba_compat/diagnose_quamba_quality.py` 中新增：

```text
simple_no_had_reorder_all
```

执行流程：

1. 把所有 Mamba layers 替换为 Quamba `Mamba2Simple(use_had_transform=False)`；
2. hook `x_conv_out`，收集 CAWR reorder stats；
3. 调用 Quamba 官方 `group_wise_sort_indices`；
4. 对 `in_proj / conv1d / A_log / D / dt_bias / norm / out_proj` 做离线 reorder；
5. 不做 activation calibration；
6. 不调用 `W8A8QMamba2.from_fp16`；
7. 不做任何 weight/activation/state quantization。

### 实验配置

输出目录：

```text
results/quamba_compat/round9_reorder_only_equivalence
```

固定设置：

- model：`/scratch2/wl730/models/nemotron-h-8b`
- dataset for quality：WikiText test
- context length：`2048`
- batch size：`2`
- decode steps：`32`
- calibration source：`local_prompts`
- calibration samples：`8`
- calibration seq len：`128`
- Hadamard：关闭
- quantization：关闭
- Mamba layers：全部 24 个 Mamba layer

对比 mode：

| mode | 目的 |
|---|---|
| `baseline_fp16` | 原始 BF16 baseline |
| `simple_no_had_all` | 只替换 Quamba `Mamba2Simple`，不做 CAWR |
| `simple_no_had_reorder_all` | `Mamba2Simple + CAWR reorder-only`，不做量化 |

### 判断标准

1. 如果 `simple_no_had_reorder_all` 与 `simple_no_had_all` / baseline 的 PPL、KL、top1 match 基本一致，则 CAWR reorder 映射在当前 hybrid Mamba 子层内部是等价的，后续可继续定位 W8A8 量化误差。
2. 如果 `simple_no_had_reorder_all` 明显变差，则说明 CAWR reorder 在 Nemotron-H adapter 中还有映射 bug 或者和 hybrid block 参数形状不完全匹配，后续必须先修 reorder。
3. 如果质量等价但 latency 变化明显，则说明 reorder 可能改变了 kernel/memory layout 行为，需要单独 profile；但质量诊断优先。

### 实验结果

结果文件：

```text
results/quamba_compat/round9_reorder_only_equivalence/quality_diagnosis.csv
```

| mode | CE | PPL | KL vs baseline | top1 match | decode ms/step | prefill s | peak memory GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_fp16` | 1.908958 | 6.746056 | 0.000000 | 1.000 | 33.114 | 6.156 | 18.382 |
| `simple_no_had_all` | 1.908983 | 6.746224 | 0.000014 | 1.000 | 25.252 | 0.373 | 18.385 |
| `simple_no_had_reorder_all` | 1.908764 | 6.744744 | 0.000015 | 1.000 | 25.778 | 0.351 | 18.385 |

`simple_no_had_reorder_all` 的 prepare metadata 确认：

- CAWR reorder layers：全部 24 个 Mamba layer；
- calibration source：`local_prompts`；
- calibration samples：`8`；
- calibration seq len：`128`；
- `use_group_heads=true`；
- 没有 W8A8 quantization。

### 结论

1. CAWR reorder-only 在当前 Nemotron-H Mamba 子层内部是等价的。
   - PPL 与 baseline / `simple_no_had_all` 基本一致；
   - KL 只有 `1.5e-5`；
   - top1 match 仍是 `100%`。
2. 因此当前 W8A8 质量退化不能归因于 CAWR reorder 映射错误。
   - `in_proj / conv1d / A_log / D / dt_bias / norm / out_proj` 的离线 reorder 与 Quamba `Mamba2Simple` 路径在 FP16 下保持等价。
3. 后续应把重点转向量化误差定位，而不是继续修 reorder。
   - 下一步应该做模块级 W8A8 ablation：`in_proj only`、`conv1d only`、`qchunk_scan/state only`、`out_proj only`、`conv+scan`、full W8A8。
   - 每组测 PPL、KL、top1、decode latency，并记录每层 output cosine / relative error。
4. latency 上，`simple_no_had_reorder_all` 比 `simple_no_had_all` 略慢约 `0.53 ms/step`，但这是 FP wrapper 路径的布局/实现差异，不是质量问题。真正的性能判断仍应在 quantized kernel path 上做。

## Quamba Compat：Round 8 官方 Pile Calibration 设计

### 背景

Round 7 的 `w8a8_local_out_had_all` 使用的是本项目为了快速 smoke test 写的短 prompt calibration：

```text
calib_source = local_prompts
num_calib_samples = 8
calib_seq_len = 128
```

这不是 Quamba 官方设置。Quamba 官方 `run_quamba2_calibration` 默认使用：

```text
dataset = monology/pile-uncopyrighted
data_files = val.jsonl.zst
split = train
seed = 42
num_samples = 512
seq_len = 512
```

因此 Round 7 只能说明 local out_proj Hadamard 的方向有效，不能说明 W8A8 质量上限。

### 本轮改动

把 `scripts/quamba_compat` 的 calibration 输入改成显式参数：

```text
--calib-source local_prompts
--calib-source quamba_pile
```

其中 `quamba_pile` 会按 Quamba 默认路径加载：

```python
load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train").shuffle(seed=42)
```

tokenize 方式也按 Quamba 默认逻辑：

```python
tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
```

### 实验配置

输出目录：

```text
results/quamba_compat/round8_quamba_pile_calib
```

固定设置：

- model：`/scratch2/wl730/models/nemotron-h-8b`
- dataset for quality：WikiText test
- quality context length：`2048`
- batch size：`2`
- decode steps：`32`
- calibration source：`quamba_pile`
- calibration samples：`512`
- calibration seq len：`512`
- grouped reorder scale：开启
- Mamba layers：全部 24 个 Mamba layer

实验 mode：

| mode | 目的 |
|---|---|
| `baseline_fp16` | 原始 BF16 baseline |
| `w8a8_no_had_all` | 官方 Pile calibration 下的 W8A8 无 Hadamard |
| `w8a8_local_out_had_all` | 官方 Pile calibration 下的 W8A8 + hybrid-safe local out_proj Hadamard |

### 判断标准

1. 如果官方 Pile calibration 明显降低 `w8a8_local_out_had_all` 的 PPL/KL，说明 Round 7 主要受 calibration 覆盖不足限制。
2. 如果改善有限，则剩余误差可能来自：
   - all-Mamba-layer 同时 W8A8 的误差累积；
   - `qchunk_scan` / conv / state scale 本身；
   - hybrid 模型里 attention/MLP 对 Mamba 输出误差更敏感。
3. 如果 `w8a8_no_had_all` 也明显改善，但仍弱于 local out_proj Hadamard，则 local Hadamard 可作为 hybrid-specific co-design 点继续推进。

### 实验结果

结果文件：

```text
results/quamba_compat/round8_quamba_pile_calib/quality_diagnosis.csv
```

| mode | CE | PPL | KL vs baseline | top1 match | decode ms/step | prefill s | peak memory GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_fp16` | 1.909 | 6.746 | 0.000 | 1.000 | 33.186 | 6.137 | 18.382 |
| `w8a8_no_had_all` | 6.002 | 404.359 | 3.855 | 0.109 | 24.160 | 6.224 | 15.936 |
| `w8a8_local_out_had_all` | 4.035 | 56.563 | 1.950 | 0.406 | 23.144 | 0.353 | 15.936 |

和 Round 7 的 local prompt calibration 对比：

| mode | Round 7 local prompt PPL | Round 8 Quamba Pile PPL | Round 7 top1 | Round 8 top1 |
|---|---:|---:|---:|---:|
| `w8a8_no_had_all` | 43.106 | 404.359 | 0.438 | 0.109 |
| `w8a8_local_out_had_all` | 16.708 | 56.563 | 0.703 | 0.406 |

### 结论

1. 之前 Round 7 的 calibration 确实是本项目自定义的短 prompt calibration，不是 Quamba 官方数据集。
2. 已经按 Quamba 默认的 `monology/pile-uncopyrighted/val.jsonl.zst`、`512` samples、`512` seq len 跑通完整 calibration。
3. 官方 Pile calibration 在当前 Nemotron-H hybrid 设置下没有改善质量，反而明显变差。
   - `w8a8_no_had_all`：PPL 从 `43.106` 变成 `404.359`。
   - `w8a8_local_out_had_all`：PPL 从 `16.708` 变成 `56.563`。
4. local out_proj Hadamard 在官方 Pile calibration 下仍然有效，但不足以恢复质量。
   - PPL：`404.359 -> 56.563`
   - KL：`3.855 -> 1.950`
   - top1 match：`0.109 -> 0.406`
5. 这说明当前问题不是简单的“calibration 数据太少”。更可能是 Quamba 针对纯 Mamba 的 min/max calibration 和当前 hybrid 模型的激活分布不匹配：
   - Pile calibration 覆盖更宽，会让 min/max observer 捕获更多 outlier，scale 变大，常见激活区间的 int8 分辨率反而下降；
   - Nemotron-H 是 attention/Mamba/MLP hybrid，Mamba 输出误差会被后续 attention/MLP 放大，和纯 Mamba 模型的误差传播不同；
   - WikiText decode slice 与 Pile validation calibration 的分布不完全一致，本地 prompt calibration 反而更接近当前测试路径的激活范围。

下一步不应该继续假设“官方 Pile calibration 一定更好”。更合理的方向是做 hybrid-aware calibration：

- 比较 `local_prompts`、`quamba_pile`、WikiText train calibration 和更贴近目标任务的 prompt calibration；
- 不只改数据源，还要改 observer 策略，比如 percentile / clipping-aware scale，而不是纯 min/max；
- 记录每层 activation clipping ratio、scale range、output cosine 和 KL，确认是少数 outlier 拉大 scale，还是所有层都有系统性偏移。

## Quamba Compat：Round 7 局部 out_proj Hadamard 实验设计

### 背景

之前的 Quamba W8A8 Mamba 替换实验显示：

- `simple_no_had_all` 与 BF16 baseline 基本一致，说明 Nemotron-H 到 Quamba `Mamba2Simple` 的 adapter、cache adapter 和 norm fuse 路径是正确的。
- `simple_had_all` 在不量化时就已经质量崩溃，说明直接使用 Quamba 的全局 Hadamard 假设不适合 hybrid partial replacement。
- `w8a8_no_had_all` 不崩溃，但 PPL 明显上升，说明 W8A8 缺少 Hadamard smoothing 后 out_proj activation / weight quantization 仍有明显误差。

### 假设

只在 Mamba block 内部的 `out_proj` 前做局部 Hadamard，并把逆变换等价地 fuse 到 `out_proj` 的输入维度上，可以满足两点：

1. FP 路径上保持 block 外部输入/输出仍在原始 residual basis，不影响 attention/MLP/embedding/lm_head。
2. W8A8 路径上仍能对 `out_proj` 输入做 Hadamard smoothing，缓解 activation outlier 对量化的影响。

局部形式为：

```text
原始：out = W_out y
局部：out = (W_out H) (H y)
```

这里 `H` 使用 Quamba 的正交归一化 Hadamard。由于只改变 `out_proj` 局部输入坐标，不传播到 residual stream，因此它应当比 Quamba 原生全局 Hadamard 更适合 Nemotron-H 这种 attention + Mamba hybrid 结构。

### 本轮实验

输出目录：

```text
results/quamba_compat/round7_local_out_had
```

使用相同 prompt 数据和诊断设置：

- model：`/scratch2/wl730/models/nemotron-h-8b`
- dataset：`wikitext`
- context length：`2048`
- batch size：`2`
- decode steps：`32`
- calibration samples：`8`
- calibration seq len：`128`
- Mamba layers：全部 24 个 Mamba layer
- Attention/MLP/embedding/final norm/lm_head：保持原始 BF16/FP16 路径

实验 mode：

| mode | 目的 |
|---|---|
| `baseline_fp16` | 原始模型质量和 latency baseline |
| `simple_no_had_all` | 确认 Quamba `Mamba2Simple` adapter 本身仍然正确 |
| `simple_local_out_had_all` | 检查局部 out_proj Hadamard 在 FP 路径是否数学等价 |
| `w8a8_no_had_all` | 当前可跑通的 W8A8 无 Hadamard 版本 |
| `w8a8_local_out_had_all` | 本轮核心：局部 out_proj Hadamard + W8A8 Mamba |

### 判断标准

1. 如果 `simple_local_out_had_all` 与 baseline 的 KL/top1/PPL 基本一致，则局部 Hadamard 的插入位置和权重 fuse 方向正确。
2. 如果 `w8a8_local_out_had_all` 相比 `w8a8_no_had_all` PPL/KL/top1 明显改善，则说明 hybrid-safe local Hadamard 是一个可继续深入的方向。
3. 如果 FP 等价但 W8A8 改善有限，则问题主要在其他 W8A8 quantization scale / state / conv / scan 路径，而不是全局坐标错配。

### 实验结果

输出目录：

```text
results/quamba_compat/round7_local_out_had
```

结果文件：

- `quality_diagnosis.csv`
- `summary.json`
- 各 mode 的 `*_prepare.json`

核心结果：

| mode | CE | PPL | KL vs baseline | top1 match | decode ms/step | peak memory GiB |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_fp16` | 1.909 | 6.746 | 0.000000 | 1.000 | 33.797 | 18.382 |
| `simple_no_had_all` | 1.909 | 6.746 | 0.000014 | 1.000 | 26.097 | 18.385 |
| `simple_local_out_had_all` | 1.909 | 6.743 | 0.000017 | 1.000 | 26.991 | 18.385 |
| `w8a8_no_had_all` | 3.764 | 43.106 | 1.829045 | 0.438 | 25.015 | 15.936 |
| `w8a8_local_out_had_all` | 2.816 | 16.708 | 0.747620 | 0.703 | 23.013 | 15.936 |

### 初步判断

1. 局部 out_proj Hadamard 的 FP 等价性成立。
   - `simple_local_out_had_all` 的 PPL、KL、top1 都和 baseline 基本一致。
   - 这说明只在 `out_proj` 前做 `H y`，并把 `W_out H` fuse 到权重里，没有破坏 hybrid residual 坐标系。
2. 相比无 Hadamard W8A8，局部 out_proj Hadamard 明显改善质量。
   - PPL：`43.106 -> 16.708`
   - KL：`1.829 -> 0.748`
   - top1 match：`43.8% -> 70.3%`
3. latency 也没有变差，反而略有下降。
   - `25.015 -> 23.013 ms/step`
   - 这可能和 `QHadamard + out_proj` 的 kernel/scale 路径更接近 Quamba 官方 W8A8 输出投影路径有关，后续需要用 nsys 进一步确认。
4. 这仍然不是最终可用质量。
   - PPL `16.7` 仍明显高于 baseline `6.75`。
   - 但它证明了“Quamba Hadamard 不能全局照搬，但可以做 hybrid-safe 局部化”的方向是有效的。

下一步建议：

1. 把 `w8a8_local_out_had_all` 作为新的 Quamba W8A8 默认诊断路径，废弃 `use_had_transform=True` 的全局 partial replacement 路径。
2. 扩大 calibration：从当前 `8 x 128` 改成至少 `128 x 512`，并记录 clipping ratio。
3. 做 layer-wise / prefix-wise W8A8 local-out-Hadamard 替换，判断剩余 PPL 误差是否来自少数敏感 Mamba 层。
4. 单独 profile `had + out_proj`、`qchunk_scan.update`、`conv update`，确认 latency 下降来自哪里。

## Quamba Compat：Round 1 Nemotron-H Mamba Mixer 兼容性审计设计

### 目的

在尝试把 Quamba/Quamba2 的 Mamba weight quantization 和 quantized state update kernel 接入 Nemotron-H-8B 之前，先判断当前模型的 Mamba mixer 是否和 Quamba2 的 `W4A8QMamba2/W8A8QMamba2` 所要求的数据流兼容。

本轮不做 PPL/latency，不修改模型权重，只做结构 introspection。

### 检查项

1. 层类型分布：
   - Mamba layer / attention layer / MLP layer 数量与 index。
2. 每个 Mamba mixer 的 Python class 与 module 路径。
3. Quamba2 必需属性是否存在：
   - `d_model`
   - `d_state` 或 `ssm_state_size`
   - `d_conv`
   - `conv1d`
   - `in_proj`
   - `out_proj`
   - `A_log`
   - `D`
   - `dt_bias`
   - `num_heads` / `nheads`
   - `head_dim` / `headdim`
   - `ngroups`
   - `d_ssm`
   - `chunk_size`
   - `rmsnorm`
   - `norm`
4. `in_proj` 输出维度能否按 Quamba2 decode step 的 split 方式解释：

```text
[d_mlp, d_mlp, d_ssm, d_ssm + 2 * ngroups * d_state, nheads]
```

5. `conv1d` 输出维度是否能按 Quamba2 的 `x/B/C` 三段解释：

```text
x_dim + 2 * ngroups * d_state
```

6. cache 结构是否能对应 Quamba2：
   - `conv_states[layer]`
   - `ssm_states[layer]`
   - 当前 state view 是否为 `[batch, num_heads, head_dim, ssm_state_size]`

### 输出

- 结果目录：`results/quamba_compat/round1_mixer_audit`
- `nemotron_mamba_mixer_audit.json`
- `nemotron_mamba_mixer_audit.csv`

### 判断标准

- 如果 Mamba mixer 属性、shape、split 规则基本匹配 Quamba2，则可以进入下一步：实现 Nemotron-H adapter，把 Mamba layer 替换成 Quamba-style quantized mixer。
- 如果属性名不同但 shape 语义一致，则仍可做 adapter，但不能直接 import Quamba 的 `W4A8QMamba2.from_fp16`。
- 如果 `in_proj/conv1d/state` 的 split 语义不同，则不能复用 Quamba2 block，需要只借鉴 kernel/scale 思路，写 Nemotron-H 专用量化 Mamba block。

### 结果

输出目录：

```text
results/quamba_compat/round1_mixer_audit
```

输出文件：

- `nemotron_mamba_mixer_audit.json`
- `nemotron_mamba_mixer_audit.csv`

模型结构：

- 总层数：`52`
- Mamba layers：`24`
- Attention layers：`4`，index 为 `[7, 18, 29, 40]`
- MLP layers：`24`
- hybrid pattern：

```text
M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-
```

所有 24 个 Mamba layer 都是同一种结构：

| item | value |
|---|---:|
| mixer class | `NemotronHMamba2Mixer` |
| `d_state` / `ssm_state_size` | `128` |
| `nheads` / `num_heads` | `128` |
| `headdim` / `head_dim` | `64` |
| inferred `ngroups` | `8` |
| `d_ssm` | `8192` |
| `in_proj` shape | `[18560, 4096]` |
| `conv1d` out channels | `10240` |
| `conv1d` kernel | `4` |
| `ssm_state` cache shape | `[B, 8192, 128]` |
| `ssm_state` Quamba-style view | `[B, 128, 64, 128]` |

兼容性检查：

| check | pass |
|---|---:|
| Quamba2 `in_proj` split | `24 / 24` |
| Quamba2 `conv1d` split | `24 / 24` |
| Quamba2 state view | `24 / 24` |

具体 split：

```text
projection_size = intermediate_size + conv_dim + num_heads
                = 8192 + 10240 + 128
                = 18560

conv_dim = intermediate_size + 2 * n_groups * ssm_state_size
         = 8192 + 2 * 8 * 128
         = 10240

Quamba2 d_mlp = 0
```

源码检查：

- Nemotron-H 的 decode fast path 使用：

```text
_, _, gate, hidden_states_B_C, dt = projected_states.split(
    [d_mlp, d_mlp, intermediate_size, conv_dim, num_heads]
)
hidden_states, B, C = torch.split(
    hidden_states_B_C,
    [intermediate_size, n_groups * ssm_state_size, n_groups * ssm_state_size]
)
```

这和 Quamba2 的 decode step 语义一致，只是字段名不同：

- Quamba2：`ngroups`, `d_state`, `d_ssm`, `headdim`
- Nemotron-H：`n_groups`, `ssm_state_size`, `intermediate_size`, `head_dim`

### 判断

第一步结论是：**形状和数据流语义基本兼容，但不能直接调用 Quamba 的现成 quantized mixer class。**

原因：

1. Nemotron-H mixer 不是 Quamba 代码里期待的 `Mamba2Simple` class。
2. 字段名不同，例如 `n_groups` vs `ngroups`，`ssm_state_size` vs `d_state`，`intermediate_size` vs `d_ssm`。
3. Nemotron-H 的 state cache 当前是 BF16 `[B, 8192, 128]`，Quamba2 quantized path 需要 int8 state 和 calibrated/grouped `ssm_state_scale`。
4. Quamba2 的收益来自 `in_proj -> conv1d -> qchunk_scan -> norm/out_proj` 的端到端 int8/W4A8 数据流，不是只替换 state update。

因此下一步可以做，但应该实现一个 Nemotron-H 专用 adapter：

1. 复用 Quamba 的 W4A8/W8A8 linear、conv、qchunk scan 思路。
2. 为 `NemotronHMamba2Mixer` 写 `from_nemotron_fp16`，显式映射字段名和 split。
3. 先做 W8A8 版本，避免 W4/GPTQ 先引入太多变量。
4. 先只替换 Mamba layers，attention/MLP 保持原样。

## Quamba Compat：Round 2 Quamba Extension 与 Adapter 实施设计

### 目的

把 Nemotron-H 的 Mamba layer 接到 Quamba/Quamba2 的量化 kernel 数据流上，而不是只替换 state update。目标路径是：

```text
BF16 hidden
  -> quantized in_proj
  -> quantized conv1d update
  -> Quamba2 qchunk_scan.update
  -> gated norm / output projection
```

### 当前前置检查

服务器项目环境 `.venv` 中已有：

- `torch`
- `triton`
- `causal_conv1d`
- `mamba_ssm`

但缺少 Quamba 自己的 CUDA extension：

- `quant_linear_cuda`
- `quamba2_conv1d_cuda`
- `quant_causal_conv1d_cuda`
- `quant_sscan_cuda`
- `rms_norm_cuda`
- `fast_hadamard_transform_cuda`

因此当前不能直接 import Quamba 的 `qLinearLayer/qConvLayer/qMamba2` 完整路径。

### 实施顺序

1. 只在项目本地 `.venv` 内编译 `third_party/Quamba_fresh`，不修改系统 Python / CUDA / 全局环境。
2. 编译前检查：
   - `nvcc -V`
   - `torch.cuda.get_device_capability()`
   - GPU 是否空闲
3. 编译后重新运行 import probe，确认关键扩展可用。
4. 写 Nemotron-H adapter：
   - 不直接调用 `W4A8QMamba2.from_fp16`；
   - 新建 `from_nemotron_fp16` 映射字段：
     - `n_groups -> ngroups`
     - `ssm_state_size -> d_state`
     - `intermediate_size -> d_ssm`
     - `head_dim -> headdim`
     - `num_heads -> nheads`
   - 第一版用 W8A8，避免 W4/GPTQ 和 Hadamard 同时引入变量。
5. 做一个单层 smoke test：
   - 只替换 layer 0 的 Mamba mixer；
   - batch=1, context 很短；
   - 检查 forward/decode 是否能跑通。

### 判断标准

- 如果 Quamba extension 无法在项目 `.venv` 内编译，则这条“直接复用 Quamba kernel”路线暂时不可执行，需要退回到复用其 Triton update 思路。
- 如果 extension 可用且单层 smoke test 通过，再扩展到全部 Mamba layers，并正式测 PPL / latency。

## Quamba 重新 Clone 与接入检查

### 目的

重新拉取一份干净的 Quamba 代码，检查能否通过“微小修改”直接把它的 Mamba/Mamba2 state quantization kernel 接入当前 Nemotron-H-8B 的 decode 实验，用来解决当前 MX8 state 每步 dequant/update/requant 带来的质量和 latency 问题。

### 本地代码状态

- 新 clone 目录：`third_party/Quamba_fresh`
- commit：`c474646`
- 未覆盖已有 `third_party/Quamba`

### 代码检查结论

Quamba 的 state update kernel 不能零改动直接替换当前 Nemotron-H 的 `selective_state_update`：

1. Quamba 的 `quant_sscan_update_triton` 要求输入已经是量化后的 `q_x/q_dt/q_A_log/q_B/q_C`，并额外传入 `x_scale/dt_scale/A_log_scale/B_scale/C_scale/ssm_state_scale`。
2. 当前 Nemotron-H decode path 调用 `selective_state_update(ssm_state, x, dt, A, B, C, ...)` 时，传入的是 BF16/FP32 的 `x/dt/A/B/C`，没有 Quamba 所需的量化 activation 和 scale。
3. Quamba2 的高质量 state scale 不是单次 post-prefill `amax/127`，而是通过 calibration + head/channel grouping 得到：
   - `x_head_group_range`
   - `x_dim_group_range`
   - `x_scales`
   - `ssm_state_scale`
4. Quamba2 还会对 `x_conv_out` 做 sorting/clustering，并把 conv/scan 侧的 scale 绑定起来。这和我们当前单独量化 state 的 kernel 不是同一个设置。

### 可行接入方向

不建议直接 import Quamba 的完整 model conversion。更合理的最小实验是：

1. 复用 Quamba 的 grouped state scale 思路，而不是直接复用完整量化模型。
2. 新增一个 calibration pass，收集当前 Nemotron-H 的 BF16 Mamba state range：
   - shape：`[batch, nheads, head_dim, dstate]`
   - grouping：先用固定 `nhead_groups=4, ndim_groups=4`，不做 weight reorder。
   - scale shape：`[ngroups, nhead_groups, ndim_groups, dstate]`
3. 修改当前 `static_mx8` kernel 的 scale load 逻辑，按 Quamba2 风格根据 head/dim group 读取 scale。
4. 先做三组对比：
   - 当前 `mx8_r1`：每步刷新 scale
   - 当前 `static_mx8`：post-prefill 单点 scale
   - 新的 `calibrated_group_static_mx8`：校准得到的 grouped static scale

### 初步判断

这条路线比直接套 Quamba kernel 更合理，因为它专门针对我们现在的问题：state scale 固定后覆盖不住 decode 动态范围。它保留当前 Nemotron-H decode 结构，不要求把整个 Mamba block 改造成 Quamba 的 int8 activation pipeline。

## MX8 State Kernel：Round 3 Tiled Update Latency 设计

### 目的

验证刚刚修改后的 MX8 Mamba state update kernel 是否降低 decode latency。修改点是把标准 `mx8_r1` 路径从“一行 state 一个 Triton program”改成 tiled kernel：一个 program 处理 16 个 `dim` rows，并保留默认逐元素 stochastic rounding，因此本轮只观察 kernel 调度粒度变化带来的 latency 影响。

### 对比组

1. `bf16`
   - 不启用 Mamba state quantization。
2. `mx8_r1_tiled`
   - 启用当前 MX8 state kernel。
   - `scale_refresh_interval=1`，每步仍然刷新 MX8 shared/micro scale。
   - `stochastic=True`，`sr_pair_shared=False`，保持默认逐元素 SR，不引入额外精度变量。

### 参数

- 模型：Nemotron-H-8B
- 模型路径：`/scratch2/wl730/models/nemotron-h-8b`
- 代码路径：`/scratch2/wl730/hybrid_codesign/han/chunk_update`
- context length：`2048`
- batch size：`8`
- decode steps：`128`
- warmup steps：`4`
- 数据：WikiText test
- 输出文件：`results/latency/data/nemotron_8b_mx8_tiled_ctx2048_latency.csv`

### 运行命令

```bash
python scripts/latency/run_nemotron_8b_latency.py \
  --sequence-length 2048 \
  --batch-sizes 8 \
  --decode-steps 128 \
  --warmup-steps 4 \
  --quantization-modes none,state_mx8 \
  --output results/latency/data/nemotron_8b_mx8_tiled_ctx2048_latency.csv
```

### 判断标准

- 如果 `state_mx8` 的 `decode_ms_per_step` 明显下降，说明之前主要瓶颈之一确实是 row-level Triton program 粒度过碎。
- 如果下降很小，则说明主瓶颈更可能来自 `exp/log2/ceil` scale 计算、SR、或者每步 dequant/update/requant 的算术本身，需要继续分别测试 `scale_refresh_interval>1`、`sr_pair_shared=True` 和 deterministic rounding。

### 运行中的修正

第一次正式运行时，MX8 组出现 GPU 利用率为 0、Python 进程 CPU 接近 100% 的长时间停顿。检查后发现 `RNG_OFFSET` 被声明为 `tl.constexpr`，但它每个 decode step 都不同，因此 Triton 会为每个 step 重新 specialization / JIT 编译。

修正：

- 将所有 state kernel 的 `RNG_OFFSET` 从 `tl.constexpr` 改成 runtime scalar 参数。
- 保留 `RNG_SEED` 为 constexpr。
- 继续保持 `MX8_UPDATE_BLOCK_M = 16`，没有降低 tile 大小。

### 结果

输出文件：

- `results/latency/data/nemotron_8b_mx8_tiled_ctx2048_latency.csv`

| mode | ctx | batch | decode steps | decode ms/step | decode ms/token | tokens/s | p50 step ms | p90 step ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 2048 | 8 | 128 | 29.587 | 3.698 | 270.385 | 29.493 | 29.816 |
| mx8_r1_tiled | 2048 | 8 | 128 | 34.857 | 4.357 | 229.509 | 34.787 | 34.964 |

和同轮 BF16 相比：

- MX8 tiled 额外 latency：`+5.270 ms/step`
- 相对 overhead：`+17.8%`
- throughput 下降：`270.4 -> 229.5 tokens/s`

和之前本地记录的旧 row-kernel `state_mx8` ctx2048/batch8 结果粗略比较：

- 旧结果：`46.386 ms/step`，decode steps=32
- 新 tiled 结果：`34.857 ms/step`，decode steps=128
- 直接看 MX8 latency 下降约 `24.9%`
- 旧结果不是同一次运行且 decode steps 不同，所以只作为方向性参考；可信对比应以后用同一脚本多 trial 重跑 old/new 或用 git commit 对照。

结论：

1. row-level program 粒度确实是一个明显瓶颈，tiled update 后 MX8 state quantization 的 decode overhead 从旧记录的约 47% 降到本轮同测的约 18%。
2. `RNG_OFFSET` constexpr 是隐藏的大问题，会导致每步 JIT specialization。这个修正本身也应该保留。
3. MX8 仍然比 BF16 慢，剩余主要开销大概率来自：
   - 每步重新计算 shared/micro scale 的 `amax/log2/ceil/exp2`；
   - stochastic rounding；
   - update 后再次量化写回。

下一步建议：

- 在同一个 ctx2048 设置下继续测：
  - `scale_refresh_interval=2/4/8`
  - `sr_pair_shared=True`
  - `stochastic=False`
- 每组都需要同时测 PPL / KL / top1，确认 latency 优化没有引入不可接受的精度损失。

## MX8 State Kernel：Round 4 BLOCK_M=32 + Group-shared SR Latency 设计

### 目的

在 Round 3 tiled MX8 state update kernel 的基础上继续降低 per-step update/requant 开销：

1. 将 MX8 update 的 `BLOCK_M` 从 16 改成 32，让一个 Triton program 一次处理更多 head-dim row，减少 program launch/grid 粒度过碎带来的开销。
2. 将 stochastic rounding 从逐元素随机数改成每个 row 的 16-value MX block 共用一个随机数，降低 `tl.rand` 生成和向量随机比较的开销。
3. 保持 MX8 动态 scale/requant 逻辑不变，不引入 scale refresh 或 static scale，避免这轮测试混入精度风险。

### 对比组

- `bf16_full`：不启用 state quantization。
- `state_mx8_bm32_group_sr`：启用 MX8 state quantization，`BLOCK_M=32`，每个 16-value MX block 共用一个 stochastic rounding 随机数。

### 参数

- 模型：Nemotron-H-8B
- 模型路径：`/scratch2/wl730/models/nemotron-h-8b`
- 代码路径：`/scratch2/wl730/hybrid_codesign/han/chunk_update`
- context length：`2048`
- batch size：`8`
- decode steps：`128`
- warmup steps：`4`
- 数据：WikiText test
- 输出文件：`results/latency/data/nemotron_8b_mx8_bm32_group_sr_ctx2048_latency.csv`

### 运行命令

```bash
python scripts/latency/run_nemotron_8b_latency.py \
  --sequence-length 2048 \
  --batch-sizes 8 \
  --decode-steps 128 \
  --warmup-steps 4 \
  --quantization-modes none,state_mx8 \
  --output results/latency/data/nemotron_8b_mx8_bm32_group_sr_ctx2048_latency.csv
```

### 判断标准

- 主要对比 Round 3 的 `state_mx8`：`34.857 ms/step`。
- 如果 `state_mx8_bm32_group_sr` 明显下降，说明 tiled granularity 和 SR 随机数生成是剩余 overhead 的重要来源。
- 如果下降不明显，说明剩余瓶颈更可能来自动态 MX8 scale 的 `amax/log2/ceil/exp2`、重复读取/重算 group state，或者 update/requant 本身的访存。

### 结果

输出文件：

- `results/latency/data/nemotron_8b_mx8_bm32_group_sr_ctx2048_latency.csv`

| mode | ctx | batch | decode steps | decode ms/step | decode ms/token | tokens/s | p50 step ms | p90 step ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 2048 | 8 | 128 | 29.980 | 3.748 | 266.841 | 29.921 | 30.136 |
| mx8_bm32_group_sr | 2048 | 8 | 128 | 32.787 | 4.098 | 243.997 | 32.750 | 32.929 |

和同轮 BF16 相比：

- MX8 BM32/group-shared SR 额外 latency：`+2.807 ms/step`
- 相对 overhead：`+9.4%`
- throughput 下降：`266.8 -> 244.0 tokens/s`

和 Round 3 的 MX8 tiled BM16/逐元素 SR 相比：

| version | MX8 decode ms/step | same-run BF16 ms/step | MX8 overhead vs BF16 |
|---|---:|---:|---:|
| BM16 + element SR | 34.857 | 29.587 | +5.270 ms / +17.8% |
| BM32 + group-shared SR | 32.787 | 29.980 | +2.807 ms / +9.4% |

结论：

1. `BLOCK_M=32` 加上 group-shared stochastic rounding 明显有效，MX8 decode latency 从 `34.857` 降到 `32.787 ms/step`，绝对下降 `2.070 ms/step`。
2. 按同轮 BF16 归一化看，MX8 overhead 从上一轮约 `17.8%` 降到本轮约 `9.4%`。
3. 这说明剩余开销里有相当一部分来自过细 program granularity 和逐元素 `tl.rand`，而不是纯粹来自 state dequant/update 算术。
4. 这轮只测了 latency。由于 group-shared SR 改变了 stochastic rounding 噪声相关性，下一步还需要补 PPL / KL / top1，确认没有可见精度退化。

## MX8 State Kernel：Round 5 Single-update Dynamic-scale Requant Latency 设计

### 目的

修复 Round 4 后仍然存在的两个实现问题，同时保留动态 scale 和当前 group-shared stochastic rounding：

1. 去掉 MX8 tiled kernel 中的重复 state update。之前先用整块 `state` 计算输出 `y`，后面为了 requant 又按 16-value MX block 重新 load/dequant/update 一次。新实现改成按 16-value MX block 循环，一次完成 `dequant -> update -> y 累加 -> dynamic scale -> requant`。
2. 去掉 pair quantization 的重复计算。之前每个 2-lane pair 循环里都会对整个 16-lane group 计算 `q_abs/floor/SR/sign`，只 store 当前 pair。新实现将 16-lane group reshape 成 8 个 pair，一次性完成全部 pair 的 micro scale 和 quant。

### 保留不变的部分

- `BLOCK_M=32`
- dynamic shared/micro scale 每步刷新
- group-shared stochastic rounding
- context length、batch size、decode steps 和 Round 4 保持一致

### 对比组

- `bf16_full`：不启用 state quantization。
- `state_mx8_single_update`：启用 MX8 state quantization，单次 update 同时服务输出和 requant。

### 参数

- 模型：Nemotron-H-8B
- 模型路径：`/scratch2/wl730/models/nemotron-h-8b`
- 代码路径：`/scratch2/wl730/hybrid_codesign/han/chunk_update`
- context length：`2048`
- batch size：`8`
- decode steps：`128`
- warmup steps：`4`
- 数据：WikiText test
- 输出文件：`results/latency/data/nemotron_8b_mx8_single_update_ctx2048_latency.csv`

### 运行命令

```bash
python scripts/latency/run_nemotron_8b_latency.py \
  --sequence-length 2048 \
  --batch-sizes 8 \
  --decode-steps 128 \
  --warmup-steps 4 \
  --quantization-modes none,state_mx8 \
  --output results/latency/data/nemotron_8b_mx8_single_update_ctx2048_latency.csv
```

### 判断标准

- 主要对比 Round 4：MX8 `32.787 ms/step`，同轮 BF16 `29.980 ms/step`。
- 如果本轮 MX8 明显低于 `32.787 ms/step`，说明重复 update 和 pair-loop 冗余是主要剩余瓶颈。
- 如果下降有限，则剩余 overhead 主要来自 dynamic scale 的 `amax/log2/ceil/exp2`、metadata 读写、以及 SR 本身。

### 结果

输出文件：

- `results/latency/data/nemotron_8b_mx8_single_update_ctx2048_latency.csv`

| mode | ctx | batch | decode steps | decode ms/step | decode ms/token | tokens/s | p50 step ms | p90 step ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 2048 | 8 | 128 | 29.897 | 3.737 | 267.583 | 29.867 | 30.007 |
| mx8_single_update | 2048 | 8 | 128 | 29.756 | 3.720 | 268.849 | 29.728 | 29.907 |

和同轮 BF16 相比：

- MX8 single-update latency 差值：`-0.141 ms/step`
- 相对差值：`-0.47%`
- 这个负 overhead 在单次测试中属于测量波动范围，合理结论是：`state_mx8` decode latency 已经和 BF16 基本持平。

和前两轮实现相比：

| version | MX8 decode ms/step | same-run BF16 ms/step | MX8 overhead vs BF16 |
|---|---:|---:|---:|
| BM16 + element SR | 34.857 | 29.587 | +5.270 ms / +17.8% |
| BM32 + group-shared SR | 32.787 | 29.980 | +2.807 ms / +9.4% |
| BM32 + group-shared SR + single-update requant | 29.756 | 29.897 | about parity |

结论：

1. 重复 state update 和 pair-loop 冗余确实是主要剩余瓶颈。去掉之后，MX8 state quantization 的 latency 从 `32.787` 降到 `29.756 ms/step`，绝对下降 `3.031 ms/step`。
2. 相比 Round 3 的 BM16/逐元素 SR 版本，MX8 latency 从 `34.857` 降到 `29.756 ms/step`，累计下降 `5.101 ms/step`。
3. 当前实现仍然保留 dynamic scale 和 group-shared stochastic rounding，因此这轮 latency 改善不是靠牺牲 scale 更新或关闭 SR 得到的。
4. 由于输出 `y` 的 reduction 顺序从一次 128-lane reduction 改成 8 个 16-lane block 累加，理论上会有极小的浮点顺序差异。下一步如果要正式写论文或表格，需要补一次 PPL / KL / top1，确认质量与 Round 4/BF16 对齐。

## Quamba Compat：Round 2.1 补齐 CUTLASS 并验证 Quamba Weight Kernel 可用性

### 背景

用户希望把 Nemotron-H-8B 的 Mamba layer weight 量化成 Quamba 需要的形式，然后对 Quamba 代码做尽量小的修改，直接作为当前 hybrid 架构的 Mamba kernel 使用。

Round 1 的结构审计已经确认 Nemotron-H-8B 的 Mamba2 mixer 与 Quamba2 的核心张量拆分兼容：

- Mamba layer 数量：24
- `in_proj.weight`：`[18560, 4096]`
- `conv1d` 通道数：`10240`
- `num_heads=128`
- `head_dim=64`
- `ssm_state_size=128`
- `n_groups=8`
- `d_ssm=8192`
- `d_mlp=0`
- `conv_dim = 8192 + 2 * 8 * 128 = 10240`
- `in_proj` split `[gate, hidden_states_B_C, dt]` 与 Quamba2 的 `[z, xBC, dt]` 在 `d_mlp=0` 时一致。

Round 2 的 Quamba import/build 探测显示当前服务器环境已有：

- `torch 2.8.0+cu128`
- `triton`
- `causal_conv1d`
- `mamba_ssm`
- `ninja`
- `nvcc 12.4`

但 Quamba 自定义扩展尚未全部编译，其中 `quant_linear_cuda` 失败，原因是 Quamba 的 `3rdparty/cutlass` 子模块为空，缺少 `cute/tensor.hpp` 和 CUTLASS 头文件。

### 目标

1. 只在项目目录 `Layer_Quant/third_party/Quamba_fresh` / 服务器对应项目目录下补齐 CUTLASS 头文件，不修改服务器全局环境。
2. 重新编译 Quamba extensions，优先保证以下模块可 import：
   - `quant_linear_cuda`
   - `quamba2_conv1d_cuda`
   - `quant_causal_conv1d_cuda`
   - `quant_sscan_cuda`
   - `rms_norm_cuda`
3. 编译成功后，实现一个最小 Nemotron-H Mamba2 -> Quamba2 adapter：
   - 把 `n_groups` 映射成 Quamba 的 `ngroups`
   - 把 `ssm_state_size` 映射成 `d_state`
   - 把 `intermediate_size` 映射成 `d_ssm`
   - 把 `head_dim` 映射成 `headdim`
   - 把 `num_heads` 映射成 `nheads`
   - 保留 Nemotron 原始 forward/step 的 cache API，不先改 attention 层。
4. 第一阶段先跑单层 smoke test：
   - 只替换一个 Mamba layer 的 mixer
   - 输入同一批 hidden states
   - 对比 BF16 mixer 输出与 Quamba mixer 输出的 shape、max error、mean error
   - 验证 decode step 的 conv/state cache shape 不崩。

### 操作计划

1. 在服务器项目目录内 sparse clone CUTLASS，只拉 `include/`：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/cutlass.git 3rdparty/cutlass_src
git -C 3rdparty/cutlass_src sparse-checkout set include
mkdir -p 3rdparty/cutlass/build/install
ln -sfn ../../cutlass_src/include 3rdparty/cutlass/build/install/include
```

2. 使用服务器项目的 `.venv` 编译 Quamba，不安装到全局环境：

```bash
cd third_party/Quamba_fresh
CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8 ../../.venv/bin/python setup.py build_ext --inplace
```

3. 编译后运行 import probe：

```bash
python scripts/quamba_compat/probe_quamba_imports.py
```

4. 如果 `quant_linear_cuda` 成功，再开始 adapter；如果仍失败，记录失败的编译目标和缺失依赖，不进入模型替换实验。

### 判断标准

- 如果 Quamba extensions 可编译：说明“直接使用 Quamba weight kernel”在工程上可行，可以继续做 adapter 和单层/全模型实验。
- 如果 `quant_linear_cuda` 仍因 CUTLASS/CUDA ABI 失败：说明当前环境下无法直接使用 Quamba 的 weight GEMM，需要退回到“借鉴 Quamba state update/conv 思路，但 linear kernel 自研或保持 BF16”的路线。

### 修正：不要继续使用 sparse include / 手动补头文件

进一步检查 Quamba README 和 `build_cutlass.sh` 后，确认正确安装方式是：

1. `git clone --recurse-submodules` 或 `git submodule update --init --recursive`
2. `bash build_cutlass.sh`
3. `pip install .`

也就是说，`3rdparty/cutlass/build/install/include` 应该由 CUTLASS 的 CMake install 产生，而不是手动把 `include/` symlink 进去。之前 sparse checkout 只包含 `include/`，会缺少 `cutlass/util/host_tensor.h` 等 util headers；这不是 Quamba kernel 本身的问题，而是源码拷贝方式不完整。

后续操作改为在项目目录中新建官方完整 clone：

- 目录：`third_party/Quamba_official`
- clone 方式：HTTPS + submodule；如遇到 `.gitmodules` 中 `git@github.com:` 的子模块 URL，则使用 git URL rewrite 转成 HTTPS。
- 构建方式：直接执行官方 `build_cutlass.sh`，然后使用项目 `.venv/bin/python -m pip install -e third_party/Quamba_official` 或 `setup.py build_ext --inplace`。
- 不再修改 Quamba `setup.py` 的 include path。

### Round 2.2：Quamba 官方构建路径验证

已经按官方路径在服务器项目目录完成：

1. `third_party/Quamba_official` 使用 `--recurse-submodules` clone，子模块完整拉取。
2. 执行 Quamba 官方 `build_cutlass.sh`，由 CUTLASS CMake install 生成 `3rdparty/cutlass/build/install/include`。
3. 使用项目本地 `.venv` 运行 `setup.py build_ext --inplace`，Quamba CUDA 扩展编译成功。
4. 使用项目本地 `.venv` 安装 `3rdparty/fast-hadamard-transform`，不修改服务器全局环境。

底层扩展 import 结果：

- `quant_linear_cuda`: OK
- `quamba2_conv1d_cuda`: OK
- `quant_causal_conv1d_cuda`: OK
- `quant_sscan_cuda`: OK
- `rms_norm_cuda`: OK
- `fast_hadamard_transform_cuda`: OK

Python 层 `quamba.qMamba2` 当前卡在官方 requirements 中的 `scikit-learn` 缺失。Quamba README 要求 `pip install -r requirements.txt`，其中 `scikit-learn==1.6.1` 和 `scipy==1.15.2` 用于 Quamba2 head grouping。为了不把当前项目 `.venv` 的 PyTorch 2.8 降级到 Quamba requirements 中的 PyTorch 2.4，下一步只在项目 `.venv` 内补齐这两个 Python 依赖：

```bash
cd /scratch2/wl730/hybrid_codesign/han/chunk_update/third_party/Quamba_official
../../.venv/bin/python -m pip install scikit-learn==1.6.1 scipy==1.15.2
```

判断标准：

- 如果 `from quamba.qMamba2 import W4A16QMamba2, W4A8QMamba2, W8A8QMamba2` 成功，则进入 Nemotron-H Mamba2 -> Quamba2 adapter feasibility check。
- 如果仍有缺失依赖，继续按官方 requirements 的必要最小集合补齐，不修改 Quamba 源码。

### Round 2.3：Nemotron-H Mamba2 到 Quamba2 的最小 adapter smoke test

目的：确认不是靠修改 Quamba 源码，而是在 Nemotron-H mixer 外层做属性映射后，Quamba 的 `from_fp16` 可以吃下当前模型的 Mamba layer weight。

先测试 `W4A16QMamba2`：

- 原因：`W4A16QMamba2.from_fp16` 只需要 Mamba2 layer 的权重和结构属性，不需要 activation/state calibration scales。
- 测试层：第 0 个 Mamba layer。
- 输入：随机 hidden states，`batch=1, seq_len=1, hidden_size=4096`，dtype 使用模型权重 dtype。
- 对比：
  - 原 Nemotron mixer 输出 shape
  - Quamba W4A16 mixer 输出 shape
  - `max_abs_error`
  - `mean_abs_error`
- 只做单层 smoke，不替换全模型，不跑 PPL。

adapter 需要映射的字段：

- `d_model <- hidden_size`
- `d_state <- ssm_state_size`
- `d_conv <- conv_kernel_size`
- `expand <- intermediate_size / hidden_size`
- `headdim <- head_dim`
- `d_ssm <- intermediate_size`
- `ngroups <- n_groups`
- `dt_limit <- time_step_limit`
- `chunk_size <- chunk_size`
- `in_proj/conv1d/A_log/D/dt_bias/norm/out_proj` 直接引用 Nemotron 原模块

判断标准：

- 如果 `W4A16QMamba2.from_fp16(adapter)` 能构造，并且 forward 不崩，说明“把 Nemotron Mamba layer 的 weight 转成 Quamba 需要的格式”工程上可行。
- 如果 forward error 很大，这是 W4 权重量化误差/Quamba kernel路径差异，不代表 adapter 不可行；下一步再决定是否需要 W8A8 或校准。
- 如果构造失败，记录缺失字段或 shape 不兼容点，不修改 Quamba 源码。

### Round 2.4：Quamba2 W8A8 单层 scale/forward smoke test

目的：从 W4A16 的“权重量化能构造”进一步推进到 Quamba2 真正的 int8 conv + quantized chunk scan 路径。

实验方式：

1. 仍然只测试第 0 个 Mamba layer。
2. 使用随机校准输入 `batch=1, seq_len=16, hidden=4096`，通过原 Nemotron mixer 手动复现一遍：
   - `in_proj`
   - split `gate/xBC/dt`
   - `causal_conv1d_fn`
   - split `x/B/C`
   - `mamba_chunk_scan_combined(return_final_states=True)`
   - `norm`
3. 从这些中间张量提取 per-tensor int8 scale，构造 Quamba `W8A8QMamba2.from_fp16` 所需的 `act_scales` 字典。
4. 为了先降低 Hadamard scale 的不确定性，本轮设置 `use_had_transform=False`。
5. 运行单 token forward，比较 BF16/FP16 原 mixer 与 W8A8 Quamba mixer 的输出 shape 和误差。

注意：

- 这不是正式校准，只是工程 smoke test。正式精度实验需要使用真实 prompt/dataset 统计 scale。
- 如果 W8A8 forward 不通过，优先定位为：
  - scale shape 不符合 Quamba kernel 假设；
  - `W8A8B8O8LinearParallel` 输入/输出 dtype 与 Nemotron hidden dtype 不匹配；
  - Quamba `QRMSNormGated` 与 Nemotron `MambaRMSNormGated` 的参数接口差异。

### Round 2.4 修正：不能把 W8A8 mixer 单独接到 FP16 block 上

W8A8 smoke test 暴露出一个正确的问题：Quamba 的 W8A8 mixer 内部 `conv1d` kernel 期望 `xBC` 是 int8。只调用 `W8A8QMamba2.from_fp16` 会量化 mixer weight，但如果上一层 block norm 仍输出 FP16，那么 `in_proj -> conv1d` 的 activation 数据流不是 Quamba 论文里的 W8A8 数据流。

因此后续不再跑“单独替换 mixer”的 W8A8 实验。正确路径改为 block 级量化：

1. 对每个 Mamba block 的 `norm` 做 Quamba `QRMSNorm` 量化，使其输出 int8，输出 scale 使用该层 `in_proj:input` calibration scale。
2. 对该 block 的 `mixer` 使用 `W8A8QMamba2.from_fp16` 或 `W4A8QMamba2.from_fp16`，用 calibration 得到的 activation scales。
3. 用一个 wrapper 适配 Nemotron 的 `mixer(hidden_states, cache_params, cache_position, attention_mask)` 接口：
   - prefill 时用 Quamba 自己的 int8 conv/state cache 初始化；
   - decode 时调用 Quamba 的 `step`，而不是 Nemotron 原始 `selective_state_update`。
4. Attention/MLP block 暂时保持 BF16，Mamba block 输出仍是 FP16，可继续参与 residual add。

这个修正不是给 Quamba 打补丁，而是按 Quamba 的模型量化边界使用它：`quantized norm -> quantized mixer -> fp16 residual path`。

### Round 2.5：实现 Nemotron-H Mamba block 级 Quamba W8A8 量化 smoke test

目标：验证当前 hybrid 模型中至少一个 Mamba block 可以按 Quamba 正确方式量化和运行。

实验配置：

- 模型：Nemotron-H-8B
- 目标层：第 0 个 Mamba block
- 量化方式：Quamba W8A8
- 校准：使用少量真实 token 或随机 fallback，统计该 Mamba layer 的：
  - `in_proj:input`
  - `z_act`
  - `x/B/C conv_in`
  - `x/B/C conv_out`
  - `dt_act`
  - `ssm_state_act`
  - `out_proj:input`
- 替换：
  - `block.norm -> QRMSNorm`
  - `block.mixer -> NemotronQuambaW8A8MixerWrapper(W8A8QMamba2)`
- 验证：
  - prefill forward 不崩；
  - decode step 不崩；
  - 输出 shape 正确；
  - 记录与原 block 的单层输出误差。

实验结果：

- 脚本：`scripts/quamba_compat/probe_nemotron_quamba_block_w8a8.py`
- 结果：`results/quamba_compat/round3_block_w8a8_smoke/result.json`
- 目标层：第 0 个 Mamba layer
- 校准输入：同一个真实 prompt，截断/补齐到 `seq_len=32`
- 替换方式：
  - `block.norm -> QRMSNorm`，输出 scale 使用该层 `in_proj:input` calibration scale
  - `block.mixer -> NemotronQuambaW8A8MixerWrapper(W8A8QMamba2)`
  - 没有修改 Quamba 官方 CUDA/Python 源码
- 运行结果：
  - prefill shape：`[1, 32, 4096]`
  - decode shape：两步均为 `[1, 1, 4096]`
  - 与原 FP16 block 对比：`max_abs_error=0.3645`，`mean_abs_error=0.00550`
  - Quamba conv weight dtype：`torch.int8`
  - decode cache dtype：conv state 为 `torch.int8`，SSM state 目前为 `torch.float16`

结论：

- 用户指出的问题是对的：W8A8 不能只量化 mixer weight 后直接接到原 Nemotron FP16 block 上。Quamba 的 W8A8 kernel 需要 block 边界处已经量化的 int8 activation，否则会在 conv/linear kernel 处出现 dtype/scale 错误。
- 现在 block-level 路径已经跑通，说明“把 Nemotron-H 的 Mamba block 按 Quamba 论文边界量化后接入”工程上可行。
- 当前还不是最终论文版实现：
  - 本轮 scale 是单 prompt 的 per-tensor smoke calibration，不是 Quamba 的正式 calibration pipeline。
  - 关闭了 Hadamard transform。
  - 还没有做 Quamba2 的 grouped/reorder scale。
  - SSM state cache 仍是 fp16，后续如果目标是复现 Quamba 的 latency 优势，需要进一步确认官方路径中 state cache 的 dtype 和 update kernel 是否真正走 int8 state。

### Round 2.6：全 Mamba 层 Quamba-aligned W8A8 替换实现

目标：把 Round 2.5 的单层 smoke 升级为尽量和 Quamba 官方路径一致的完整实现，替换 Nemotron-H-8B 中全部 Mamba 层；做不到完全一致的地方必须显式记录，不能用简单 per-tensor smoke 替代。

实现原则：

- 不修改 Quamba 官方源码。
- 不使用单 prompt `amax/127` 作为正式校准。
- 对所有 Mamba layer 做同一套流程：
  1. 用真实 token calibration 样本跑原始模型；
  2. 通过 hook 收集每个 Mamba layer 的 Quamba2 activation observer；
  3. 使用 Quamba 官方 observer 的 `get_quantization_parameters()` 生成 scale；
  4. 使用 Quamba 官方 `W8A8QMamba2.from_fp16(..., use_had_transform=True)` 构造量化 mixer；
  5. 使用 Quamba 官方 `QRMSNorm.from_fp16` 替换 block 输入 norm；
  6. 用最小 wrapper 仅适配 Nemotron `cache_params/cache_position` 到 Quamba `inference_params`。

和 Quamba 官方纯 Mamba pipeline 的预期差异：

- 当前模型是 hybrid 架构，attention layer 不应该被 Quamba 替换；只替换 Mamba layers。
- Nemotron block/mixer forward signature 与 Quamba/Mamba2 不同，需要 wrapper 做接口适配；这是必要适配，不改变 Quamba kernel。
- Quamba 官方 `run_quamba2_calibration` 依赖其 `Mamba2Simple` 和纯 Mamba block 调用形式；本轮会复用其 observer/quantized module，但 calibration hook 需要按 Nemotron-H 的 block/mixer 接口重写。
- 如果 grouped/reorder scale 在当前 Nemotron-H 上无法从官方工具直接得到，会停止并记录为“不完全对齐”，不能退回到单 scale 冒充。

实验配置：

- 模型：`/scratch2/wl730/models/nemotron-h-8b`
- Mamba layers：全部 24 层
- calibration：
  - 默认 `num_calib_samples=8`
  - 默认 `seq_len=128`
  - 使用本地内置 prompt 列表，避免联网下载数据集
- 测试：
  - 替换后跑一次 prefill forward；
  - 使用 `generate(max_new_tokens=4)` 测 decode；
  - 记录替换层数量、每层关键 scale shape、模块 dtype、是否启用 Hadamard、是否使用 grouped scale。

实现结果：

- 脚本：`scripts/quamba_compat/run_nemotron_quamba_full_w8a8.py`
- 结果：`results/quamba_compat/round4_full_w8a8/result.json`
- 成功跑通严格 grouped + Hadamard 路径：
  - `use_had_transform=true`
  - `use_group_heads=true`
  - `grouped_available=true`
- 全部 24 个 Mamba layer 已替换为 `W8A8QMamba2`：
  - `[0, 2, 4, 6, 9, 11, 13, 15, 17, 20, 22, 24, 26, 28, 31, 33, 35, 37, 39, 42, 44, 46, 48, 50]`
- 所有替换层均满足：
  - `block.norm -> QRMSNorm`
  - `block.mixer -> W8A8QMamba2`
  - `in_proj.weight dtype = torch.int8`
  - `conv1d.weight dtype = torch.int8`
  - `out_proj.weight dtype = torch.int8`
  - `x_conv_out:input` 是 grouped list scale，`ssd_groups=8`
  - `ssm_state_act:input` 是 grouped list scale，`ssd_groups=8`
- smoke test：
  - prefill logits shape：`[1, 128, 131072]`
  - 手动 decode logits shape：`[1, 1, 131072]`
  - `generate(max_new_tokens=4)` 能返回文本

中间遇到并修复的问题：

- Quamba 官方 `group_wise_sort_indices` 内部默认 index 在 CPU，因此 reorder stats 必须保持 CPU 输入；这不是算法改动，只是按官方函数假设调用。
- Nemotron-H 不会在 `model(..., use_cache=True)` 时自动初始化 `HybridMambaAttentionDynamicCache`。为了统计 Quamba `ssm_state_act`，必须显式构造并传入 cache，否则 state observer 不会触发。

仍然和 Quamba 官方纯 Mamba 模型不同的地方：

- 当前只量化 hybrid 模型中的 Mamba blocks；attention、MLP、embedding、final norm、lm_head 保持原 dtype。这是 hybrid 架构选择，不是 Quamba 纯 Mamba 全模型替换。
- 由于 Nemotron-H 的接口是 `cache_params/cache_position`，而 Quamba 使用 `inference_params`，脚本对 `Mamba2Simple` 和 `W8A8QMamba2` 的 `forward` 做了最小签名适配；没有改 Quamba kernel 或量化模块。
- calibration 数据用本地 prompt 列表，避免联网下载 Quamba 默认的 `monology/pile-uncopyrighted`。因此这次只能证明完整工程路径跑通，不能代表最终 PPL。

### Round 2.7：Quamba-aligned W8A8 全 Mamba 替换版 latency / PPL 对比

目标：在已经跑通的 Quamba-aligned W8A8 Mamba 全层替换实现上，正式测一次 decode latency 和 decode-slice PPL，并与原始 BF16 baseline 对比。

实验原则：

- 不使用 `generate()` 做 latency，因为当前实现通过 wrapper 把 Nemotron `cache_params/cache_position` 适配到 Quamba `inference_params`，`generate()` 不一定稳定传入显式 hybrid cache。
- 使用显式 cache 的手写 decode loop：
  1. prefill 固定 context；
  2. warmup 若干 decode step；
  3. 用 CUDA event 计时后续 decode step。
- PPL 也用同一条 decode path：
  - prefill context；
  - teacher-forcing decode 后续 token；
  - 对 decode step 的 next-token logits 计算 CE/PPL。
- Quamba W8A8 版本只替换 Mamba blocks；attention、MLP、embedding、final norm、lm_head 保持原 dtype/原 kernel。这是 hybrid 架构设定，不是 Quamba 纯 Mamba 全模型。

实验配置：

- 模型：`/scratch2/wl730/models/nemotron-h-8b`
- 对比组：
  - `baseline_bf16`：原始 Nemotron-H-8B，BF16；
  - `quamba_w8a8_mamba`：全部 24 个 Mamba layer 替换成 Quamba-aligned `QRMSNorm + W8A8QMamba2`，残差/attention/MLP 保持原路径。
- Quamba calibration：
  - `num_calib_samples=8`
  - `calib_seq_len=128`
  - 启用 Quamba grouped reorder scale；
  - 启用 Hadamard transform；
  - 使用本地内置 calibration prompt，避免联网下载数据集。
- latency：
  - dataset：WikiText test
  - context length：`2048`
  - batch size：`8`
  - warmup decode steps：`8`
  - timed decode steps：`64`
- quality / PPL：
  - dataset：同一份 WikiText test
  - context length：`2048`
  - batch size：`2`
  - warmup decode steps：`0`
  - measured decode steps：`64`
  - 额外记录 Quamba logits 与 baseline logits 的 KL 和 top1 match，辅助判断 PPL 变化来自何处。

输出目录：

- `results/quamba_compat/round5_latency_ppl/`
- `latency.csv`：每组 decode latency / throughput / peak memory；
- `quality.csv`：每组 decode CE/PPL，以及相对 BF16 baseline 的 KL/top1；
- `summary.json`：完整配置和结果。

判断标准：

- 如果 W8A8 Mamba 版本 PPL 接近 BF16 baseline 且 latency 下降，说明 Quamba-style 全 Mamba block 替换在 hybrid 模型中有继续优化价值。
- 如果 PPL 接近但 latency 不降，说明当前瓶颈可能转移到 attention/MLP 或 wrapper/cache 初始化，也可能是 Quamba kernel 对当前 batch/context 不占优。
- 如果 latency 下降但 PPL 明显变差，需要回到 calibration 数据、grouped scale、state scale 覆盖和 hybrid attention/Mamba 交互误差分析。

实验结果：

- 脚本：`scripts/quamba_compat/eval_nemotron_quamba_w8a8_latency_ppl.py`
- 输出目录：`results/quamba_compat/round5_latency_ppl/`
- latency 文件：`latency.csv`
- quality 文件：`quality.csv`
- prepare 文件：`prepare.csv`

latency 对比，`context=2048, batch=8, decode_steps=64`：

| mode | decode ms/step | tokens/s | prefill s | peak memory GiB |
|---|---:|---:|---:|---:|
| baseline_bf16 | 31.075 | 257.44 | 7.616 | 28.237 |
| quamba_w8a8_mamba | 22.735 | 351.88 | 28.259 | 25.794 |

quality 对比，`context=2048, batch=2, decode_steps=64`：

| mode | CE | PPL | KL vs baseline | top1 match |
|---|---:|---:|---:|---:|
| baseline_bf16 | 1.678 | 5.357 | 0.000 | 1.000 |
| quamba_w8a8_mamba | 10.150 | 25592.579 | 8.597 | 0.039 |

派生结论：

- Quamba W8A8 Mamba decode latency 相比 BF16 baseline 有明显下降：
  - `31.075 -> 22.735 ms/step`
  - speedup：`1.367x`
  - peak memory 下降约 `2.44 GiB`
- 但质量完全不可用：
  - PPL 从 `5.36` 升到 `25592.58`
  - top1 match 只有 `3.9%`
  - KL 达到 `8.60`
- prefill latency 明显变慢：
  - `7.616 -> 28.259 s`
  - 说明当前 Quamba W8A8 path 主要改善 decode step，不改善长 prefill；如果研究目标包含 long-context prefill，这条路径还需要单独优化。

判断：

当前实现证明了“Quamba-style W8A8 Mamba layer 替换可以显著降低 decode step latency”，但也证明了当前量化/适配版本还不能作为有效模型结果。退化幅度太大，不像正常 W8A8 量化误差，更像 calibration 覆盖不足或 block/interface 边界仍有不完全对齐。

需要重点检查：

1. calibration 数据。
   - 本轮只用 8 条短 prompt，长度 128，和 Quamba 官方大量 calibration 数据差距很大。
   - Mamba state / grouped scale 对 activation outlier 很敏感，短 prompt 很可能覆盖不住 WikiText decode 分布。
2. block norm + in_proj fuse 边界。
   - 当前按 Quamba `fuse_ln_linear -> QRMSNorm -> W8A8QMamba2` 路径做。
   - 但 Nemotron-H hybrid block 的 norm/residual 包装和 Quamba 官方纯 Mamba block 不完全一样，需要做逐层输出误差定位，判断误差从哪一层开始爆。
3. cache 适配。
   - decode latency 使用显式 cache，Quamba wrapper 通过 `seqlen_offset=1` 进入 `step()`。
   - 仍需要验证每个 Mamba layer 的 Quamba internal state 是否和 prefill 后的原始 state 数值对应，而不是只验证 shape 能跑通。
4. hybrid 交互。
   - 只量化 Mamba blocks，attention/MLP 保持原路径。attention 输入来自前面多个量化 Mamba 层，误差会被后续 attention/MLP 放大。

下一步不建议直接把这个结果用于论文对比。更合理的是先做 layer-wise error localization：

- 单独替换第 0 个 Mamba layer，测 full-model decode PPL/KL/top1；
- 逐步增加替换层数或按 block index 替换，找到质量从可接受到崩溃的临界层；
- 对临界层记录 block input/output 的 cosine、relative error、activation clipping ratio 和 scale range；
- 如果单层替换已经明显崩，优先修 calibration / norm fuse / cache 对齐；如果单层正常但多层崩，则研究 hybrid 架构下误差传播和 selective high-precision layers。

### Round 2.8：Quamba W8A8 质量崩溃原因定位设计

Round 2.7 发现 Quamba W8A8 Mamba 全层替换有明显 decode latency 收益，但质量完全崩溃。这个退化幅度远大于正常 W8A8 误差，因此本轮目标不是继续调参数，而是定位“从哪一步开始偏离 baseline”。

核心假设：

1. `Mamba2Simple` adapter 或 norm/in_proj fuse 边界有语义错误。
   - 如果只做 `fuse_ln_linear + Mamba2Simple`，不量化，full-model logits 已经明显偏离 baseline，则质量崩溃不是 W8A8 的问题。
2. `use_had_transform=True` 在 hybrid 模型中可能破坏 residual-stream 坐标系。
   - Quamba 官方纯 Mamba pipeline 会对 embedding / final norm / lm_head 等做配套 Hadamard 变换。
   - 当前只替换 Mamba blocks，attention / MLP / embedding / lm_head 保持原坐标系；如果 Mamba block 内部启用 Hadamard 但模型其余部分没有配套变换，可能出现坐标系不一致。
3. Quamba cache adapter 可能只保证 shape 跑通，但 prefill 后 internal `conv_state/ssm_state` 与 Nemotron 原始 cache 不对齐。
   - 如果 prefill q_len>1 输出正常，但 decode step 输出崩，优先检查 cache adapter。
4. calibration 数据过少可能导致 W8A8 scale 不覆盖 decode 分布。
   - 如果 FP16 `Mamba2Simple` 路径正常，但 W8A8 路径崩，再进一步检查 activation clipping 和 scale 覆盖。

诊断实验：

- 模型：`/scratch2/wl730/models/nemotron-h-8b`
- 数据：同 Round 2.7 的 WikiText decode slice，`context=2048, batch=2, decode_steps=32`
- baseline：
  - 原始 BF16/FP16 Nemotron-H，显式 cache decode。
- 诊断组：
  1. `simple_no_had_all`：全部 Mamba blocks 替换成 Quamba `Mamba2Simple`，不量化，`use_had_transform=False`。
  2. `simple_had_all`：全部 Mamba blocks 替换成 Quamba `Mamba2Simple`，不量化，`use_had_transform=True`。
  3. `simple_no_had_layer0`：只替换第 0 个 Mamba block，判断单层 adapter 是否正确。
  4. `w8a8_no_had_layer0`：只量化第 0 个 Mamba block，关闭 Hadamard，判断 W8A8 单层质量。

记录指标：

- CE / PPL；
- KL vs baseline；
- top1 match vs baseline；
- prefill 最后一 token logits KL；
- 第 1 个 decode step logits KL；
- 如果 `simple_no_had_all` 已经崩，停止继续 W8A8 大实验，先修 adapter/fuse。
- 如果 `simple_no_had_all` 正常但 `simple_had_all` 崩，说明 Hadamard 在 hybrid partial replacement 下不能直接启用。
- 如果 simple 正常、W8A8 单层崩，优先修 calibration/scale。

实验结果：

- 脚本：`scripts/quamba_compat/diagnose_quamba_quality.py`
- 输出目录：`results/quamba_compat/round6_quality_diagnosis/`
- 结果文件：`quality_diagnosis.csv`
- 设置：`context=2048, batch=2, decode_steps=32`

| mode | CE | PPL | KL vs baseline | top1 match | decode ms/step |
|---|---:|---:|---:|---:|---:|
| baseline_fp16 | 1.909 | 6.746 | 0.000000 | 1.000 | 33.562 |
| simple_no_had_layer0 | 1.910 | 6.750 | 0.000012 | 1.000 | 29.858 |
| simple_no_had_all | 1.909 | 6.746 | 0.000014 | 1.000 | 26.214 |
| simple_had_all | 11.860 | 141463.262 | 9.377986 | 0.016 | 42.624 |
| w8a8_no_had_all | 3.764 | 43.106 | 1.829045 | 0.438 | 25.320 |

诊断结论：

1. `Mamba2Simple` adapter / norm fuse / cache adapter 本身不是质量崩溃主因。
   - `simple_no_had_all` 与 baseline 几乎完全一致：
     - PPL `6.746` vs `6.746`
     - KL `1.4e-5`
     - top1 match `100%`
   - 说明 `fuse_ln_linear + Mamba2Simple(use_had_transform=False)` 的函数语义和原 Nemotron-H Mamba path 对齐。
2. `use_had_transform=True` 是 Round 2.7 灾难性质量崩溃的直接主因。
   - `simple_had_all` 没有做任何 W8A8 量化，只打开 Quamba `HadLinear/Hadamard`，PPL 就从 `6.746` 炸到 `141463`。
   - 因此 Round 2.7 的 PPL `25592` 不能归因于 W8A8 本身；它首先是 Hadamard 在 hybrid partial replacement 下坐标系不一致。
3. Quamba 官方 Hadamard pipeline 不能直接套当前 hybrid 模型。
   - Quamba 官方 `configure_model(..., use_had_transform=True)` 会对纯 Mamba 模型的 embedding / final norm / lm_head 做配套 Hadamard 变换。
   - 当前实现只替换 Mamba blocks，attention / MLP / embedding / lm_head 仍在原坐标系里。
   - 这会让 Mamba block 的输入/输出坐标系和 hybrid 模型其余部分不一致。
4. 我尝试调用官方 `fuse_had_matrices(model)` 时发现它假设所有 blocks 都有 `mixer.in_proj`。
   - 在 Nemotron-H hybrid 模型里，MLP block 的 mixer 是 `NemotronHMLP`，只有 `up_proj/gate_proj/down_proj`，没有 `in_proj`。
   - 因此 Quamba 官方 Hadamard fuse 不能直接用于 hybrid 模型，必须写 hybrid-safe 版本，只处理 Mamba blocks，并且还要决定 attention/MLP 是否也要进入同一 Hadamard 坐标系。
5. 关闭 Hadamard 后，W8A8 全 Mamba 仍有明显质量损失，但不再灾难性崩溃。
   - `w8a8_no_had_all`：PPL `43.106`，KL `1.829`，top1 `43.8%`
   - 这部分才是 W8A8 量化 / calibration / scale 覆盖 / 多层误差累积的问题。

因此 Round 2.7 质量炸裂的原因分成两层：

- 主因：错误启用 Quamba Hadamard path。它要求模型整体进入 Hadamard 坐标系；当前 hybrid partial replacement 没有做 attention/MLP/embedding/lm_head 的配套变换。
- 次因：即使关闭 Hadamard，当前 W8A8 全 Mamba 替换仍然质量不够，说明 calibration 和 activation/state scale 还不足以支撑全部 24 个 Mamba blocks 同时量化。

后续修复顺序：

1. 立刻把 hybrid Quamba 实验默认改成 `use_had_transform=False`，不要在当前 partial replacement 下启用 Hadamard。
2. 做 `w8a8_no_had_layerwise`：
   - 单层量化；
   - prefix 层数递增量化；
   - 找出 PPL 从可接受到崩溃的临界层。
3. 扩大 calibration：
   - 从 8 条短 prompt 改成 WikiText/真实 calibration，`num_samples >= 128`，`seq_len >= 512`；
   - 记录 clipping ratio 和每层 scale range。
4. 如果还想用 Hadamard，必须设计 hybrid-safe Hadamard：
   - 要么只在 Mamba block 内部用 mathematically equivalent fused weights，保证 block input/output 仍回到原坐标系；
   - 要么把 attention/MLP/embedding/lm_head 全部一起变换，这工程量更大，也偏离“只替换 Mamba block”的设定。
### Round 2.14：新 calibration + local Hadamard 的 W8A8 PPL 测试

目标：

验证在使用新的 calibration 设置后，Quamba-compatible W8A8 Mamba 替换是否能在开启本地 local Hadamard 的情况下恢复 PPL。这里的 local Hadamard 不是 Quamba 官方要求全模型坐标系配套旋转的 Hadamard，而是只在 Mamba block 的 `out_proj` 前后做块内等价变换：在 `out_proj` 输入前插入 Hadamard，同时把逆变换融合进 `out_proj` 的输入维度，使 FP 路径保持等价，避免破坏 hybrid 模型中 attention/MLP 仍使用原 residual 坐标系这一事实。

实验设置：

- 工作目录：Ru-server 的 `chunk_update` 目录。
- 模型：`/scratch2/wl730/models/nemotron-h-8b`。
- 量化对象：只替换 Nemotron-H 中的 Mamba layers 为 Quamba-compatible W8A8；attention、MLP、embedding、final norm、lm head 保持原精度。
- calibration：使用新的 calibration 数据/配置；如果脚本参数化运行，优先使用 `--calib-source wikitext` 或已准备好的新 calibration source，而不是短 prompt smoke calibration。
- Hadamard：开启我们写的 local out Hadamard；不启用会改变全模型坐标系的 Quamba 官方全局 Hadamard 路径。
- 评测：先跑 PPL/quality，不优先测 latency。
- 建议参数：
  - `context_length=2048`
  - `quality_batch_size=2`
  - `quality_decode_steps=64`
  - `num_calib_samples=8` 或新 calibration 已指定的样本数
  - `calib_seq_len=2048`
  - modes：`baseline_bf16,quamba_w8a8_mamba`

需要记录：

- baseline BF16 PPL；
- W8A8 + local Hadamard PPL；
- KL vs baseline；
- top1 match vs baseline；
- calibration source、样本数、长度；
- 是否完成 grouped reorder 和所有 Mamba layer 替换。

判断标准：

- 如果 PPL 仍明显高于 baseline，需要继续定位是不是 offline scale 覆盖不足、某些 layer clipping 严重、或者 local Hadamard 只缓解 out_proj 而无法修复前面 `in_proj/conv/ssm_state` 的 activation 量化误差。
- 如果 PPL 接近 baseline，再进入 latency 和 kernel 级优化测试。
