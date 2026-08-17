# PERF-ALI-010：ForcedAligner 的 torch.compile spike

验证给 PyTorch ForcedAligner 加编译优化能否降低 GPU 前向耗时。**只在实验进程内包装
`aligner.model.thinker`，不修改 `src/aligner.py`、不改生产默认路径、不触碰磁盘权重。**

## 为什么打这个靶

生产实测（目标 A10、MPS 在线、`ALIGNER_BATCH_SIZE=32`、并发 96）的时间去向：

```text
total          3437 ms
├─ asr          820 ms   24%
├─ aligner-queue 1306 ms  38%
└─ aligner     1147 ms   33%
```

Aligner 批内进一步拆解：

```text
model_call     780.82 ms   83.7%   ← 本实验目标
audio_decode   146.93 ms   15.8%   CPU 侧
result_build     ~5 ms      0.5%
```

主 ASR 原始能力约 1102.95 audio_s/s，是端到端 274 的四倍，不是瓶颈；模型层投入应全部对着
Aligner 的这 780 ms。

关键事实：**vLLM 侧默认已启用 inductor 编译与 `FULL_AND_PIECEWISE` CUDA graph，而 Aligner
是裸 PyTorch eager 执行，没有任何编译优化。** 同一张卡上两个模型的执行栈不对等。

## 官方实现结构

`Qwen3ForcedAligner.align()` 的 GPU 计算集中在一行：

```python
inputs = self.processor(text=..., audio=..., return_tensors="pt", padding=True)  # CPU
logits = self.model.thinker(**inputs).logits                                     # GPU
output_ids = logits.argmax(dim=-1)                                               # GPU
```

因此编译目标是 `model.thinker`。同时注意 `padding=True`：批内音频会被 padding 到最长，
这是另一个独立问题（批内长度分桶），不在本实验范围。

## 两段测量

| 阶段 | 范围 | 用途 |
|---|---|---|
| Phase A | `align()` 全程 | 口径与生产日志 `aligner_batch_model_call_ms` 一致，可直接对照 780.82 ms |
| Phase B | 仅 thinker 前向与 argmax | 排除 CPU 前后处理，反映编译能影响的上限 |

两段都用 `torch.cuda.synchronize()` 界定，避免异步执行导致计时失真。

## 环境

使用项目根统一环境，不新建 venv、不安装额外依赖：

```bash
python -c "import torch, importlib.metadata as m; print(torch.__version__, torch.version.cuda, m.version('qwen-asr'))"
```

脚本会强制校验 PyTorch 为 CUDA 12.x 构建——目标 A10 宿主驱动为 535，不支持 CUDA 13.0。

## 执行

需要独占 GPU，先停掉服务：

```bash
pkill -f 'src/gateway.py'
pkill -f 'vllm serve'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

文本必须与音频真实内容对应，可直接用同名参考文本：

```bash
python experiments/perf_ali_010/spike.py \
  --audio test_data/audio_16000_10s.wav \
  --text test_data/audio_16000_10s.txt --text-is-file \
  --language Chinese \
  --batch-sizes 1,8,32 \
  --output-json performance-results/PERF-ALI-010/compile-default.json
```

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `models/qwen3-forced-aligner-0.6b/pt` | 与生产同一目录，只读 |
| `--audio` | 必填 | 16 kHz 单声道，最长 32 秒 |
| `--language` | 必填 | 官方语言名称 |
| `--text` | 必填 | 参考文本；配 `--text-is-file` 时为路径 |
| `--batch-sizes` | `1,8,32` | 生产对照值为 32 |
| `--warmup` | `15` | 预热不足会让 eager 均值虚高，见“预热不足的教训” |
| `--iterations` | `30` | 正式执行次数 |
| `--dtype` | `bfloat16` | 与生产一致；`float16` 为独立单变量 |
| `--attention` | `auto` | 目标机 `auto` 实际选择 `sdpa` |
| `--compile-mode` | `default` | 可选 `reduce-overhead`、`max-autotune` 等 |
| `--compile-dynamic` | 自动 | 显式指定用于观察动态 shape 行为 |
| `--compile-fullgraph` | 关闭 | 先允许图断裂取得可用基线 |

## 正确性

编译只应改变执行方式，不应改变结果。脚本对编译前后逐单元比对文本与时间戳，默认容差 1 ms，
不一致即抛错。这一点比性能数字更重要——**收益再大，结果变了就是不可用。**

报告中的 `dynamo_graphs_before/after` 用于观察编译图数量。若同一 batch 下图数量持续增长，
说明发生了重编译，收益不可信。

## 已知风险

`qwen-asr==0.0.6` 是第三方封装，`torch.compile` 可能踩到不支持的算子而大量图断裂，此时收益
接近零。这属于预期结果之一，如实记录即可，不要为了通过而放宽 `fullgraph` 或改动官方代码。

文本长度可变会改变输入 shape，进而触发重编译。首轮固定同一文本与音频以取得上限值；真实混合
长度下的收益必须另立单变量验证，不能沿用首轮结论。

首次编译有秒级开销，已由 `--warmup` 排除在计时之外，但生产若启用需要考虑冷启动代价。

## 晋级条件

- 编译前后对齐单元文本与时间戳完全一致。
- batch=32 的 `align()` 平均耗时相对 eager 至少下降 15%。
- 无重编译迹象，无 NaN、OOM 或崩溃。
- 达标后才考虑改 `src/aligner.py`，并需补真实混合长度与三轮压测验证。

## 实测结果（2026-08-17，目标 A10，独占 GPU）

有效数据只有一组：`--batch-sizes 32 --warmup 15 --iterations 30`，报告
`performance-results/PERF-ALI-010/compile-b32-warm.json`。

| 阶段 | eager mean | compiled mean | 变化 |
|---|---:|---:|---:|
| `align()` 全程 | 926.41 ms | 868.02 ms | **+6.30%** |
| thinker 前向 | 670.05 ms | 559.62 ms | **+16.48%** |
| CPU 部分（反推） | 256.36 ms | 308.40 ms | — |

**判定：不晋级。** thinker 前向确实快了 16.48%，但生产口径的 `align()` 只降 6.30%，未达
15% 晋级线。时间戳一致性通过，无 NaN、OOM 或崩溃，因此这是收益不足而非实现失败。
不改 `src/aligner.py`。

### 预热不足的教训

首轮用默认 `--warmup 5`，得到的数字全部不可用：

| batch | align 首轮 | thinker 首轮 |
|---:|---:|---:|
| 1 | +42.09% | +49.37% |
| 8 | +15.89% | +19.18% |
| 32 | **+47.40%** | +15.63% |

batch=32 的 `align` 看似收益最大，实际是 eager 分母虚高：eager 的 min 836.70、p50 1547.16、
max 1939.73，极差 2.3 倍。用 `align - thinker` 反推纯 CPU 部分可以直接定位问题——batch=1 两次
均为 6.79 ms、batch=8 为 43.67/42.47 ms 都接近，唯独 batch=32 为 752.14/182.23 ms，差 570 ms。
编译只替换 GPU 前向，不可能让 CPU 前后处理差半秒，所以差异只能来自测量本身。
提到 `--warmup 15 --iterations 30` 后 align 收益从 +47.40% 落到真实的 +6.30%。

由此固化两条防御：

- `_stats()` 增加 `max_over_min` 极差比字段，稳定性是报告的一等数据而非事后人工核对；
- `_STABILITY_LIMIT = 1.2`，任一组超限即在 `measurement_unstable` 标注并打印警告，该组数据
  不得用于判定。同时把 `--warmup` 默认提到 15、`--iterations` 提到 30。

**通用结论：编译类实验必须先证明 eager 基线稳定，再谈收益。** 分母不稳时收益百分比无意义，
而且方向恰好是让收益偏大，容易误判为达标。

### 副产品发现一：CPU 特征提取是更大的靶点

batch=32 时 `align()` 的 926.41 ms 里约 **256 ms 在 CPU 上**（约 28%），主要是 processor 的
mel 特征提取。这部分此前完全没有被识别出来：生产日志的 `aligner_batch_audio_decode_ms`
（146.93 ms）只统计 WAV 解码，特征提取被算进了 `aligner_batch_model_call_ms`。

也就是说 780 ms 的“模型调用”里有约三分之一不在 GPU 上跑。相对本实验能拿到的 6.30%，
这个靶点空间更大，应另立实验。

### 副产品发现二：图断裂与重编译

`dynamo_graphs` 计数为 `-1→8`（batch=1 首次编译产生 8 处图断裂）、`8→12`（batch 从 1 变到 8
又新增 4 个图，即 batch 变化触发重编译）、`12→12`（batch=32 复用）。

`qwen-asr==0.0.6` 的 thinker 存在多处图断裂，这解释了收益为何有限。更重要的是生产影响：
动态微批的批大小在 1～32 之间浮动，每个新形状都会重编译。本实验未量化该代价，若将来重启
编译路线，必须先用 `--compile-dynamic` 单变量验证。

### 后续可选单变量

按一次只改一个变量的规则，尚未验证的方向：`--compile-mode reduce-overhead`（启用 CUDA graph
进一步压 launch 开销）、`--compile-dynamic`（减少重编译）、`--dtype float16`。这些都只影响
thinker 那 670 ms，而 CPU 的 256 ms 不受编译影响，因此 `align()` 的收益上限被 CPU 部分锁死。
