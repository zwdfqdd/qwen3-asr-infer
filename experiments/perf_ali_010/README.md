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
| `--warmup` | `5` | 编译需要足够预热，避免把编译耗时计入 |
| `--iterations` | `10` | 正式执行次数 |
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
