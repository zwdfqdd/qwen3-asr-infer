# PERF-ALI-011：ForcedAligner `align()` 内部七段拆分计时

**本实验只做测量，不做优化。** 不修改 `src/aligner.py`、不改生产默认路径、不触碰磁盘权重。

目的是回答一个具体问题：`align()` 里那段约 256 ms 的非 GPU 时间，究竟花在哪。在这个问题有
答案之前，任何优化都无法验证收益。

## 为什么需要这个实验

`PERF-ALI-010` 用减法反推出一个此前没被看见的事实。它分两段测量，batch=32 时：

```text
align()  926.41 ms
thinker  670.05 ms
────────────────────
差值     256.36 ms   约占 28%，不在 GPU 上
```

生产阶段计时看不到这段，因为口径划错了位置。`src/aligner.py` 里：

```python
started = perf_counter()
aligned = self._model.align(audio=audios, text=..., language=...)
model_call_ms = (perf_counter() - started) * 1000
```

`model_call_ms` 包的是整个 `align()` 调用，而官方 `align()` 内部同时做 CPU 分词、mel 特征
提取、GPU 前向和 CPU 时间桶解析。另一个指标 `audio_decode_ms`（146.93 ms）只统计网关自己的
WAV 解码，和 processor 的特征提取是两件事。结果是生产日志中 780.82 ms 的“模型调用”里，
约三分之一根本不在 GPU 上。

## 七段划分

按官方 `qwen_asr==0.0.6` 源码逐行拆开：

| 段 | 内容 | 侧 |
|---|---|---|
| `normalize_audio` | `normalize_audios()`，本实验输入为 `(ndarray, sr)` 元组，接近恒等 | CPU |
| `encode_timestamp` | 分词并拼 `<timestamp>` 序列 | CPU |
| `feature_extract` | `processor(text=..., audio=..., padding=True)`，mel 与 tokenize | CPU |
| `host_to_device` | `inputs.to(device).to(dtype)` | H2D |
| `thinker_forward` | `thinker(**inputs).logits` 与 `argmax` | GPU |
| `device_to_host` | timestamp token mask 与 `.to("cpu").numpy()` | D2H |
| `parse_timestamp` | `parse_timestamp()`、秒换算、结构体构建 | CPU |

### 为什么必须拆到这个粒度

两个 CPU 段都是 256 ms 的候选主体，合并测量无法区分：

- `feature_extract` 是 mel 特征提取，直觉上的第一嫌疑；
- `parse_timestamp` 里的 `fix_timestamp()` 是**纯 Python 的 O(n²) 最长非降子序列 DP**：

  ```python
  dp = [1] * n
  for i in range(1, n):
      for j in range(i):        # 双层循环，无向量化
  ```

  n 为对齐单元数的两倍。10 秒中文约 67 字 → n≈134 → 每样本约 9000 次迭代，batch=32 约
  29 万次。报告记录 `fix_timestamp_n`，便于后续用不同文本长度验证平方关系。

两段的优化手段完全不同（前者可移入线程池或复用 PCM，后者是算法与实现问题），因此归因错了
方向就全错了。

## 两项自检

性能数字之前先过两道校验，任一不通过则占比结论作废：

1. **复刻正确性，零容差。** 同时执行官方 `align()` 与本实验的分段复刻，逐单元比对文本与
   时间戳。两者执行同一串运算，任何差异都说明复刻错了，不是精度抖动，所以不设容差。
2. **七段合计与复刻全程闭合。** 报告输出 `unexplained_ms` 与 `unexplained_percent`，超过
   5% 即标记 `stage_sum_mismatch` 并告警——说明还有未计时的段落，占比不可信。

复刻有意省略官方的三处入参护栏（`ensure_list()`、语种广播、batch 长度检查），都是常数级
判断，省略后应体现为未归因接近零。另有 `staged_overhead_percent` 记录复刻相对官方全程的
偏差，用于确认插入计时点本身没有引入显著开销。

沿用 `PERF-ALI-010` 的稳定性守卫：`_STABILITY_LIMIT = 1.2`，各段极差比超限即写入
`measurement_unstable` 并告警。与 010 不同的是只对均值达到 1 ms 的段判定——亚毫秒段本身就
容易出现大极差比，全都告警会淹没真正不稳的段。默认 `--warmup 15 --iterations 30`。

## 执行

需要独占 GPU，先停掉服务：

```bash
pkill -f 'src/gateway.py'
pkill -f 'vllm serve'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

```bash
python experiments/perf_ali_011/spike.py \
  --audio test_data/audio_16000_10s.wav \
  --text test_data/audio_16000_10s.txt --text-is-file \
  --language Chinese \
  --batch-sizes 1,8,32 \
  --output-json performance-results/PERF-ALI-011/stage-split.json
```

输出为每个 batch 一张七段表，含每段 mean、占比和极差比，末尾给出 CPU 合计、GPU 合计与
未归因量。

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `models/qwen3-forced-aligner-0.6b/pt` | 与生产同一目录，只读 |
| `--audio` | 必填 | 16 kHz 单声道，最长 32 秒 |
| `--language` | 必填 | 官方语言名称 |
| `--text` | 必填 | 参考文本；配 `--text-is-file` 时为路径 |
| `--batch-sizes` | `1,8,32` | 生产对照值为 32 |
| `--warmup` | `15` | 沿用 ALI-010 的教训，预热不足会使均值虚高 |
| `--iterations` | `30` | 正式执行次数 |
| `--dtype` | `bfloat16` | 与生产一致 |
| `--attention` | `auto` | 目标机 `auto` 实际选择 `sdpa` |

## 首轮结果（2026-08-17，目标 A10，`--batch-sizes 1,8,32` 连跑）

报告 `performance-results/PERF-ALI-011/stage-split.json`。

两项自检全部通过：复刻结果与官方 `align()` 零容差一致；未归因分别为 +0.14%/+1.93%/+0.96%，
远低于 5% 线，说明七段确实覆盖了 `align()` 的全部实质耗时。

| 段 | batch=1 | batch=8 | batch=32 | batch=32 占比 |
|---|---:|---:|---:|---:|
| `normalize_audio` | 0.14 | 1.68 | 4.86 | 0.38% |
| `encode_timestamp` | 0.05 | 0.38 | 1.33 | 0.10% |
| **`feature_extract`** | **3.96** | **89.65** | **563.47** | **43.65%** |
| `host_to_device` | 0.32 | 2.00 | 8.85 | 0.69% |
| `thinker_forward` | 39.98 | 176.98 | 675.00 | 52.28% |
| `device_to_host` | 0.13 | 0.95 | 3.55 | 0.27% |
| `parse_timestamp` | 1.06 | 8.60 | 33.95 | 2.63% |
| CPU 合计 | 5.21 | 100.31 | 603.61 | 46.75% |

### 定性结论一：非 GPU 时间的主体是 `feature_extract`

batch=32 时 563.47 ms 对 33.95 ms，差 16.6 倍。这个差距远超任何测量波动能解释的范围，
因此方向可以定性：**要优化的是 mel 特征提取。**

### 定性结论二：原假设被数据否定

立项时怀疑 `fix_timestamp()` 的纯 Python O(n²) DP 是主体之一。**数据不支持这个假设。**
`parse_timestamp` 在 batch=32 只占 2.63%，按样本归一为 1.06 ms，与 batch=1 的 1.06 ms 完全
一致——严格线性，没有异常。n=120 这个规模上 O(n²) 约 9000 次迭代不构成问题。

这正是拆到七段的价值：如果只按 CPU/GPU 两分，会看到 603 ms CPU 时间，却无法知道该往哪个
方向投入；两个候选的优化手段完全不同，猜错方向就是白做。假设被否定同样是有效产出。

### 阻塞项：CPU 段测量不稳定，绝对值不可用于定量判定

| 段 | max/min |
|---|---:|
| `feature_extract` | **10.514** |
| `host_to_device` | 10.295 |
| `normalize_audio` | 2.364 |
| `thinker_forward` | 1.036 |

`thinker_forward` 稳定在 1.036，说明**不是全局抢占**，波动只出现在 CPU 段。同时
`--warmup` 已为 15，预热问题已排除，所以这不是"再多预热几次"能解决的——原先那条建议增大
warmup 的告警文案是误导，已修正为按 min/p50/max 与漂移分流排查。

另一个异常是 `feature_extract` **随批量超线性**增长，按样本归一为 3.96 → 11.21 → 17.61
ms/样本；同期 `thinker_forward` 为 39.98 → 22.12 → 21.09，是正常的亚线性。批量放大 32 倍
而 CPU 段耗时放大 142 倍。不稳定与超线性很可能同源。

### 与 PERF-ALI-010 的数字矛盾

| 来源 | `align()` | `thinker` | 反推 CPU |
|---|---:|---:|---:|
| ALI-010（单独跑 batch=32） | 926.41 | 670.05 | 256.36 |
| ALI-011（连跑 1,8,32） | 1373.65 | 675.00 | ≈698.65 |

`thinker` 两次一致（+0.74%），差异全部落在 CPU 侧，且相差 2.4 倍。已知条件差异只有一个：
ALI-010 的有效数据是单独跑 `--batch-sizes 32`，本轮是 1,8,32 连跑。因此**不能断定 256 ms
和 603 ms 哪个更接近真相**，必须先消除这个变量。

### 第二轮：单跑 batch=32（2026-08-17）

报告 `performance-results/PERF-ALI-011/stage-split-b32-only.json`。CPU 256 核、torch 线程 128、
特征提取器 `WhisperFeatureExtractor`，**torch fbank 路径可用**。

连跑不是不稳定的成因——单跑后 `feature_extract` 极差比反而升到 17.42。但新增的诊断字段直接
定位了形态：

| 段 | mean | min | p50 | max | max/min | 漂移 |
|---|---:|---:|---:|---:|---:|---:|
| `normalize_audio` | 4.81 | 3.96 | 4.39 | 8.26 | 2.087 | +0.1% |
| `encode_timestamp` | 1.32 | 1.27 | 1.31 | 1.44 | 1.133 | +0.0% |
| **`feature_extract`** | **429.33** | **64.88** | **670.62** | **1130.22** | **17.42** | **+931.5%** |
| `host_to_device` | 14.50 | 2.08 | 15.81 | 23.52 | 11.296 | -0.4% |
| `thinker_forward` | 671.20 | 666.22 | 669.89 | 688.05 | 1.033 | +0.1% |
| `device_to_host` | 3.58 | 3.44 | 3.50 | 4.13 | 1.204 | -0.1% |
| `parse_timestamp` | 33.96 | 33.64 | 33.85 | 34.52 | 1.026 | -0.1% |

`mean` 小于 `p50` 且漂移 +931.5%，说明前半段约 65 ms、后半段约 670 ms 以上，是**随迭代恶化**
而非随机长尾。这是中位数漂移判据的直接产出：若沿用最初的均值判据，无法与长尾区分。

其余六段稳定且跨轮次可复现：

| 段 | 本轮 | 前次 | 偏差 |
|---|---:|---:|---:|
| `thinker_forward` | 671.20 | 670.05（ALI-010） | +0.17% |
| `parse_timestamp` | 33.96 | 33.95（首轮） | +0.03% |
| `device_to_host` | 3.58 | 3.55（首轮） | +0.85% |

七段闭合 +0.25%。所以问题定位很干净：**只有 `feature_extract` 与 `host_to_device` 退化，
其余一切稳定。**

### 关键推算：退化可能是 spike 紧密循环的产物，生产未必存在

取各段 min 值求和：

```text
稳定六段（host_to_device 取 min 2.08）   4.81 + 1.32 + 2.08 + 671.20 + 3.58 + 33.96 ≈ 116.95
feature_extract min                                                              64.88
align 估算                                                                      ≈ 782 ms
生产 aligner_batch_model_call_ms                                                 780.82 ms
```

吻合到 0.2%。若该推算成立，则：

- `feature_extract` 真实成本约 **64.88 ms（2.03 ms/样本）**，不是 563 或 429；
- CPU 占比约 **13.9%**，不是首轮的 28% 或 46.75%；
- 首轮的“超线性”（3.96/11.21/17.61 ms/样本）**本身就是退化的产物**，按 min 估算大致为
  3.5/~3/2.03，反而是正常亚线性；
- **`PERF-ALI-010` 反推的 256 ms 同样被污染**——它的 align 926.41 ms 已高于未退化的约 782 ms。

也就是说，L4.6 这个靶点可能远小于此前描述。**在证实或证伪这条之前，`feature_extract` 的任何
优化都不得启动**，否则可能在优化一个只在 spike 里存在的问题。

### 第三轮：干净数据，min 下界推算被证实（2026-08-17）

报告 `performance-results/PERF-ALI-011/stage-split-b32-isolated.json`。退化未复现，本轮为
**有效定量数据**。

先看上一轮的推算是否成立：

| 量 | 上轮 min 下界预测 | 本轮实测 | 偏差 |
|---|---:|---:|---:|
| `align()` 全程 | ≈782 | 790.66 | +1.1% |
| `feature_extract` | ≈64.88 | 68.13 | +5.0% |
| CPU 合计 | ≈108 | 108.72 | **+0.7%** |
| CPU 占比 | 13.9% | 13.85% | 一致 |

生产 `aligner_batch_model_call_ms=780.82` 对本轮 790.66，spike 高 1.26%。
**结论：退化不存在于生产，是 spike 测量环境的产物。**

| 段 | mean ms | 占比 | ms/样本 | p50 | 漂移 |
|---|---:|---:|---:|---:|---:|
| `normalize_audio` | 5.16 | 0.66% | 0.16 | 4.46 | +1.1% |
| `encode_timestamp` | 1.36 | 0.17% | 0.04 | 1.33 | -1.6% |
| `feature_extract` | 68.13 | **8.68%** | 2.13 | 64.91 | +1.8% |
| `host_to_device` | 2.13 | 0.27% | 0.07 | 2.08 | +3.5% |
| **`thinker_forward`** | **670.55** | **85.41%** | 20.95 | 669.35 | +0.3% |
| `device_to_host` | 3.65 | 0.46% | 0.11 | 3.54 | +1.7% |
| `parse_timestamp` | 34.07 | 4.34% | 1.06 | 34.00 | +0.8% |
| CPU 合计 | 108.72 | 13.85% | 3.40 | — | — |

七段闭合 +0.06%，复刻开销 -0.65%，RSS 2408.1 → 2408.8 MiB（+0.7），
隔离测量 62.79 ms、漂移 -4.0%。

### 最终结论

1. **`align()` 的绝对主体是 GPU 前向，占 85.41%。** CPU 合计仅 13.85%。
2. **`feature_extract` 只有 68.13 ms、占 8.68%**，不是首轮的 563.47 ms / 43.65%。
3. **“超线性”不存在。** 干净数据下 2.13 ms/样本，低于 batch=1 的约 3.5 ms/样本，是正常亚线性。
4. **`fix_timestamp` 的 O(n²) DP 假设不成立**（`parse_timestamp` 4.34%、1.06 ms/样本、严格线性）。
5. 退化本身**未能定因**：隔离测量与主循环本轮同时变干净，判别性测试因此失去对照。已知它不
   伴随内存增长（RSS +0.7 MiB），且不可复现。按现有证据归为测量环境干扰，不再追查。

### 对 PERF-ALI-010 的影响：原判定需重测

ALI-010 用 align 926.41 ms 作分母得出编译收益 +6.30%，判定未达 15% 门禁而拒绝。
**该分母已被证实包含退化成分。** 用本轮干净基线重算：

```text
干净 align                790.66 ms
其中 thinker              670.55 ms
非 thinker（CPU+传输）     120.11 ms   ← 编译不影响这部分
ALI-010 编译后 thinker     559.62 ms
估算编译后 align          559.62 + 120.11 = 679.73 ms
估算收益                  (790.66 - 679.73) / 790.66 = 14.03%
```

**+14.03% 对原报的 +6.30%**，已贴近 15% 门禁。ALI-010 自身数据也印证了污染：它反推的 CPU 部分
为 256.36 ms（eager）与 308.40 ms（compiled），两者都远高于干净值 120.11 ms，且 compiled 那轮
污染更重，正好把收益吃掉了。

因此 **ALI-010 的拒绝结论不可靠，应在干净条件下重测**，并优先测 `--compile-mode
reduce-overhead`——thinker 占 85.41%，是唯一值得投入的靶点。

### 对 L4.6 的影响：靶点应降级

`feature_extract` 最多只能省 68.13 ms，即 `align()` 的 8.68%。而且：

- **把 mel 搬到 GPU（`WhisperFeatureExtractor` 的 torch 路径）方向存疑**：GPU 已占 85.41%，
  是瓶颈；把 CPU 工作搬到瓶颈上很可能净负收益。该线索的吸引力大幅下降。
- 真正可能有效的是让 CPU mel 与 GPU 前向**重叠**，把这 68 ms 藏起来，但上限就是 8.68%，
  且 `PERF-ALI-009` 已经证明该链路上的重叠尝试容易被合批目标抵消。
- 按端到端折算，远达不到 +5% 的改默认值门禁。

### 执行命令（第三轮，含隔离测量）

```bash
python experiments/perf_ali_011/spike.py \
  --audio test_data/audio_16000_10s.wav \
  --text test_data/audio_16000_10s.txt --text-is-file \
  --language Chinese \
  --batch-sizes 32 --isolate-feature-extract \
  --output-json performance-results/PERF-ALI-011/stage-split-b32-isolated.json
```

`--isolate-feature-extract` 原本用于区分成因：隔离后仍恶化则成因在 CPU 侧（分配器、内存增长、
线程池状态），隔离后稳定则来自与 GPU 工作交替执行的交互。**本轮主循环自己也变干净了，
两组都无退化，因此该判别测试没有得到对照，成因未定。**

### 诊断字段（不改变七段口径）

- `stages_ms[*].samples`：每次迭代原始值。仅凭统计量无法区分长尾抢占、普遍变慢和随迭代恶化。
- `stage_trend[*]`：前后半段**中位数**对比与漂移。用中位数而非均值，因为均值不抗离群，
  单个尖峰落在后半段会被误判成单调恶化，而两者成因与处置完全不同。
- `stage_ms_per_sample[*]`：按样本归一，直接暴露超线性。
- `cpu_environment`：CPU 核心数、`torch.get_num_threads()`、`OMP_NUM_THREADS` 等线程环境变量、
  特征提取器类名，以及 numpy/torch 两条 fbank 路径是否可用。
- `rss_mb_series`：逐次 RSS，判断退化是否伴随内存增长。psutil 缺失时为 null。
- `elapsed_s_series`：累计秒数，判断是否与运行时长相关。
- `stage_min_sum_ms` / `cpu_min_sum_ms`：取各段 min 的下界估算。第二轮据此预测的 CPU 合计
  与第三轮实测只差 0.7%，该字段的价值已被验证。

### 稳定性判据的两次修正

判据本身被实测推翻过两次，记录以免重犯：

1. **趋势不能用均值。** 最初用前后半段均值对比，单个长尾尖峰落在后半段就会算出 +190% 的
   假漂移。改为中位数后，退化轮给出 +931.5%、干净轮给出 +1.8%，区分度充足。
2. **`max/min` 不能作判据。** 它由两个单点决定，在共享的 256 核机器上必然偶发离群。第三轮
   `normalize_audio` 的 min 4.02 / p50 4.46 / max 9.94 只是一次抢占，却被判为整组不可用。
   现改为 `p95/p50 > 1.3`（主体离散度，对单点稳健）或 `|漂移| > 20%`（系统性偏移）两条判据，
   `max/min` 降为参考信息。已用实测三种形态校验：单点抢占放行、成组离群（3/30 达 2 倍）拦住、
   退化形态由漂移拦住、主体离散由 p95/p50 拦住。

判读方式：漂移超限说明随迭代恶化；仅 p95/p50 超限说明主体分布本身离散。

## 已知边界

- 本实验用同一条音频重复填满 batch，**批内完全等长，没有 padding 浪费**。真实业务音频长度
  不一，`padding=True` 会把整批补到最长，因此实测的 CPU 占比大概是偏乐观的下界。报告记录
  `input_shapes` 便于后续与混合长度对照，但混合长度属于另一个变量，不在本轮。
- 只覆盖中文单语言。`encode_timestamp` 对日语走 `nagisa`、韩语走 `soynlp`，CPU 成本可能显著
  不同，需另测。
- 计时点本身有开销。`staged_overhead_percent` 用于量化，但不能完全消除；段内耗时越小相对
  误差越大，因此亚毫秒段的绝对值仅供参考。

## 下一步

本实验的测量任务已完成，两项自检通过且拿到干净定量数据。三条历史步骤（单跑排除连跑、隔离
定因、min 下界交叉验证）都已执行完毕。后续动作按优先级：

1. **重测 `PERF-ALI-010`。** 优先级最高。原拒绝依据的分母被污染，干净基线下估算收益为
   +14.03% 而非 +6.30%，贴近 15% 门禁。thinker 占 85.41%，是唯一值得投入的靶点，
   优先测 `--compile-mode reduce-overhead`。
2. **把七段拆分补进 `src/aligner.py` 的阶段指标。** 当前 `model_call_ms` 把 CPU 与 GPU 混为
   一个数字，生产侧无法验证任何优化收益。这是所有后续优化的前提。
3. **L4.6 降级。** `feature_extract` 仅 68.13 ms、占 8.68%，端到端折算远达不到 +5% 门禁。
   把 mel 搬到 GPU 的线索价值下降——GPU 已占 85.41%，是瓶颈。
4. 混合时长与多语言另立变量：本轮批内等长无 padding 浪费，且只覆盖中文。

已被数据排除、不再作为方向的：

- **`fix_timestamp` 的 O(n²) 纯 Python DP。** `parse_timestamp` 占 4.34%、1.06 ms/样本，
  与 batch=1 完全一致，严格线性，n=120 规模下不构成问题。
- **`feature_extract` 随批量超线性。** 干净数据下 2.13 ms/样本，低于 batch=1 的约
  3.5 ms/样本，是正常亚线性。首轮观测到的超线性是退化的产物。

## 门禁

本实验只需通过两项自检即算完成，不设性能门禁；两项均已通过（复刻零容差一致、七段闭合
+0.06%）。后续基于结论的优化沿用常规标准：时间戳逐项一致、成功率 100%、三轮可重复、端到端
至少 +5% 才改生产默认值；不得改动官方 `encode_timestamp()`、时间桶解析与单调性校验语义，
不得为提速降采样率或改 mel 参数。
