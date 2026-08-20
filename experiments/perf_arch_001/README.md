# PERF-ARCH-001：vLLM token-classify ForcedAligner spike

主服务当前使用单 Python 环境 `vllm[audio]==0.19.1`（CUDA 12.9），时间戳由
`qwen-asr==0.0.6` 的 PyTorch ForcedAligner 提供。本目录只验证后续把该后端替换为 vLLM
pooling / token-classification 的语义正确性与原始吞吐。

`src/aligner.py`、网关和生产启动入口**尚未集成**该后端。本 spike 通过不代表生产已支持。

本路线已于 2026-08-19 定案为**拒绝**：Phase B 原始吞吐达到 +188.1%，但生产 oracle 严格门禁
仅 6/11；两向窗口语义诊断虽显著减少桶分歧，仍未达到 11/11。继续逐层兼容性定位投入产出不足，
因此停止 ARCH-001，不进入 Phase C、不重复性能 spike，历史工具和结果仅作为失败证据保留。
后续转入独立 `PERF-ARCH-002` TensorRT FP16 导出路线。

## 边界

- 使用项目根统一环境，不安装第二个 vLLM、不建双 venv、不使用 `--no-deps`。
- 不引入 CUDA 13 构建：目标 A10 宿主驱动为 535，只支持 CUDA 12.x。
- 模型只从 ModelScope 固定 revision 下载到独立目录，不覆盖生产模型目录。
- `words` 必须来自官方 processor，不得按字符或空格自行猜测，也不得因后端变化简化 11 种
  语言的分词与时间戳后处理。
- 未达 Phase A/B 门禁前不修改现有 PyTorch 时间戳链路。

目录：

```text
experiments/perf_arch_001/
├── phase_a.py                       11 语种预检 + PyTorch/vLLM 直接对照
├── phase_a_manifest.example.json    样本 manifest 模板
├── prepare_phase_a.py               样本准备：音频规范化、参考文本与 manifest 生成
├── requirements.txt                 有意为空，完全复用根精确依赖
├── spike.py                         单样本原始吞吐与缓存实验
└── README.md
```

## 1. 环境

实验完全复用项目根已精确固定的单一环境，`requirements.txt` 有意保持为空，不在这里安装
第二套依赖：

```bash
python -m pip install -r experiments/perf_arch_001/requirements.txt
python -m pip check
python -c "import importlib.metadata as m; assert m.version('vllm') == '0.19.1'; assert m.version('qwen-asr') == '0.0.6'"
python -c "import torch; assert torch.version.cuda.startswith('12.')"
```

`spike.py` 启动时会再次校验 vLLM 版本，不匹配直接失败。

## 2. 独立模型目录

复用项目下载器，必须同时覆盖三个参数：

```bash
VLLM_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B \
VLLM_MODEL_DIR=models/qwen3-forced-aligner-0.6b/vllm-spike \
MODELSCOPE_REVISION=cf1c50164ea3ac48240d12bef5ead74aee0720cc \
python scripts/download_model.py
```

不得使用生产目录 `models/qwen3-forced-aligner-0.6b/pt`。脚本会读取
`.modelscope-manifest.json`，仓库或 revision 不一致时直接失败。

## 3. Phase A：11 语种语义对照

支持语言固定为：Chinese、English、Cantonese、French、German、Italian、Japanese、Korean、
Portuguese、Russian、Spanish。必须每种语言提供一条真实、最长 32 秒的 16 kHz 单声道音频及
与内容一致的 UTF-8 参考文本。

### 3.1 准备样本

`phase_a_manifest.example.json` 里的 `test_data/phase_a/*` 是**占位路径，仓库中不存在**。
仓库现有 `test_data` 只覆盖中文和一条 3.91 秒英文，斯瓦希里语样本不在 Aligner 的 11 语种内，
必须自行补齐 Cantonese、French、German、Italian、Japanese、Korean、Portuguese、Russian、
Spanish 共 9 种真实人声音频。

拿到源音频后用 `prepare_phase_a.py` 一次完成规范化、参考文本与 manifest。源文件按
`<语种>.<扩展名>` 命名（大小写不敏感，扩展名不限），脚本用 ffmpeg 统一成 16 kHz 单声道 PCM
WAV 并截到 32 秒内：

```bash
python experiments/perf_arch_001/prepare_phase_a.py \
  --source-dir raw_samples \
  --output-dir test_data/phase_a \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json
```

参考文本优先用语料自带的逐字稿，此时传 `--no-transcribe` 并手工放好同名 `.txt`。没有逐字稿
时脚本默认调用本机 `/chinese_asr` 显式指定语种转写（请求字段是 `base64`，响应取 `istar_asr`），
需要服务已在 `:8080` 运行。Phase A 比较的是两个 Aligner 后端在**同一** audio+text 上是否等价，
文本只需真实对应音频内容，不必是权威 ground truth；但主模型识别错的词会同时进入两个后端，
所以生成后必须人工抽查，脚本不替代这一步，也不做语种判定。

脚本跑完 11 种才写 manifest，未就绪时列出全部缺口并返回非零；已就绪语种会沿用，不重复转写。
手工维护 manifest 也可以，路径按执行命令时的工作目录解析，必须恰好 11 项且语种不重复。

### 3.2 预检

样本准备好后先跑不占 GPU 的预检，它只做 CPU 侧检查
（manifest 结构、11 语种齐全与去重、16 kHz 单声道、32 秒上限、UTF-8 文本、官方
`encode_timestamp()` 分词），不加载任何模型权重，可以在服务运行期间执行：

```bash
python experiments/perf_arch_001/phase_a.py check \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json \
  --output-json performance-results/PERF-ARCH-001/phase-a-check.json
```

`check` 一次列出全部问题而不是首错停止，`11/11` 且退出码 0 之后再往下走。它默认调用官方
processor；缺 qwen-asr 时会报错而不是静默跳过，只有明确不需要分词预检才传 `--no-processor`。
预检还会输出单元数与每秒单元密度，密度明显偏离常见范围时给出提示——这只是提示，**不能证明
文本与音频内容对应**，那一项仍需人工确认。

### 3.3 对照

先独占 GPU 生成 PyTorch oracle：

```bash
python experiments/perf_arch_001/phase_a.py oracle \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json \
  --output-json performance-results/PERF-ARCH-001/phase-a-oracle.json
```

进程退出、GPU 释放后，再启动 vLLM token-classify 对照：

```bash
python experiments/perf_arch_001/phase_a.py compare \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json \
  --expected-json performance-results/PERF-ARCH-001/phase-a-oracle.json \
  --output-json performance-results/PERF-ARCH-001/phase-a-compare.json
```

attention 单变量复测固定 Set2、PyTorch BF16 oracle、candidate BF16、top-5、eager 和 1 ms 门禁，
只把 MM encoder/音频塔从自动选择改为显式 `TORCH_SDPA`：

```bash
python experiments/perf_arch_001/phase_a.py compare \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json \
  --expected-json performance-results/PERF-ARCH-001/phase-a-oracle.json \
  --output-json performance-results/PERF-ARCH-001/phase-a-compare-sdpa.json \
  --dtype bfloat16 --diagnostic-top-k 5 \
  --mm-encoder-attn-backend TORCH_SDPA
```

参数值不使用静态猜测。目标 A10/vLLM 0.19.1 探针返回的 ViT 后端为 `FLASH_ATTN`、
`TRITON_ATTN`、`TORCH_SDPA`、`FLASHINFER`，且 `EngineArgs` 明确暴露
`mm_encoder_attn_backend` 并写入 `MultiModalConfig`。`TORCH_SDPA` 的枚举 value 虽为空字符串，
枚举 name 仍为 `TORCH_SDPA`；脚本按 name 匹配并传枚举对象，不按 value 判断。运行时仍会从当前
平台重新读取支持列表，模型加载后再从
`model_config.multimodal_config.mm_encoder_attn_backend` 回读。后端不支持、入口不存在或回读
不一致都会直接失败，不会改用普通 `attention_config.backend`，也不会静默退回 FLASH_ATTN。

显式复测的启动日志必须同时出现：

```text
Using backend AttentionBackendEnum.TORCH_SDPA for vit attention
Using AttentionBackendEnum.TORCH_SDPA for MMEncoderAttention
```

如果仍显示 `FLASH_ATTN`，即使脚本产出了比较结果，本轮也作废。报告会记录请求值、平台支持列表、
配置回读值和严格验证状态；`auto` 模式无法从控制进程证明实际自动选择，因此 `actual` 留空并要求
以启动日志为准，不伪造实际后端。

### 3.4 音频窗口 mask 诊断（不参与晋级）

源码审计确认生产 PyTorch oracle 的 SDPA 忽略 `cu_seq_lens_q/k`，而 vLLM Torch SDPA 按
`cu_seqlens` 拆窗。只为量化该差异，可生成启用 PyTorch 源码已有块对角 mask 的诊断 oracle：

```bash
python experiments/perf_arch_001/phase_a.py oracle \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json \
  --output-json performance-results/PERF-ARCH-001/phase-a-oracle-set2-audio-block-mask-diagnostic.json \
  --dtype bfloat16 --diagnostic-top-k 5 \
  --diagnostic-oracle-audio-block-mask
```

释放 PyTorch 进程后，用同一 Set2、BF16 和 vLLM `TORCH_SDPA` 比较：

```bash
python experiments/perf_arch_001/phase_a.py compare \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json \
  --expected-json performance-results/PERF-ARCH-001/phase-a-oracle-set2-audio-block-mask-diagnostic.json \
  --output-json performance-results/PERF-ARCH-001/phase-a-compare-set2-audio-block-mask-diagnostic.json \
  --dtype bfloat16 --diagnostic-top-k 5 \
  --mm-encoder-attn-backend TORCH_SDPA \
  --allow-audio-block-mask-diagnostic-oracle
```

诊断 oracle 使用独立 experiment ID；compare 必须显式允许，并且完成后固定返回退出码 2。报告中的
通过数只用于判断窗口 mask 对桶分歧的贡献，即使 11/11 也**不得**作为 Phase A 晋级证据。默认
oracle/compare 路径没有改变。

### 3.5 vLLM 跨窗口反向诊断（不参与晋级）

块 mask 诊断显著改善后，迁移门禁需要反向验证：只让 vLLM `TORCH_SDPA` 忽略
`cu_seqlens`，复刻当前生产 PyTorch oracle 的跨窗口全序列语义。因为该补丁必须与模型执行位于
同一进程，脚本会设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，并按每条请求检查补丁调用计数；
补丁未命中时直接失败，不产出伪结果。

关闭独立 EngineCore 本身也是执行条件变化，所以必须先在相同单进程条件下、不启用补丁重跑基线：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
python experiments/perf_arch_001/phase_a.py compare \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json \
  --expected-json performance-results/PERF-ARCH-001/phase-a-oracle-set2-refresh-bf16.json \
  --output-json performance-results/PERF-ARCH-001/phase-a-compare-set2-sdpa-inprocess-baseline.json \
  --dtype bfloat16 --diagnostic-top-k 5 \
  --mm-encoder-attn-backend TORCH_SDPA
```

基线必须复现 6/11、59/996 原始桶分歧和 80 个修复后超限边界；否则停止，不运行候选。复现后只加
一个诊断开关：

```bash
python experiments/perf_arch_001/phase_a.py compare \
  --manifest performance-results/PERF-ARCH-001/phase-a-manifest.json \
  --expected-json performance-results/PERF-ARCH-001/phase-a-oracle-set2-refresh-bf16.json \
  --output-json performance-results/PERF-ARCH-001/phase-a-compare-set2-sdpa-cross-window-diagnostic.json \
  --dtype bfloat16 --diagnostic-top-k 5 \
  --mm-encoder-attn-backend TORCH_SDPA \
  --diagnostic-vllm-audio-cross-window
```

该候选只能对照未修改的生产 oracle，不能与 `--allow-audio-block-mask-diagnostic-oracle` 叠加；
非 `TORCH_SDPA` 后端也会被拒绝。报告使用独立 experiment ID、记录补丁命中次数，并固定返回
退出码 2。

目标 A10 实测中，单进程未打补丁基线精确复现 **6/11、59/996、80 个超限边界**，证明进程拓扑
没有改变语义口径。只增加跨窗口补丁后仍为 **6/11**，但原始桶分歧降至 **7/996（0.70%）**，
修复后超限边界降至 **11**；按数量分别减少 88.14% 和 86.25%。原始桶分布由
Chinese/Cantonese/French/Japanese/Korean/Spanish 的 **0/25/15/1/17/1** 变为
**1/3/1/0/1/1**。跨窗语义是主要贡献因素，但作为直接兼容修复仍未达到 11/11，判定拒绝。项目负责人已决定停止
逐层定位；本诊断不再继续扩展，Phase C 不启动，后续转入 ARCH-002。

`oracle` 与 `compare` 故意分进程执行，避免同时加载 PyTorch 与 vLLM 两个模型。`oracle` 使用官方
`encode_timestamp()`、`align()` 和 `parse_timestamp()` 生成基线；`compare` 固定 batch=1、
关闭 prefix cache、不做音频扰动，每个 vLLM 结果直接与对应 PyTorch 样本对照，不经跨 batch
或首样本间接比较。

门禁为 **11/11**：单元文本与数量完全一致，最终 `start/end` 逐项偏差不超过 1 ms，时间戳非降、
非重叠、不越音频边界。官方允许相邻边界相等和零宽单元，因此这里不伪称“严格递增”。任一语种
失败时 `compare` 继续收集其余语种，报告列出全部失败并返回非零退出码。

为定因而不放宽门禁，oracle 与 compare 默认用 `--diagnostic-top-k 5` 记录每个 timestamp 位置的
候选桶。compare 报告只汇总分歧位置的双侧 top-k、各自 top1-top2 logit 间距和是否互为候选；
该参数允许 2～50，只影响诊断信息，**不参与 1 ms 通过判定**。修改 top-k 后必须用相同值重新运行
oracle 和 compare，不能拿旧 oracle 推断双侧候选关系。compare 默认拒绝 oracle/candidate dtype 不同，
防止误混口径；需要固定生产 BF16 oracle、仅测试 FP16 candidate 时，必须显式传
`--allow-oracle-dtype-mismatch`，报告会同时记录两侧 dtype。

oracle 含派生分词单元、时间戳和候选数据，可能还原正文，**不得提交 Git**；仅在台账提交去敏后的语言、
音频/文本 SHA256、单元数、最大偏差和 11/11 结论。原始音频、参考文本和 `test_data/phase_a/`
下的样本同样不入库。

## 4. 单样本性能模式：准备 words

必须来自官方 `Qwen3ForceAlignProcessor.encode_timestamp()`，且**文本要与音频真实内容对应**。
用示例文本对不上的音频会得到全部挤在音频开头的无效时间戳，而格式校验拦不住这种错误。

```bash
python - <<'PY'
import json
import pathlib
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor

text = pathlib.Path("test_data/audio_16000_10s.txt").read_text(encoding="utf-8").strip()
processor = Qwen3ForceAlignProcessor()
words, _ = processor.encode_timestamp(text, "Chinese")
print(len(words))
print(json.dumps(words, ensure_ascii=False))
PY
```

可选的 `--expected-json` 用于与当前 PyTorch 实现逐单元对照，格式为：

```json
{
  "units": [
    {"text": "你", "start": 0.12, "end": 0.35},
    {"text": "好", "start": 0.35, "end": 0.61}
  ]
}
```

单元文本与数量必须完全一致，时间默认允许 1 ms 误差。真实基线与业务正文不要提交 Git。

## 5. 执行

spike 需要独占 GPU，先停掉服务并确认显存释放：

```bash
pkill -f 'src/gateway.py'
pkill -f 'vllm serve'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python experiments/perf_arch_001/spike.py \
  --model models/qwen3-forced-aligner-0.6b/vllm-spike \
  --audio test_data/audio_16000_10s.wav \
  --language Chinese \
  --words-json '["这里","填写","官方分词结果"]' \
  --batch-sizes 1,8,16,32 \
  --warmup 3 --iterations 10 \
  --output-json performance-results/PERF-ARCH-001/chinese-10s-nocache.json
```

## 6. 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `models/qwen3-forced-aligner-0.6b/vllm-spike` | 固定 revision 本地目录 |
| `--audio` | 必填 | 16 kHz 单声道，最长 32 秒 |
| `--language` | 必填 | 官方语言名称，仅写入报告 |
| `--words-json` | 必填 | 官方 processor 输出的单元数组 |
| `--batch-sizes` | `1,8,16,32` | 不重复正整数，不得大于 `--max-num-seqs` |
| `--warmup` | `3` | 每个 batch 的预热次数 |
| `--iterations` | `10` | 每个 batch 的正式执行次数 |
| `--dtype` | `bfloat16` | 可选 `float16` |
| `--max-num-seqs` | `32` | 必须不小于最大 batch size |
| `--gpu-memory-utilization` | `0.80` | 取值须在 0～1 之间 |
| `--enforce-eager` | 开启 | 按官方示例；图模式属后续单变量 |
| `--enable-prefix-caching` | 关闭 | 见下节缓存说明 |
| `--distinct-batch-audio` | 开启 | 见下节缓存说明 |
| `--expected-json` | 无 | PyTorch 基线 units 对照文件 |
| `--timestamp-tolerance-ms` | `1.0` | 时间戳对照容差 |
| `--output-json` | 无 | 原子写入报告 |

## 7. 缓存抑制（结果可信度的前提）

批内如果重复完全相同的 prompt 与音频，vLLM 的前缀与多模态缓存可能让实际计算次数远少于
batch size，使原始吞吐虚高。因此默认：

- `--enable-prefix-caching` 关闭；
- `--distinct-batch-audio` 开启，对批内第 `i>0` 个请求的单个采样点施加 1e-6 扰动，改变
  多模态哈希但不改变时长、声道与波形形状。索引 0 始终保留原始音频，语义结果只从该请求提取。

开启 `--distinct-batch-audio` 时，同批输出不再要求逐位相等，而是逐单元比对文本并要求时间戳
落在 `--timestamp-tolerance-ms` 容差内，因此校验强度不降低。

量化缓存影响时再跑一次对照即可：

```bash
python experiments/perf_arch_001/spike.py \
  ... \
  --enable-prefix-caching --no-distinct-batch-audio \
  --output-json performance-results/PERF-ARCH-001/chinese-10s-cached.json
```

两份接近说明抑制版数据可信；差距明显则一律以抑制版为准。已实测：batch=32 抑制版
1180.51、开缓存版 1168.90 audio_s/s，差 -1.0% 属噪声。

## 8. 历史校验与晋级条件（路线已拒绝）

`spike.py` 内置校验：模型身份、vLLM 版本、16 kHz 单声道、最长 32 秒、timestamp 数量等于
单元数 × 2、官方 `fix_timestamp()` 等价的非递减修复、时间戳非降且不重叠、`start <= end`
且不越音频边界。允许相邻边界相等和 `start == end` 的零宽单元，这与官方修复后的输出一致，
不伪称“严格递增”。

晋级条件：

- 11 种语言的单元文本、数量与最终时间戳全部通过 PyTorch 基线对照。
- batch=32 原始中位吞吐相对同环境 PyTorch 至少 +30%。
- 成功率 100%，无 OOM、NaN、死锁或静默降级。
- 未达标即停止，不创建 HTTP 服务、不修改网关。
- 达标后才进入 Phase C：独立服务、网关适配、三轮 2000 请求、端到端 +20%、混合时长、CER
  与故障隔离。

## 9. 已有结论

- 同环境 PyTorch 分母：满批 32 片 × 10 秒下 `aligner_batch_model_call_ms=780.82`，
  即纯 `model.align()` 约 409.8 audio_s/s，含 WAV 解码约 344.9。
- spike batch=32 为 1180.51 audio_s/s，同口径 **+188.1%**，Phase B 在“单语言、10 秒、单片”
  范围内通过。
- Phase A 样本、check 与 PyTorch oracle 均为 11/11；vLLM BF16 candidate 在 Set1/Set2 分别为
  **7/11、6/11**。固定 Set2 BF16 oracle、只改 FP16 candidate 后为 **7/11**，原始桶分歧仅从
  59/996 降到 58/996，dtype 路线无实质收益。刷新同一 Set2 oracle 后，AUTO 再次精确复现
  **6/11、59/996**；只把 MM encoder attention 改为 `TORCH_SDPA` 仍为 **6/11、59/996**，French
  减少 1 桶而 Korean 增加 1 桶，修复后超限边界仍为 80。样本、dtype 和 MM encoder SDPA 三条
  假设均被否定；不盲扫其他 ViT/MM encoder 后端。源码审计确认 PyTorch 生产 oracle 未调用已定义的
  音频块对角 mask，Transformers SDPA 也忽略 `cu_seq_lens_q/k`，实际做跨窗口全序列 attention；
  vLLM Torch SDPA 则按 `cu_seqlens` 拆窗。显式块 mask 诊断 oracle 已在 24 层音频 attention 上
  生效并生成 11/11 基线；对同一 vLLM `TORCH_SDPA` 比较由生产 oracle 的 6/11、59/996 原始桶
  分歧、80 个修复后超限边界改善为诊断 oracle 的 **8/11、6/996（0.60%）、5 个超限边界**。
  按数量口径原始桶分歧减少 89.83%，Cantonese 25→0、French 15→4、Japanese 1→0、Korean
  17→1、Spanish 1→1；剩余 6 处均只差 1 桶且双侧近似平局。窗口 mask 是主要贡献因素但不是
  唯一差异；诊断修改了 PyTorch 语义，独立 ID、显式允许及固定退出码 2 继续防止误晋级。
  反向跨窗诊断的单进程基线精确复现 6/11、59/996、80 个超限边界；只让 vLLM 忽略
  `cu_seqlens` 后仍为 6/11，但降至 **7/996（0.70%）和 11 个超限边界**。两种方向都证明窗口
  语义是主要贡献因素，却都没有达到生产 oracle 11/11；跨窗补丁作为直接兼容修复被拒绝。
  项目负责人据此终止逐层定位，ARCH-001 整体拒绝，Phase C 不启动。
- 未完成项不再继续：Phase A 生产 oracle 严格 11/11、逐层兼容性定位和 Phase C 集成。
- 注意端到端与原始吞吐不等价：PyTorch 前向从 1097.04 ms 优化到 780.82 ms（-28.8%）时，
  端到端仅 +3.82%。端到端收益主要受进程拓扑约束，集成前需先解决拓扑问题。

`spike.py` 不采集 GPU 指标，正式证据需同时运行：

```bash
nvidia-smi dmon -i 0 -s pucm -d 1 \
  > performance-results/PERF-ARCH-001/gpu-dmon.txt
```

## 10. 实现依据

- [vLLM 0.19.0 Token Classification 用法](https://docs.vllm.ai/en/v0.19.0/models/pooling_models/token_classify/)
- [vLLM 官方 forced alignment 示例源码](https://github.com/vllm-project/vllm/blob/main/examples/pooling/token_classify/forced_alignment_offline.py)
- [Qwen3 ForcedAligner vLLM 模型实现](https://docs.vllm.ai/en/v0.19.0/api/vllm/model_executor/models/qwen3_asr_forced_aligner/)

官方资料内容已转述，并按本项目的环境隔离、ModelScope 固定 revision、语义校验和证据要求改写。
