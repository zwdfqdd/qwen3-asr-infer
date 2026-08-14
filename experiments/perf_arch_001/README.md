# PERF-ARCH-001：vLLM token-classify ForcedAligner spike

主服务当前使用单 Python 环境 `vllm[audio]==0.19.1`（CUDA 12.9），时间戳由
`qwen-asr==0.0.6` 的 PyTorch ForcedAligner 提供。本目录只验证后续把该后端替换为 vLLM
pooling / token-classification 的语义正确性与原始吞吐。

`src/aligner.py`、网关和生产启动入口**尚未集成**该后端。本 spike 通过不代表生产已支持。

时间戳技术路线已定稿并暂停对齐后端替换，依据见 [性能调优](../../docs/性能调优.md) 第 11.5
节；本实验属保留路线 L5，恢复条件见 [性能实验台账](../../docs/性能实验台账.md)。

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
├── requirements.txt
├── spike.py
└── README.md
```

## 1. 环境

实验依赖只复用根 Aligner 依赖并补充 spike 直接导入的精确版本：

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

不得使用生产目录 `models/qwen3-forced-aligner-0.6b/pt`。`spike.py` 会读取
`.modelscope-manifest.json`，仓库或 revision 不一致时直接失败。

## 3. 准备 words

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

## 4. 执行

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

## 5. 参数

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

## 6. 缓存抑制（结果可信度的前提）

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

## 7. 校验与晋级条件

`spike.py` 内置校验：模型身份、vLLM 版本、16 kHz 单声道、最长 32 秒、timestamp 数量等于
单元数 × 2、官方 `fix_timestamp()` 等价的非递减修复、严格单调、`start <= end` 且不越音频
边界。允许 `start == end` 的零宽单元，这与官方修复后的输出一致。

晋级条件：

- 11 种语言的单元文本、数量与最终时间戳全部通过 PyTorch 基线对照。
- batch=32 原始中位吞吐相对同环境 PyTorch 至少 +30%。
- 成功率 100%，无 OOM、NaN、死锁或静默降级。
- 未达标即停止，不创建 HTTP 服务、不修改网关。
- 达标后才进入 Phase C：独立服务、网关适配、三轮 2000 请求、端到端 +20%、混合时长、CER
  与故障隔离。

## 8. 已有结论

- 同环境 PyTorch 分母：满批 32 片 × 10 秒下 `aligner_batch_model_call_ms=780.82`，
  即纯 `model.align()` 约 409.8 audio_s/s，含 WAV 解码约 344.9。
- spike batch=32 为 1180.51 audio_s/s，同口径 **+188.1%**，Phase B 在“单语言、10 秒、单片”
  范围内通过。
- 尚未完成：Phase A 逐单元语义对照、11 种语言、非 10 秒时长、图模式与 float16 单变量、
  Phase C 集成。
- 注意端到端与原始吞吐不等价：PyTorch 前向从 1097.04 ms 优化到 780.82 ms（-28.8%）时，
  端到端仅 +3.82%。端到端收益主要受进程拓扑约束，集成前需先解决拓扑问题。

`spike.py` 不采集 GPU 指标，正式证据需同时运行：

```bash
nvidia-smi dmon -i 0 -s pucm -d 1 \
  > performance-results/PERF-ARCH-001/gpu-dmon.txt
```

## 9. 实现依据

- [vLLM 0.19.0 Token Classification 用法](https://docs.vllm.ai/en/v0.19.0/models/pooling_models/token_classify/)
- [vLLM 官方 forced alignment 示例源码](https://github.com/vllm-project/vllm/blob/main/examples/pooling/token_classify/forced_alignment_offline.py)
- [Qwen3 ForcedAligner vLLM 模型实现](https://docs.vllm.ai/en/v0.19.0/api/vllm/model_executor/models/qwen3_asr_forced_aligner/)

官方资料内容已转述，并按本项目的环境隔离、ModelScope 固定 revision、语义校验和证据要求改写。
