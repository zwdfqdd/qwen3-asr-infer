# Qwen3-ASR vLLM 推理服务

面向 Linux + NVIDIA CUDA 的单 GPU 高吞吐语音识别服务。GPU 数据面使用原生 vLLM
Speech-to-Text；独立 aiohttp CPU 网关提供长音频切分、Base64 业务兼容接口、动态热词
Prompt、错误契约与健康/指标代理。v1.1.0 对外只暴露 `8080`。

```text
客户端 → aiohttp 网关 :8080
           ├─ ≤32 秒：原字节直送
           ├─ >32 秒：最长 32 秒 PCM16 WAV 分片、受控并发、顺序合并
           ├─ /chinese_asr 协议转换与 Prompt 热词
           └─ 默认 ForcedAligner 字/词级及句级时间戳
       → vLLM :8081（仅 127.0.0.1）
       → Qwen3-ASR-0.6B / continuous batching / CUDA Graph
```

## 当前版本能力（v1.1.0）

| 能力 | 状态 |
|---|---|
| `/chinese_asr` Base64 JSON | 已实现 |
| `/v1/audio/transcriptions` multipart 文本转写 | 已实现；仅有限兼容，不宣称完整 OpenAI API |
| 0～2000 秒音频、超过 32 秒固定切分 | 已实现 |
| 动态热词 | 已实现；Qwen3-ASR Prompt 软偏置，不保证强制命中 |
| 主模型标点 | 已实现；直接使用 Qwen3-ASR 输出，不使用 CT-Transformer |
| 主模型语种识别 | 已实现；30 种语言与 22 种中国方言/口音，逐物理分片填充 `slid` |
| 分片级时间边界 | 始终提供；关闭或跳过对齐时不是字/词级时间戳 |
| ForcedAligner 字/词级及句级时间戳 | 已实现；仅 `/chinese_asr`、仅 11 种语言，默认开启 |
| 结构化请求追踪 | 已实现；保留统计摘要，并记录受限 `input_json`/`result_json`；输入 Base64 只保留前 64 字节 |
| VAD、流式 HTTP 网关、说话人识别 | 未实现；说话人识别需独立模型 |

## 语种与方言支持

### Qwen3-ASR 主模型

主模型支持以下 **30 种语言**。`/chinese_asr` 的 `language` 可使用英文名称或括号中的 ISO
代码显式指定；不传或传空值时由模型逐物理分片自动检测：

- Chinese（中文，`zh`）、English（英语，`en`）、Cantonese（粤语，`yue`）、Arabic（阿拉伯语，`ar`）
- German（德语，`de`）、French（法语，`fr`）、Spanish（西班牙语，`es`）、Portuguese（葡萄牙语，`pt`）
- Indonesian（印度尼西亚语，`id`）、Italian（意大利语，`it`）、Korean（韩语，`ko`）、Russian（俄语，`ru`）
- Thai（泰语，`th`）、Vietnamese（越南语，`vi`）、Japanese（日语，`ja`）、Turkish（土耳其语，`tr`）
- Hindi（印地语，`hi`）、Malay（马来语，`ms`）、Dutch（荷兰语，`nl`）、Swedish（瑞典语，`sv`）
- Danish（丹麦语，`da`）、Finnish（芬兰语，`fi`）、Polish（波兰语，`pl`）、Czech（捷克语，`cs`）
- Filipino（菲律宾语，`fil`）、Persian（波斯语，`fa`）、Greek（希腊语，`el`）、Hungarian（匈牙利语，`hu`）
- Macedonian（马其顿语，`mk`）、Romanian（罗马尼亚语，`ro`）

主模型还声明支持以下 **22 种中国方言/口音的自动识别**；这些标签可能出现在响应 `slid`，
但不能作为当前接口的显式 `language` 参数：

- Anhui（安徽）、Dongbei（东北）、Fujian（福建）、Gansu（甘肃）、Guizhou（贵州）
- Hebei（河北）、Henan（河南）、Hubei（湖北）、Hunan（湖南）、Jiangxi（江西）
- Ningxia（宁夏）、Shandong（山东）、Shaanxi（陕西）、Shanxi（山西）、Sichuan（四川）
- Tianjin（天津）、Yunnan（云南）、Zhejiang（浙江）
- Cantonese (Hong Kong accent)（粤语香港口音）
- Cantonese (Guangdong accent)（粤语广东口音）
- Wu language（吴语）、Minnan language（闽南语）

### Qwen3 ForcedAligner

ForcedAligner 只支持以下 **11 种标准语言**及对应别名：

- Chinese（中文，`zh`）、English（英语，`en`）、Cantonese（粤语，`yue`）
- French（法语，`fr`）、German（德语，`de`）、Italian（意大利语，`it`）
- Japanese（日语，`ja`）、Korean（韩语，`ko`）、Portuguese（葡萄牙语，`pt`）
- Russian（俄语，`ru`）、Spanish（西班牙语，`es`）

Aligner 不直接接受上述 22 种方言/口音标签。分片 `slid` 为空或不是这 11 种标准标签时，服务
不会猜测、映射或回退中文，而是保留 ASR 文本与物理分片边界、返回 `words=[]`，并在顶层
`message` 中列出跳过的分片和语种。

## 快速开始

必须使用独立环境，不与 Paraformer、funasr 或 TensorRT 推理栈混装。默认开启 ForcedAligner，
因此推理和 Aligner 两套精确依赖都必须安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-infer.txt
python -m pip install -r requirements-aligner.txt
python -m pip check
bash run.sh
```

`run.sh` 会仅从 ModelScope 检查或下载主模型和 ForcedAligner 的固定 revision。若要关闭时间戳，
必须显式设置两个运行开关：

```bash
ENABLE_WORD_TIMESTAMP=false ENABLE_SENTENCE_TIMESTAMP=false bash run.sh
```

Dockerfile 默认 `INSTALL_ALIGNER=true`，并执行 `COPY models models`、`COPY tests tests`、
`COPY test_data test_data`。构建前应确保两个模型目录均包含固定 revision 权重及 manifest：

```bash
python scripts/download_model.py
python scripts/download_model.py \
  --model-id Qwen/Qwen3-ForcedAligner-0.6B \
  --dir models/qwen3-forced-aligner-0.6b/pt \
  --revision cf1c50164ea3ac48240d12bef5ead74aee0720cc

docker build --no-cache --build-arg INSTALL_ALIGNER=true \
  -t qwen3-asr-infer:0.16.0 .
docker run -d --gpus '"device=0"' --restart=always \
  -p 8080:8080 qwen3-asr-infer:0.16.0
```

主模型固定 revision 为 `4ce9cc728b473a5aedbe7b6e1ea45646316824dc`。下载器校验文件大小
与 SHA256，并在模型目录写入 `.modelscope-manifest.json`；镜像启动时仍会执行完整性校验。

`/chinese_asr` 不传 `language` 时由主模型逐物理分片自动检测并填入 `slid`。主模型官方范围
是 30 种语言与 22 种中国方言/口音；默认 ForcedAligner 仅支持其中 11 种语言。支持的分片
执行真实对齐；Malay、Arabic 等不受支持的分片仍随整条请求返回 HTTP 200，保留 ASR 文本、
`slid` 和物理分片边界，但 `words=[]`，顶层 `message` 明确列出跳过项。服务不会伪造时间戳
或回退中文。完整列表和部分对齐契约见 [API 文档](docs/API文档.md)。

启动后：

```bash
curl http://127.0.0.1:8080/health
python tests/test_chinese_asr_single.py --audio test_data/audio_16000_10s.wav
```

## 默认输入与性能基线

当前默认 `AUDIO_CHUNK_SECONDS=32`、`MAX_AUDIO_SECONDS=2000`、`MAX_UPLOAD_MB=280`、
`MAX_JSON_BODY_MB=280`。Base64 会膨胀约 1/3，因此 280 MiB JSON 请求体只能承载约 210 MiB
原始音频；multipart 仍由 280 MiB 请求体和音频上限共同约束。时长与两层字节限制分别保护
解码后音频、HTTP 接收和进程内存。

当前默认 `CHUNK_CONCURRENCY=3`、`LONG_CHUNKS_IN_FLIGHT=96`。历史 A10 纯 90 秒测试中
`3/96` 吞吐最高；仍须在 0～2000 秒真实混合流量和默认同卡 Aligner 模式下复测显存、吞吐及
P95/P99。详细数据和方法见[性能调优](docs/性能调优.md)。

## 文档

- [使用说明](docs/使用说明.md)：环境、下载、启动、全部参数、调用和排障
- [API 文档](docs/API文档.md)：字段、输入输出、错误码、响应头和能力边界
- [技术文档](docs/技术文档.md)：技术路线、处理链路、并发模型与设计取舍
- [性能调优](docs/性能调优.md)：A10 数据、参数解释、测试方法和混合流量建议
- [发布验收](docs/发布验收.md)：静态检查、Linux/CUDA 验收、上线与回滚
- [修改日志](docs/修改日志.md)：v1.1.0 变更及 v1.0.0 历史记录

## 测试

```bash
python tests/test_chinese_asr_single.py --audio test_data/audio_16000_10s.wav
python tests/verify_qwen_single.py --api-mode chinese-asr \
  --input test_data --ref test_data --baseline-cer 0.05
python tests/test_service.py --api-mode chinese-asr \
  --audio test_data/audio_16000_30s.wav --concurrency 96 --total 2000
```

CER 门禁会拒绝缺失或空参考，不会以零计分样本误放行。生产发布还必须在目标 Linux/CUDA
环境完成 [发布验收](docs/发布验收.md)。

docker run -it --gpus '"device=0"' --restart=always -p30960:8080 zhxgharbor.istarshine.com/asr/qwen3-asr-infer:0.16.0
docker run -it --gpus '"device=1"' --restart=always -p30961:8080 zhxgharbor.istarshine.com/asr/qwen3-asr-infer:0.16.0