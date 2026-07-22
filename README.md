# Qwen3-ASR vLLM 推理服务

面向 Linux + NVIDIA CUDA 的单 GPU 高吞吐语音识别服务。GPU 数据面使用原生 vLLM
Speech-to-Text；独立 aiohttp CPU 网关提供长音频切分、Base64 业务兼容接口、动态热词
Prompt、错误契约与健康/指标代理。v1.0.0 对外只暴露 `8080`。

```text
客户端 → aiohttp 网关 :8080
           ├─ ≤30 秒：原字节直送
           ├─ >30 秒：最长 30 秒 PCM16 WAV 分片、受控并发、顺序合并
           └─ /chinese_asr 协议转换与 Prompt 热词
       → vLLM :8081（仅 127.0.0.1）
       → Qwen3-ASR-0.6B / continuous batching / CUDA Graph
```

## v1.0.0 能力

| 能力 | 状态 |
|---|---|
| `/chinese_asr` Base64 JSON | 已实现 |
| `/v1/audio/transcriptions` multipart 文本转写 | 已实现；仅有限兼容，不宣称完整 OpenAI API |
| 0～300 秒音频、超过 30 秒固定切分 | 已实现 |
| 动态热词 | 已实现；Qwen3-ASR Prompt 软偏置，不保证强制命中 |
| 分片级时间边界 | 已实现；不是字级时间戳 |
| VAD / CT 标点 / ForcedAligner | 未实现，相关开关开启时拒绝启动 |
| 流式识别、说话人识别、字级时间戳 | 未实现 |

## 快速开始

必须使用独立环境，不与 Paraformer、funasr 或 TensorRT 推理栈混装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-infer.txt
python -m pip check
bash run.sh
```

首次启动仅从 ModelScope 下载 `Qwen/Qwen3-ASR-0.6B`，固定 revision：
`4ce9cc728b473a5aedbe7b6e1ea45646316824dc`。下载器校验文件大小与 SHA256，并在模型目录
写入 `.modelscope-manifest.json`。

启动后：

```bash
curl http://127.0.0.1:8080/health
python tests/test_chinese_asr_single.py --audio test_data/audio_16000_10s.wav
```

## 默认性能基线

混合在线流量默认 `CHUNK_CONCURRENCY=3`、`LONG_CHUNKS_IN_FLIGHT=64`；vLLM 活跃序列上限
为 128。A10 纯 90 秒测试的吞吐优先候选为 `3/96`（1537.09 audio_s/s），但不作为短音频
占 80% 场景的默认值。详细数据和复测方法见[性能调优](docs/性能调优.md)。

## 文档

- [使用说明](docs/使用说明.md)：环境、下载、启动、全部参数、调用和排障
- [API 文档](docs/API文档.md)：字段、输入输出、错误码、响应头和能力边界
- [技术文档](docs/技术文档.md)：技术路线、处理链路、并发模型与设计取舍
- [性能调优](docs/性能调优.md)：A10 数据、参数解释、测试方法和混合流量建议
- [发布验收](docs/发布验收.md)：静态检查、Linux/CUDA 验收、上线与回滚
- [修改日志](docs/修改日志.md)：v1.0.0 变更记录

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