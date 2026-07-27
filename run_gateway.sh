#!/usr/bin/env bash
# CPU 音频切分网关：对外 :8080，转发到本机 vLLM :8081。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# 服务元数据与路由；该调试入口不启动 vLLM，也不下载模型。
export SERVICE_VERSION="${SERVICE_VERSION:-1.1.0}"  # /health 返回的版本字符串。
export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"  # aiohttp 网关监听地址。
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"  # 客户端访问的网关端口。
export VLLM_PORT="${VLLM_PORT:-8081}"  # 已独立启动的 vLLM 端口。
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:$VLLM_PORT}"  # 网关访问 vLLM 的 HTTP 基址。
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-asr}"  # 网关提交的 vLLM 模型别名。
# 结构化日志：请求线程只入有界队列；stdout 始终输出，文件日志按大小轮转。
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export LOG_FILE_ENABLED="${LOG_FILE_ENABLED:-true}"
export LOG_DIR="${LOG_DIR:-logs}"
export LOG_MAX_FILE_MB="${LOG_MAX_FILE_MB:-200}"
export LOG_BACKUP_COUNT="${LOG_BACKUP_COUNT:-7}"
export LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-7}"
export LOG_QUEUE_SIZE="${LOG_QUEUE_SIZE:-10000}"

# 音频与请求限制：大小单位 MiB，时长/超时单位秒，并发项单位为在途任务数。
# 280 MiB JSON 会因 Base64 约 1/3 膨胀而先于同值音频上限触发。
export AUDIO_CHUNK_SECONDS="${AUDIO_CHUNK_SECONDS:-32}"  # 长音频单个物理分片的最长秒数。
export MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-2000}"  # 解码后按采样帧计算的最大总秒数。
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-280}"  # 解码后音频或 multipart 文件上限，MiB。
export MAX_JSON_BODY_MB="${MAX_JSON_BODY_MB:-280}"  # 完整 Base64 JSON 请求体上限，MiB。
export CHUNK_CONCURRENCY="${CHUNK_CONCURRENCY:-3}"  # 单个长请求最多并发提交的分片数。
export LONG_CHUNKS_IN_FLIGHT="${LONG_CHUNKS_IN_FLIGHT:-96}"  # 全局长音频分片在途任务上限。
export BACKEND_TIMEOUT="${BACKEND_TIMEOUT:-300}"  # 每个 vLLM 分片 HTTP 请求超时秒数。
export ENABLE_VAD="${ENABLE_VAD:-false}"  # 未实现；设为 true 时拒绝启动。

# 动态热词仅写入 Prompt；以下三项分别限制数量、单词长度和总字符数。
export ENABLE_HOTWORD="${ENABLE_HOTWORD:-true}"  # 是否将请求热词写入 Prompt。
export MAX_HOTWORDS="${MAX_HOTWORDS:-100}"  # 去空去重后的最大热词数量。
export MAX_HOTWORD_LENGTH="${MAX_HOTWORD_LENGTH:-64}"  # 单个热词最大 Unicode 字符数。
export MAX_HOTWORD_CHARS="${MAX_HOTWORD_CHARS:-1000}"  # 全部热词最大 Unicode 字符总数。

# ForcedAligner 默认后处理；运行前需安装 requirements-aligner.txt 并准备本地模型。
# 单 A10 同卡保持 cuda:0、bfloat16、并发 1、batch 1。
export ENABLE_WORD_TIMESTAMP="${ENABLE_WORD_TIMESTAMP:-true}"  # 是否返回真实字/词级 words。
export ENABLE_SENTENCE_TIMESTAMP="${ENABLE_SENTENCE_TIMESTAMP:-true}"  # 是否按标点聚合真实句级边界。
export ALIGNER_MODEL_DIR="${ALIGNER_MODEL_DIR:-models/qwen3-forced-aligner-0.6b/pt}"  # Aligner 本地权重目录。
export ALIGNER_DEVICE="${ALIGNER_DEVICE:-cuda:0}"  # Aligner 执行设备，如 cuda:0 或 cpu。
export ALIGNER_DTYPE="${ALIGNER_DTYPE:-bfloat16}"  # Aligner 精度；CPU 必须使用 float32。
export ALIGNER_CONCURRENCY="${ALIGNER_CONCURRENCY:-1}"  # 同时进入 Aligner 的任务数。
export ALIGNER_BATCH_SIZE="${ALIGNER_BATCH_SIZE:-1}"  # 单次模型调用包含的物理分片数。

exec python src/gateway.py
