#!/usr/bin/env bash
# CPU 音频切分网关：对外 :8080，转发到本机 vLLM :8081。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# 服务元数据与路由；该调试入口不启动 vLLM，也不下载模型。
export SERVICE_VERSION="${SERVICE_VERSION:-2.1.0}"  # /health 返回的版本字符串。
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
# 字节上限按 7500 秒 16 kHz 单声道 PCM16 WAV 推导：音频 229 MiB、Base64 JSON 请求体 308 MiB。
export AUDIO_CHUNK_SECONDS="${AUDIO_CHUNK_SECONDS:-32}"  # 长音频单个物理分片的最长秒数。
export MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-7500}"  # 解码后按采样帧计算的最大总秒数。
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-229}"  # 解码后音频或 multipart 文件上限，MiB。
export MAX_JSON_BODY_MB="${MAX_JSON_BODY_MB:-308}"  # 完整 Base64 JSON 请求体上限，MiB。
export CHUNK_CONCURRENCY="${CHUNK_CONCURRENCY:-3}"  # 单个长请求最多并发提交的分片数。
export LONG_CHUNKS_IN_FLIGHT="${LONG_CHUNKS_IN_FLIGHT:-96}"  # 全局长音频分片在途任务上限。
export BACKEND_TIMEOUT="${BACKEND_TIMEOUT:-300}"  # 每个 vLLM 分片 HTTP 请求超时秒数。
export BACKEND_CONNECTION_LIMIT="${BACKEND_CONNECTION_LIMIT:-100}"  # 网关到 vLLM 的连接池上限。
export BACKEND_KEEPALIVE_TIMEOUT="${BACKEND_KEEPALIVE_TIMEOUT:-4}"  # 客户端空闲连接保留秒数；低于 vLLM 关闭窗口。
export ENABLE_VAD="${ENABLE_VAD:-false}"  # 未实现；设为 true 时拒绝启动。

# 动态热词仅写入 Prompt；以下三项分别限制数量、单词长度和总字符数。
export ENABLE_HOTWORD="${ENABLE_HOTWORD:-true}"  # 是否将请求热词写入 Prompt。
export MAX_HOTWORDS="${MAX_HOTWORDS:-100}"  # 去空去重后的最大热词数量。
export MAX_HOTWORD_LENGTH="${MAX_HOTWORD_LENGTH:-64}"  # 单个热词最大 Unicode 字符数。
export MAX_HOTWORD_CHARS="${MAX_HOTWORD_CHARS:-1000}"  # 全部热词最大 Unicode 字符总数。

# ForcedAligner 默认后处理；worker 通过有界队列动态聚合跨请求物理分片。
# batch=1 是逐片回滚值；增大 batch 前必须验证显存、吞吐、尾延迟和时间戳契约。
export ENABLE_WORD_TIMESTAMP="${ENABLE_WORD_TIMESTAMP:-true}"  # 是否返回真实字/词级 words。
export ENABLE_SENTENCE_TIMESTAMP="${ENABLE_SENTENCE_TIMESTAMP:-true}"  # 是否按标点聚合真实句级边界。
export ALIGNER_MODEL_DIR="${ALIGNER_MODEL_DIR:-models/qwen3-forced-aligner-0.6b/pt}"  # Aligner 本地权重目录。
export ALIGNER_DEVICE="${ALIGNER_DEVICE:-cuda:0}"  # Aligner 执行设备，如 cuda:0 或 cpu。
export ALIGNER_DTYPE="${ALIGNER_DTYPE:-bfloat16}"  # Aligner 精度；CPU 必须使用 float32。
export ALIGNER_ATTENTION_BACKEND="${ALIGNER_ATTENTION_BACKEND:-auto}"  # auto 保持现状；显式后端失败时拒绝启动。
export ALIGNER_CONCURRENCY="${ALIGNER_CONCURRENCY:-1}"  # 并发模型 worker 数；共享模型需验证线程安全。
# batch=32 为目标 A10 实测最佳平衡；batch=1 保留为逐片回滚值。与 run.sh 保持一致。
export ALIGNER_BATCH_SIZE="${ALIGNER_BATCH_SIZE:-32}"  # 单次模型调用跨请求合并的最大物理分片数。
export ALIGNER_DECODE_WORKERS="${ALIGNER_DECODE_WORKERS:-1}"  # 每批 WAV 并行解码线程上限；1 保留串行回滚。
export ALIGNER_PREDECODE_ENABLED="${ALIGNER_PREDECODE_ENABLED:-false}"  # 是否在 ASR 阶段提前有界解码 PCM。
export ALIGNER_PREDECODE_MAX_MB="${ALIGNER_PREDECODE_MAX_MB:-128}"  # 全局已解码 PCM 驻留预算，MiB。
export ALIGNER_BATCH_WAIT_MS="${ALIGNER_BATCH_WAIT_MS:-5}"  # 首条分片入队后的最大动态合批等待毫秒数。
export ALIGNER_QUEUE_SIZE="${ALIGNER_QUEUE_SIZE:-256}"  # 有界对齐分片队列容量。

# MPS 是宿主级透明特性，只靠环境变量无法决定是否走 MPS。本调试入口不启动 vLLM，也不主动
# 改变守护进程状态（避免与并行运行的 run.sh 抢管理权），只声明状态并在不一致时拒绝启动。
export ENABLE_MPS="${ENABLE_MPS:-false}"  # 期望的 MPS 状态；不一致时报错，由 run.sh 负责调整。
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"  # MPS 命名管道目录。
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"  # MPS 控制日志目录。

mps_online=false
if command -v nvidia-cuda-mps-control >/dev/null 2>&1 \
  && [[ "$(echo get_server_list | nvidia-cuda-mps-control 2>&1)" != *"Cannot find MPS control daemon"* ]]; then
  mps_online=true
fi

case "${ENABLE_MPS,,}" in
  true|1|yes|on)
    if [[ "$mps_online" != "true" ]]; then
      echo "配置错误：ENABLE_MPS=true，但 MPS 控制守护进程不可用（$CUDA_MPS_PIPE_DIRECTORY）。" >&2
      echo "请先执行 nvidia-cuda-mps-control -d；守护进程缺失会导致吞吐静默下降且无告警。" >&2
      exit 1
    fi
    echo "MPS 状态：守护进程在线（$CUDA_MPS_PIPE_DIRECTORY），网关将作为 MPS 客户端运行。"
    ;;
  *)
    if [[ "$mps_online" == "true" ]]; then
      echo "配置错误：ENABLE_MPS=false，但 MPS 守护进程正在运行（$CUDA_MPS_PIPE_DIRECTORY）。" >&2
      echo "网关仍会作为 MPS 客户端运行，性能结果不可与无 MPS 基线比较。" >&2
      echo "请先停止守护进程：echo quit | nvidia-cuda-mps-control" >&2
      exit 1
    fi
    echo "MPS 状态：守护进程不在线，网关使用独立 CUDA 上下文。"
    ;;
esac

if [[ "$ALIGNER_ATTENTION_BACKEND" == "flash_attention_2" ]]; then
  if ! python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("flash_attn") else 1)'; then
    echo "配置错误：ALIGNER_ATTENTION_BACKEND=flash_attention_2，但当前固定环境未安装 flash_attn。" >&2
    echo "网关不会静默回退或运行时安装依赖；请恢复 auto/sdpa，或使用独立固定依赖镜像重新验收。" >&2
    exit 1
  fi
fi

exec python src/gateway.py
