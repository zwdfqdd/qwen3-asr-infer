#!/usr/bin/env bash
# 一次启动 Qwen3-ASR vLLM 后端和对外 CPU 音频切分网关。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# 保留容器原始 stdout 给网关 JSON；run.sh、下载器和 vLLM 统一继承 stderr。
exec 3>&1
exec 1>&2

# 让 Python 启动时自动加载 src/sitecustomize.py 的固定版本兼容修复。
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# PORT 是容器/平台常见变量，不能用作内部端口；显式清除，避免污染子进程。
if [[ -n "${PORT:-}" ]]; then
  echo "提示：忽略宿主环境 PORT=$PORT；请使用 VLLM_PORT 或 GATEWAY_PORT 配置端口。"
  unset PORT
fi
# 服务元数据与端口：8080 是唯一对外端口，8081 仅供本机网关访问。
export SERVICE_VERSION="${SERVICE_VERSION:-2.1.0}"  # /health 返回的版本字符串。
export VLLM_PORT="${VLLM_PORT:-8081}"  # vLLM 内部监听端口，范围 1～65535。
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"  # vLLM 监听地址；默认禁止外部绕过网关。
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-asr}"  # vLLM 对外模型别名。
export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"  # aiohttp 网关监听地址。
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"  # 客户端访问的网关端口，范围 1～65535。
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:$VLLM_PORT}"  # 网关访问 vLLM 的 HTTP 基址。
# 结构化日志：请求线程只入有界队列；stdout 始终输出，文件日志按大小轮转。
export LOG_LEVEL="${LOG_LEVEL:-INFO}"  # DEBUG/INFO/WARNING/ERROR/CRITICAL。
export LOG_FILE_ENABLED="${LOG_FILE_ENABLED:-true}"  # 是否额外写入本地轮转文件。
export LOG_DIR="${LOG_DIR:-logs}"  # 网关文件日志目录。
export LOG_MAX_FILE_MB="${LOG_MAX_FILE_MB:-200}"  # 单个 gateway.log 文件上限，MiB。
export LOG_BACKUP_COUNT="${LOG_BACKUP_COUNT:-7}"  # 最多保留的大小轮转备份数。
export LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-7}"  # 启动时清理超过该天数的旧日志。
export LOG_QUEUE_SIZE="${LOG_QUEUE_SIZE:-10000}"  # 异步日志队列容量；满时丢弃并上报计数。

# ASR 模型下载：仅从 ModelScope 固定 revision 下载到独立本地目录。
export VLLM_MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"  # ModelScope 主模型仓库 ID。
export VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-models/qwen3-asr-0.6b/vllm}"  # 主模型持久化目录。
export MODELSCOPE_REVISION="${MODELSCOPE_REVISION:-4ce9cc728b473a5aedbe7b6e1ea45646316824dc}"  # 固定提交 revision。

# vLLM 引擎与调度：精度/权重格式、显存比例、token/序列预算、音频文件保护及启动超时。
export DTYPE="${DTYPE:-bfloat16}"  # GPU 计算精度；A10 默认 bfloat16。
export VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-safetensors}"  # 本地权重加载格式。
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"  # GPU 显存占用比例，范围 (0,1)。
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-10240}"  # 单序列最大上下文 token 数。
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"  # scheduler 最大活跃序列数，不是固定 batch。
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-10240}"  # 单轮调度 token 总预算。
# vLLM 多模态加载器接受的单个音频剪辑文件上限，单位 MiB；它不限制整条网关请求时长。
# 长音频通常已被网关切成最长 AUDIO_CHUNK_SECONDS 秒的独立 WAV，再逐片交给 vLLM。
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="${VLLM_MAX_AUDIO_CLIP_FILESIZE_MB:-96}"
export VLLM_ATTENTION_BACKEND_NAME="${VLLM_ATTENTION_BACKEND_NAME:-FLASH_ATTN}"  # 多模态 attention 后端。
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"  # 等待 /health 就绪的总秒数。

# 网关音频与请求限制。大小单位为 MiB，时长单位为秒，并发项单位为任务数。
# 字节上限按 7500 秒 16 kHz 单声道 PCM16 WAV 推导：240,000,044 B 约 228.88 MiB → 229 MiB；
# Base64 膨胀 4/3 后约 305.18 MiB，加 JSON 字段与热词开销 → 308 MiB。三项互相绑定。
export AUDIO_CHUNK_SECONDS="${AUDIO_CHUNK_SECONDS:-32}"  # 长音频单个物理分片的最长秒数。
export MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-7500}"  # 解码后按采样帧计算的最大总秒数。
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-229}"  # 解码后音频或 multipart 文件上限，MiB。
export MAX_JSON_BODY_MB="${MAX_JSON_BODY_MB:-308}"  # 完整 Base64 JSON 请求体上限，MiB。
export CHUNK_CONCURRENCY="${CHUNK_CONCURRENCY:-3}"  # 单个长请求最多并发提交的分片数。
export LONG_CHUNKS_IN_FLIGHT="${LONG_CHUNKS_IN_FLIGHT:-96}"  # 全局长音频分片在途任务上限。
export BACKEND_TIMEOUT="${BACKEND_TIMEOUT:-300}"  # 每个 vLLM 分片 HTTP 请求超时秒数。
export BACKEND_CONNECTION_LIMIT="${BACKEND_CONNECTION_LIMIT:-100}"  # 网关到 vLLM 的连接池上限。
export BACKEND_KEEPALIVE_TIMEOUT="${BACKEND_KEEPALIVE_TIMEOUT:-4}"  # 客户端空闲连接保留秒数；低于 vLLM 关闭窗口。
export ENABLE_VAD="${ENABLE_VAD:-false}"  # 预留开关；当前设为 true 会拒绝启动。

# 动态热词：仅作为 Qwen3-ASR Prompt 软偏置，不保证强制命中。
export ENABLE_HOTWORD="${ENABLE_HOTWORD:-true}"  # 是否将请求热词写入 Prompt。
export MAX_HOTWORDS="${MAX_HOTWORDS:-100}"  # 去空去重后的最大热词数量。
export MAX_HOTWORD_LENGTH="${MAX_HOTWORD_LENGTH:-64}"  # 单个热词最大 Unicode 字符数。
export MAX_HOTWORD_CHARS="${MAX_HOTWORD_CHARS:-1000}"  # 全部热词最大 Unicode 字符总数。

# ForcedAligner：worker 从有界队列动态聚合跨请求分片；batch=1 可回滚为逐片调用。
# 单 A10 首轮保持单 worker；batch/max-wait 必须在目标机按显存、吞吐和尾延迟验收。
export ENABLE_WORD_TIMESTAMP="${ENABLE_WORD_TIMESTAMP:-true}"  # 是否返回真实字/词级 words。
export ENABLE_SENTENCE_TIMESTAMP="${ENABLE_SENTENCE_TIMESTAMP:-true}"  # 是否按标点聚合真实句级边界。
export ALIGNER_MODEL_ID="${ALIGNER_MODEL_ID:-Qwen/Qwen3-ForcedAligner-0.6B}"  # ModelScope Aligner 仓库 ID。
export ALIGNER_MODEL_DIR="${ALIGNER_MODEL_DIR:-models/qwen3-forced-aligner-0.6b/pt}"  # Aligner 本地权重目录。
export ALIGNER_MODELSCOPE_REVISION="${ALIGNER_MODELSCOPE_REVISION:-cf1c50164ea3ac48240d12bef5ead74aee0720cc}"  # 固定提交 revision。
export ALIGNER_DEVICE="${ALIGNER_DEVICE:-cuda:0}"  # Aligner 执行设备，如 cuda:0 或 cpu。
export ALIGNER_DTYPE="${ALIGNER_DTYPE:-bfloat16}"  # Aligner 精度；CPU 必须使用 float32。
export ALIGNER_ATTENTION_BACKEND="${ALIGNER_ATTENTION_BACKEND:-auto}"  # auto 保持 Transformers 自动选择；可实验 sdpa/flash_attention_2。
export ALIGNER_CONCURRENCY="${ALIGNER_CONCURRENCY:-1}"  # 并发模型 worker 数；共享模型需验证线程安全。
export ALIGNER_BATCH_SIZE="${ALIGNER_BATCH_SIZE:-1}"  # 单次模型调用跨请求合并的最大物理分片数。
export ALIGNER_DECODE_WORKERS="${ALIGNER_DECODE_WORKERS:-1}"  # 每批 WAV 并行解码线程上限；1 保留串行回滚。
export ALIGNER_PREDECODE_ENABLED="${ALIGNER_PREDECODE_ENABLED:-false}"  # 是否在 ASR 阶段提前有界解码 PCM。
export ALIGNER_PREDECODE_MAX_MB="${ALIGNER_PREDECODE_MAX_MB:-128}"  # 全局已解码 PCM 驻留预算，MiB。
export ALIGNER_BATCH_WAIT_MS="${ALIGNER_BATCH_WAIT_MS:-5}"  # 首条分片入队后的最大动态合批等待毫秒数。
export ALIGNER_QUEUE_SIZE="${ALIGNER_QUEUE_SIZE:-256}"  # 有界对齐分片队列容量。

# 参数：待判断的布尔文本；true/1/yes/on（不区分大小写）视为开启。
enabled() {
  case "${1,,}" in
    true|1|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if enabled "$ENABLE_WORD_TIMESTAMP" || enabled "$ENABLE_SENTENCE_TIMESTAMP"; then
  if [[ "$ALIGNER_ATTENTION_BACKEND" == "flash_attention_2" ]]; then
    if ! python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("flash_attn") else 1)'; then
      echo "配置错误：ALIGNER_ATTENTION_BACKEND=flash_attention_2，但当前固定环境未安装 flash_attn。" >&2
      echo "服务不会静默回退或运行时安装依赖；请恢复 auto/sdpa，或使用独立固定依赖镜像重新验收。" >&2
      exit 1
    fi
  fi
  if [[ "$ALIGNER_DEVICE" == "cuda:0" ]]; then
    echo "警告：单卡双进程启用 ForcedAligner；vLLM 显存预算=$VLLM_GPU_MEMORY_UTILIZATION，Aligner=$ALIGNER_DEVICE/$ALIGNER_DTYPE。"
    echo "该模式仅用于低流量试验；若 OOM，请降低 VLLM_GPU_MEMORY_UTILIZATION 或改用独立 GPU/CPU。"
  fi
fi

# 参数：监听主机、端口、中文服务名；通过临时 bind 验证启动前端口未被占用。
check_port_available() {
  local host="$1"
  local port="$2"
  local service="$3"
  if ! python -c 'import socket,sys; host=sys.argv[1]; family=socket.AF_INET6 if ":" in host else socket.AF_INET; sock=socket.socket(family); sock.bind((host,int(sys.argv[2]))); sock.close()' "$host" "$port" 2>/dev/null; then
    echo "$service 端口已被占用：$host:$port" >&2
    echo "请先定位旧进程：ss -ltnp \"sport = :$port\"" >&2
    echo "也可使用：lsof -nP -iTCP:$port -sTCP:LISTEN" >&2
    echo "确认进程后正常终止，再重新执行 bash run.sh；不要直接使用 kill -9。" >&2
    exit 1
  fi
}

if [[ "$VLLM_PORT" == "$GATEWAY_PORT" ]]; then
  echo "配置错误：VLLM_PORT 与 GATEWAY_PORT 不能相同（当前均为 $VLLM_PORT）" >&2
  exit 1
fi
check_port_available "$VLLM_HOST" "$VLLM_PORT" "vLLM 后端"
check_port_available "$GATEWAY_HOST" "$GATEWAY_PORT" "对外网关"

python -c 'import importlib, importlib.metadata as m; version=m.version("vllm"); assert version == "0.19.1", f"需要 vllm==0.19.1，当前为 {version}"; importlib.import_module("vllm.transformers_utils.configs.qwen3_asr")' || {
  echo "当前环境不兼容；请在独立虚拟环境执行: python -m pip install -r requirements-infer.txt" >&2
  exit 1
}
# CUDA 12.x 构建校验：cu130 的 PyTorch 需要 580+ 驱动，在 535 驱动上会在 EngineCore 初始化时失败。
python -c 'import torch; cuda=torch.version.cuda; assert cuda and cuda.startswith("12."), f"需要 CUDA 12.x 构建的 PyTorch，当前为 {cuda}"; torch.cuda.init()' || {
  echo "PyTorch 与宿主 NVIDIA 驱动不兼容；请使用 CUDA 12.9 基础镜像，或将宿主驱动升级到 580 以上。" >&2
  exit 1
}
python scripts/download_model.py \
  --model-id "$VLLM_MODEL_ID" \
  --dir "$VLLM_MODEL_DIR" \
  --revision "$MODELSCOPE_REVISION"
if enabled "$ENABLE_WORD_TIMESTAMP" || enabled "$ENABLE_SENTENCE_TIMESTAMP"; then
  python -c 'import importlib.metadata as m; from qwen_asr import Qwen3ForcedAligner; assert m.version("qwen-asr") == "0.0.6"' || {
    echo "时间戳功能已开启，但镜像未安装 qwen-asr==0.0.6。" >&2
    echo "Docker 请使用 --build-arg INSTALL_ALIGNER=true 重新构建；本机环境请安装 requirements-aligner.txt。" >&2
    exit 1
  }
  python scripts/download_model.py \
    --model-id "$ALIGNER_MODEL_ID" \
    --dir "$ALIGNER_MODEL_DIR" \
    --revision "$ALIGNER_MODELSCOPE_REVISION"
fi
MODEL_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$VLLM_MODEL_DIR")"
VLLM_PID=""
GATEWAY_PID=""
cleanup() {
  trap - EXIT INT TERM
  for pid in "$GATEWAY_PID" "$VLLM_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
  done
  [[ -z "$GATEWAY_PID" ]] || wait "$GATEWAY_PID" 2>/dev/null || true
  [[ -z "$VLLM_PID" ]] || wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "=== 启动 vLLM 后端 $VLLM_HOST:$VLLM_PORT（model=$MODEL_DIR）==="
vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --dtype "$DTYPE" --load-format "$VLLM_LOAD_FORMAT" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --attention-config.backend "$VLLM_ATTENTION_BACKEND_NAME" \
  --enable-chunked-prefill --generation-config vllm &
VLLM_PID=$!
DEADLINE=$((SECONDS + VLLM_STARTUP_TIMEOUT))
until python -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1],timeout=2)' "$BACKEND_URL/health" >/dev/null 2>&1; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM 启动失败" >&2; exit 1; }
  (( SECONDS < DEADLINE )) || { echo "等待 vLLM 就绪超时（${VLLM_STARTUP_TIMEOUT}s）" >&2; exit 1; }
  sleep 2
done

echo "=== vLLM 已就绪，启动对外网关 $GATEWAY_HOST:$GATEWAY_PORT ==="
# 仅网关 stdout 恢复到容器原始 stdout，便于采集器按严格 JSON 解析。
python src/gateway.py >&3 3>&- &
GATEWAY_PID=$!
set +e
wait -n "$VLLM_PID" "$GATEWAY_PID"
STATUS=$?
set -e
exit "$STATUS"


#  docker run -it --gpus '"device=0"' --restart=always -p30960:8080 zhxgharbor.istarshine.com/asr/qwen3-asr-infer:0.19.1
#  docker run -it --gpus '"device=1"' --restart=always -p30961:8080 zhxgharbor.istarshine.com/asr/qwen3-asr-infer:0.19.1