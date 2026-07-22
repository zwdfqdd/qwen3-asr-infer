#!/usr/bin/env bash
# 一次启动 Qwen3-ASR vLLM 后端和对外 CPU 音频切分网关。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# PORT 是容器/平台常见变量，不能用作内部端口；显式清除，避免污染子进程。
if [[ -n "${PORT:-}" ]]; then
  echo "提示：忽略宿主环境 PORT=$PORT；请使用 VLLM_PORT 或 GATEWAY_PORT 配置端口。"
  unset PORT
fi
export SERVICE_VERSION="${SERVICE_VERSION:-1.0.0}"
export VLLM_PORT="${VLLM_PORT:-8081}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
export VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-models/qwen3-asr-0.6b/vllm}"
export MODELSCOPE_REVISION="${MODELSCOPE_REVISION:-4ce9cc728b473a5aedbe7b6e1ea45646316824dc}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-asr}"
export DTYPE="${DTYPE:-bfloat16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="${VLLM_MAX_AUDIO_CLIP_FILESIZE_MB:-64}"
export VLLM_ATTENTION_BACKEND_NAME="${VLLM_ATTENTION_BACKEND_NAME:-FLASH_ATTN}"
export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:$VLLM_PORT}"
export AUDIO_CHUNK_SECONDS="${AUDIO_CHUNK_SECONDS:-30}"
export MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-300}"
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-64}"
export CHUNK_CONCURRENCY="${CHUNK_CONCURRENCY:-3}"
export LONG_CHUNKS_IN_FLIGHT="${LONG_CHUNKS_IN_FLIGHT:-64}"
export BACKEND_TIMEOUT="${BACKEND_TIMEOUT:-300}"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
export MAX_JSON_BODY_MB="${MAX_JSON_BODY_MB:-96}"
export ENABLE_HOTWORD="${ENABLE_HOTWORD:-true}"
export ENABLE_VAD="${ENABLE_VAD:-false}"
export ENABLE_WORD_TIMESTAMP="${ENABLE_WORD_TIMESTAMP:-false}"
export ENABLE_SENTENCE_TIMESTAMP="${ENABLE_SENTENCE_TIMESTAMP:-false}"
export MAX_HOTWORDS="${MAX_HOTWORDS:-100}"
export MAX_HOTWORD_LENGTH="${MAX_HOTWORD_LENGTH:-64}"
export MAX_HOTWORD_CHARS="${MAX_HOTWORD_CHARS:-1000}"

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

python -c 'import importlib, importlib.metadata as m; version=m.version("vllm"); assert version == "0.16.0", f"需要 vllm==0.16.0，当前为 {version}"; importlib.import_module("vllm.transformers_utils.configs.qwen3_asr")' || {
  echo "当前环境不兼容；请在独立虚拟环境执行: python -m pip install -r requirements-infer.txt" >&2
  exit 1
}
python scripts/download_model.py \
  --model-id "$VLLM_MODEL_ID" \
  --dir "$VLLM_MODEL_DIR" \
  --revision "$MODELSCOPE_REVISION"
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
  --dtype "$DTYPE" --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --attention-config.backend "$VLLM_ATTENTION_BACKEND_NAME" \
  --enable-chunked-prefill --generation-config vllm --disable-log-requests &
VLLM_PID=$!
DEADLINE=$((SECONDS + VLLM_STARTUP_TIMEOUT))
until python -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1],timeout=2)' "$BACKEND_URL/health" >/dev/null 2>&1; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM 启动失败" >&2; exit 1; }
  (( SECONDS < DEADLINE )) || { echo "等待 vLLM 就绪超时（${VLLM_STARTUP_TIMEOUT}s）" >&2; exit 1; }
  sleep 2
done

echo "=== vLLM 已就绪，启动对外网关 $GATEWAY_HOST:$GATEWAY_PORT ==="
python src/gateway.py &
GATEWAY_PID=$!
set +e
wait -n "$VLLM_PID" "$GATEWAY_PID"
STATUS=$?
set -e
exit "$STATUS"
