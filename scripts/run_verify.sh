#!/usr/bin/env bash
# 原生 vLLM 服务的模型准备、健康检查、冒烟和 CER 验证。
set -euo pipefail
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# DO_INSTALL：仅值 1 时在验收前安装精确依赖；设为 0 可复用已准备环境。
if [[ "${DO_INSTALL:-1}" == "1" ]]; then
  python -m pip install -r requirements-infer.txt
fi
# DO_DOWNLOAD：仅值 1 时在验收前校验/下载固定 ModelScope 模型。
if [[ "${DO_DOWNLOAD:-1}" == "1" ]]; then
  python scripts/download_model.py
fi

curl --fail --silent http://127.0.0.1:8080/health >/dev/null || {
  echo "服务未运行。请执行 bash run.sh，脚本会统一启动 vLLM 后端和切分网关。" >&2
  exit 1
}

python tests/verify_qwen_single.py --api-mode chinese-asr --input test_data --limit 3
python tests/verify_qwen_single.py \
  --api-mode chinese-asr \
  --input test_data \
  --ref test_data \
  --baseline-cer 0.05
