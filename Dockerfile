# Qwen3-ASR 原生 vLLM 0.19.1 + aiohttp 网关运行镜像。
# 必须使用 CUDA 12.9 的 x86_64 基础镜像；cu130 变体需要 580+ 驱动，目标宿主为 535。
# 不包含 TensorRT、ONNX、VAD、CT 标点或 Faiss；模型、测试和验收数据从构建上下文打入镜像。
FROM zhxgharbor.istarshine.com/asr/vllm-openai:v0.19.1-x86_64

USER root
WORKDIR /qwen3asr_infer

# 构建/运行环境：非交互 apt、上海时区、UTF-8、Python 实时日志且不写 pyc。
# PYTHONPATH 使 sitecustomize.py 自动生效；两个 OFFLINE 禁止 Hugging Face 运行期联网回退。
# 容器内以 root 安装固定依赖，因此关闭 root 和 pip 新版本提示，构建日志保持可读。
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/qwen3asr_infer/src \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 补齐运行库；私有基础镜像若通过 apt 预装 blinker 1.4，则先由包管理器完整移除。
RUN apt-get update \
    && if dpkg-query -W python3-blinker >/dev/null 2>&1; then \
         apt-get remove -y python3-blinker; \
       fi \
    && apt-get install -y --no-install-recommends \
       ca-certificates libsndfile1 python3-cairo tini \
    && PYTHON_BIN="$(command -v python3.12 || command -v python3)" \
    && test -n "$PYTHON_BIN" \
    && ln -sf "$PYTHON_BIN" /usr/local/bin/python \
    && python --version \
    && rm -rf /var/lib/apt/lists/*

# pip 清华源加速
RUN python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

COPY requirements-infer.txt requirements-aligner.txt ./
# 构建阶段参数：仅精确字符串 true 安装 qwen-asr/ForcedAligner；发布镜像默认安装。
ARG INSTALL_ALIGNER=true
# 默认 Aligner 镜像先覆盖任何残留的 distutils blinker，再安装 qwen-asr 完整固定依赖。
RUN python -m pip install --no-cache-dir -r requirements-infer.txt \
    && if [ "$INSTALL_ALIGNER" = "true" ]; then \
         python -m pip install --no-cache-dir --ignore-installed blinker==1.9.0 \
         && python -c "import importlib.metadata as m; assert m.version('blinker') == '1.9.0'" \
         && python -m pip install --no-cache-dir -r requirements-aligner.txt; \
       fi \
    && python -m pip check \
    && python -c "import importlib.metadata as m; assert m.version('vllm') == '0.19.1'; assert m.version('transformers') == '4.57.6'" \
    && python -c "import torch; assert torch.version.cuda.startswith('12.'), f'需要 CUDA 12.x 构建的 PyTorch，当前为 {torch.version.cuda}'" \
    && if [ "$INSTALL_ALIGNER" = "true" ]; then \
         python -c "import importlib.metadata as m; from qwen_asr import Qwen3ForcedAligner; assert m.version('qwen-asr') == '0.0.6'; assert m.version('blinker') == '1.9.0'"; \
       fi

COPY models models
COPY src/ src/
COPY tests tests
COPY test_data test_data
COPY scripts/download_model.py scripts/download_model.py
COPY run.sh run_gateway.sh ./
RUN chmod +x run.sh run_gateway.sh

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "run.sh"]
