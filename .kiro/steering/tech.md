# 技术栈与常用命令

- Python 3.10+；目标平台 Linux + NVIDIA CUDA。
- GPU 推理：`vllm[audio]==0.16.0` + `transformers==4.57.6`，原生 Speech-to-Text。
- CPU 网关：aiohttp 3.13.3 + soundfile 0.13.1。
- 测试：aiohttp、soundfile、psutil；GPU 采样使用 `nvidia-smi`。
- 必须使用独立 venv/镜像，不与 Paraformer、funasr、TensorRT 或 Transformers 5.x 混装。

系统配置集中在 `src/config.py` 的模块级 `settings`，部署入口通过 `run.sh` 环境变量覆盖。
模型只从 ModelScope 下载，固定 revision，不使用 Hugging Face 链路。

```bash
python -m pip install -r requirements-infer.txt
python scripts/download_model.py
bash run.sh
python tests/test_chinese_asr_single.py --audio test_data/audio_16000_10s.wav
python tests/verify_qwen_single.py --api-mode chinese-asr --input test_data --ref test_data --baseline-cer 0.05
python tests/test_service.py --api-mode chinese-asr --audio test_data/audio_16000_30s.wav --concurrency 96 --total 2000
```

约定：每张 GPU 只运行一个 vLLM；网关不做固定攒批；日志/注释/文档用中文；依赖精确固定；
未实现的 VAD、CT 标点和 ForcedAligner 不得伪造或静默降级。