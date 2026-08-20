"""PERF-ALI-011：ForcedAligner `align()` 内部七段拆分计时。

**本实验只做测量，不做优化。** 不修改 `src/aligner.py`、不改生产默认路径、不触碰磁盘权重。

立项依据来自 PERF-ALI-010 的反推：batch=32 时 `align()` 926.41 ms 中约 256 ms（约 28%）
不在 GPU 上。生产阶段计时把这段并入了 `aligner_batch_model_call_ms`，因为该指标包裹的是
整个 `self._model.align()` 调用，而 `aligner_batch_audio_decode_ms`（146.93 ms）只统计网关
自己的 WAV 解码，与 processor 的特征提取是两件事。

官方 `align()` 的实际构成（读 qwen_asr 0.0.6 源码得到）：

```python
audios = normalize_audios(audio)                       # 1 CPU
word_list, input_text = encode_timestamp(t, lang)      # 2 CPU 分词
inputs = self.processor(text=..., audio=..., padding=True)  # 3 CPU 特征提取
inputs = inputs.to(device).to(dtype)                   # 4 H2D
logits = self.model.thinker(**inputs).logits; argmax    # 5 GPU
timestamp_ms = masked.to("cpu").numpy()                # 6 D2H
parse_timestamp(word_list, timestamp_ms)               # 7 CPU 后处理
```

第 7 段内的 `fix_timestamp()` 是纯 Python 的 O(n²) 最长非降子序列 DP，n 为对齐单元数的两倍。
因此本实验必须把它与第 3 段的 mel 特征提取分开计时，二者都是 256 ms 的候选主体，
合并测量无法区分。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

# 只使用本地固定 revision 权重，禁止运行期联网回退。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import soundfile as sf

_EXPECTED_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
_EXPECTED_REVISION = "cf1c50164ea3ac48240d12bef5ead74aee0720cc"

# 七段固定顺序，输出与占比均按此排列，便于跨轮次对照。
_STAGES = (
    "normalize_audio",
    "encode_timestamp",
    "feature_extract",
    "host_to_device",
    "thinker_forward",
    "device_to_host",
    "parse_timestamp",
)

# 稳定性判据。沿用 PERF-ALI-010 的教训：分母不稳时收益与占比结论都不可用。
#
# 最初用 max/min > 1.2 判定，实测连续两轮误报：`normalize_audio` 出现 min 4.02、p50 4.46、
# max 9.94，只是一次外部抢占，却让整组数据被标为不可用。max/min 由两个单点决定，在共享的
# 256 核机器上必然偶发离群，因此改为两条稳健判据：
#   * p95/p50 反映主体分布离散度，对单点离群不敏感；
#   * 前后半段中位数漂移捕捉系统性偏移，实测退化轮为 +931.5%，干净轮为 +1.8%，区分度充足。
_DISPERSION_LIMIT = 1.3
_DRIFT_LIMIT_PERCENT = 20.0

# 七段之和与官方 align() 全程的允许偏差；超过说明复刻漏段或计时口径有误。
_UNEXPLAINED_LIMIT_PERCENT = 5.0


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch size 必须是逗号分隔整数") from error
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("batch size 必须为不重复的正整数")
    return sizes


def _load_audio(path: Path) -> tuple[np.ndarray, float]:
    with sf.SoundFile(path) as source:
        if source.samplerate != 16000:
            raise ValueError(f"只接受 16 kHz 音频，实际为 {source.samplerate} Hz")
        if source.channels != 1 or source.frames <= 0:
            raise ValueError("只接受非空单声道音频")
        audio = source.read(dtype="float32", always_2d=False)
        duration = source.frames / source.samplerate
    if duration > 32.0:
        raise ValueError("Aligner 单片音频不得超过 32 秒")
    return np.ascontiguousarray(audio), duration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_identity(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / ".modelscope-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 ModelScope manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {
        "repo_id": manifest.get("repo_id"),
        "revision": manifest.get("revision"),
    }
    if identity != {"repo_id": _EXPECTED_REPO, "revision": _EXPECTED_REVISION}:
        raise RuntimeError(f"Aligner 模型身份不符合固定值：{identity}")
    return identity


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stats(values: list[float]) -> dict[str, Any]:
    low, high = min(values), max(values)
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "min": round(low, 3),
        "max": round(high, 3),
        # max/min 由两个单点决定，一次外部抢占就会让它失真，因此只作参考信息。
        "max_over_min": round(high / low, 3) if low > 0 else float("inf"),
        # p95/p50 反映主体分布的离散度，对单点离群稳健，用作稳定性判据。
        "p95_over_p50": (
            round(_percentile(values, 0.95) / _percentile(values, 0.50), 3)
            if _percentile(values, 0.50) > 0
            else float("inf")
        ),
        # 首轮实测 feature_extract 出现 max/min=10.5，而 warmup 已为 15、thinker 段稳定在
        # 1.03，说明不是预热问题。仅凭统计量无法区分长尾抢占、双峰和随迭代单调恶化，
        # 因此保留每次迭代原始值：iterations 量级为几十，数据量可忽略。
        "samples": [round(value, 3) for value in values],
    }


def _trend(values: list[float]) -> dict[str, Any]:
    """比较前后半段均值，用于区分“随迭代恶化”与“随机长尾”。

    首轮实测 CPU 段极差比达 10 倍，两种成因的处置方式完全不同：随迭代单调恶化指向进程内
    状态累积（内存碎片、分配器行为），随机长尾指向外部抢占。统计量看不出差别，趋势可以。
    """
    half = len(values) // 2
    if half == 0:
        return {
            "first_half_p50": None,
            "second_half_p50": None,
            "drift_percent": None,
        }
    # 用中位数而非均值：均值不抗离群，单个长尾尖峰落在后半段就会被误判成单调恶化，
    # 而这两种成因的处置方式完全不同。中位数对单点尖峰稳健，仍能反映整体抬升。
    first = statistics.median(values[:half])
    second = statistics.median(values[half:])
    return {
        "first_half_p50": round(first, 3),
        "second_half_p50": round(second, 3),
        "drift_percent": round((second - first) / first * 100, 2) if first > 0 else None,
    }


def _unstable(named: dict[str, dict[str, Any]]) -> list[str]:
    """返回测量不可用于定量判定的段名及原因。

    判据为 p95/p50 离散度或前后半段中位数漂移，二者任一超限即判定不稳。极短的段（亚毫秒）
    相对计时误差天然偏大，因此只对均值达到 1 ms 的段判定。

    ``named`` 的每项需同时含 `_stats()` 的字段与 `trend` 子字典。
    """
    flagged: list[str] = []
    for name, item in named.items():
        if item["mean"] < 1.0:
            continue
        reasons = []
        if item["p95_over_p50"] > _DISPERSION_LIMIT:
            reasons.append(f"p95/p50={item['p95_over_p50']}")
        drift = (item.get("trend") or {}).get("drift_percent")
        if drift is not None and abs(drift) > _DRIFT_LIMIT_PERCENT:
            reasons.append(f"漂移={drift:+.1f}%")
        if reasons:
            flagged.append(f"{name}({', '.join(reasons)})")
    return flagged


def _rss_mb() -> float | None:
    """当前进程 RSS，用于判断退化是否伴随内存增长。psutil 缺失时返回 None。"""
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _units(results: Any) -> list[list[dict[str, Any]]]:
    """把官方 ForcedAlignResult 列表转成可比对的纯数据结构。"""
    return [
        [
            {
                "text": item.text,
                "start": round(float(item.start_time), 3),
                "end": round(float(item.end_time), 3),
            }
            for item in result.items
        ]
        for result in results
    ]


def _assert_same_units(
    baseline: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
) -> None:
    """复刻流程必须与官方 align() 结果完全一致，否则拆分出的占比没有意义。

    这里不设容差：两者执行的是同一串运算，任何差异都说明复刻错了，而不是精度抖动。
    """
    if len(baseline) != len(candidate):
        raise RuntimeError("复刻流程与官方 align() 的样本数量不一致")
    for sample, (left, right) in enumerate(zip(baseline, candidate)):
        if len(left) != len(right):
            raise RuntimeError(f"第 {sample} 个样本对齐单元数量不一致")
        for index, (a, b) in enumerate(zip(left, right)):
            if a != b:
                raise RuntimeError(
                    f"第 {sample} 个样本第 {index} 个单元不一致：{a} 对 {b}"
                )


def _staged_align(
    aligner: Any,
    align_audios: list[Any],
    texts: list[str],
    languages: list[str],
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """逐段复刻官方 align()，返回结果、七段耗时与输入张量形状。

    复刻必须与 `qwen_asr/inference/qwen3_forced_aligner.py` 的 `align()` 逐行等价，
    只在段间插入计时点。任何简化都会让占比失真，因此不省略 round 与结构体构建，
    并直接调用官方私有的 `_to_structured_items` 以保证结果对象构造方式相同。

    有意省略官方的三处入参护栏：`ensure_list()`（本函数已传入等长列表，为恒等变换）、
    语种广播和 batch 长度一致性检查。三者都是常数级判断，省略后应体现为“未归因”接近零；
    若报告中 `unexplained_percent` 显著为正，说明确实漏了实质段落而不是这三处。

    ``align_audios`` 是 ``(np.ndarray, sr)`` 元组列表：官方 ``normalize_audios`` 只接受
    路径、URL、base64 或该元组，不接受裸 ndarray。
    """
    import torch
    from qwen_asr.inference.utils import normalize_audios

    stage: dict[str, float] = {}
    with torch.inference_mode():
        started = perf_counter()
        audios = normalize_audios(align_audios)
        stage["normalize_audio"] = (perf_counter() - started) * 1000

        started = perf_counter()
        word_lists = []
        aligner_input_texts = []
        for text, language in zip(texts, languages):
            word_list, aligner_input_text = aligner.aligner_processor.encode_timestamp(
                text, language
            )
            word_lists.append(word_list)
            aligner_input_texts.append(aligner_input_text)
        stage["encode_timestamp"] = (perf_counter() - started) * 1000

        # 第 3 段：mel 特征提取与文本 tokenize，PERF-ALI-010 反推出的主要嫌疑。
        started = perf_counter()
        inputs = aligner.processor(
            text=aligner_input_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        stage["feature_extract"] = (perf_counter() - started) * 1000

        shapes = {
            key: list(value.shape)
            for key, value in inputs.items()
            if hasattr(value, "shape")
        }

        started = perf_counter()
        inputs = inputs.to(aligner.model.device).to(aligner.model.dtype)
        torch.cuda.synchronize()
        stage["host_to_device"] = (perf_counter() - started) * 1000

        started = perf_counter()
        logits = aligner.model.thinker(**inputs).logits
        output_ids = logits.argmax(dim=-1)
        torch.cuda.synchronize()
        stage["thinker_forward"] = (perf_counter() - started) * 1000

        # 第 6/7 段：官方在同一个循环里做 D2H 与 parse_timestamp，这里按样本分两次累加，
        # 因为 fix_timestamp 的 O(n²) DP 与设备同步开销必须分开归因。
        device_to_host_ms = 0.0
        parse_ms = 0.0
        results = []
        for input_id, output_id, word_list in zip(
            inputs["input_ids"], output_ids, word_lists
        ):
            started = perf_counter()
            masked_output_id = output_id[input_id == aligner.timestamp_token_id]
            timestamp_ms = (
                masked_output_id * aligner.timestamp_segment_time
            ).to("cpu").numpy()
            device_to_host_ms += (perf_counter() - started) * 1000

            started = perf_counter()
            timestamp_output = aligner.aligner_processor.parse_timestamp(
                word_list, timestamp_ms
            )
            for item in timestamp_output:
                item["start_time"] = round(item["start_time"] / 1000.0, 3)
                item["end_time"] = round(item["end_time"] / 1000.0, 3)
            results.append(aligner._to_structured_items(timestamp_output))
            parse_ms += (perf_counter() - started) * 1000

        stage["device_to_host"] = device_to_host_ms
        stage["parse_timestamp"] = parse_ms

    return results, stage, shapes


def _time_staged(
    aligner: Any,
    align_audios: list[Any],
    texts: list[str],
    languages: list[str],
    warmup: int,
    iterations: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float], Any, dict[str, Any]]:
    """多轮执行拆分流程，返回各段统计、复刻全程统计、结果与输入形状。"""
    import torch

    for _ in range(warmup):
        _staged_align(aligner, align_audios, texts, languages)
    torch.cuda.synchronize()

    collected: dict[str, list[float]] = {name: [] for name in _STAGES}
    totals: list[float] = []
    # 实测 feature_extract 出现 +931.5% 的随迭代恶化，需要判断是否伴随内存增长与
    # 是否与运行时长相关，因此逐次记录 RSS 与自循环开始的累计秒数。
    rss_series: list[float | None] = []
    elapsed_series: list[float] = []
    results = None
    shapes: dict[str, Any] = {}
    loop_started = perf_counter()
    for _ in range(iterations):
        started = perf_counter()
        results, stage, shapes = _staged_align(aligner, align_audios, texts, languages)
        totals.append((perf_counter() - started) * 1000)
        rss_series.append(_rss_mb())
        elapsed_series.append(round(perf_counter() - loop_started, 3))
        for name in _STAGES:
            collected[name].append(stage[name])
    return (
        {name: _stats(values) for name, values in collected.items()},
        _stats(totals),
        results,
        {"input_shapes": shapes, "rss_mb": rss_series, "elapsed_s": elapsed_series},
    )


def _time_feature_extract_only(
    aligner: Any,
    raw_audios: list[Any],
    aligner_input_texts: list[str],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """隔离测量：只反复调用 processor，循环内不含任何 GPU 工作。

    这是判定 `feature_extract` 退化成因的单变量测试。若隔离后仍然恶化，成因在 CPU 侧
    （分配器、内存增长、线程池状态）；若隔离后稳定，则退化来自与 GPU 工作交替执行的交互，
    两者的处置方式完全不同。
    """
    for _ in range(warmup):
        aligner.processor(
            text=aligner_input_texts,
            audio=raw_audios,
            return_tensors="pt",
            padding=True,
        )

    latencies: list[float] = []
    rss_series: list[float | None] = []
    for _ in range(iterations):
        started = perf_counter()
        aligner.processor(
            text=aligner_input_texts,
            audio=raw_audios,
            return_tensors="pt",
            padding=True,
        )
        latencies.append((perf_counter() - started) * 1000)
        rss_series.append(_rss_mb())
    return {
        "latency_ms": _stats(latencies),
        "trend": _trend(latencies),
        "rss_mb": rss_series,
    }


def _time_official_align(
    aligner: Any,
    align_audios: list[Any],
    texts: list[str],
    languages: list[str],
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], Any]:
    """测量官方 align() 全程，作为拆分流程的参照与正确性基准。

    口径与生产日志的 ``aligner_batch_model_call_ms`` 一致，可直接对照 780.82 ms。
    """
    import torch

    for _ in range(warmup):
        aligner.align(audio=align_audios, text=texts, language=languages)
    torch.cuda.synchronize()

    latencies: list[float] = []
    results = None
    for _ in range(iterations):
        started = perf_counter()
        results = aligner.align(audio=align_audios, text=texts, language=languages)
        torch.cuda.synchronize()
        latencies.append((perf_counter() - started) * 1000)
    return _stats(latencies), results


def _load_aligner(args: argparse.Namespace) -> Any:
    import torch
    from qwen_asr import Qwen3ForcedAligner

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model_kwargs: dict[str, Any] = {
        "device_map": args.device,
        "dtype": dtype,
        "local_files_only": True,
    }
    if args.attention != "auto":
        model_kwargs["attn_implementation"] = args.attention
    return Qwen3ForcedAligner.from_pretrained(str(args.model_dir), **model_kwargs)


def _cpu_environment(aligner: Any) -> dict[str, Any]:
    """记录影响 CPU 段耗时的线程与实现路径信息。

    `Qwen3ASRProcessor` 把音频交给 Transformers 的 `WhisperFeatureExtractor`。该类同时提供
    numpy 与 torch 两条 fbank 实现，默认走 numpy 逐样本循环；只有显式传入非 cpu 的 `device`
    才改走 torch 批量 STFT。首轮实测 `feature_extract` 随批量超线性增长且极差达 10 倍，
    因此必须把线程数与实际可用路径记录下来，否则无法判断成因。
    """
    import torch

    extractor = getattr(aligner.processor, "feature_extractor", None)
    thread_env = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
        )
    }
    return {
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "thread_env": thread_env,
        "feature_extractor_class": type(extractor).__name__ if extractor else None,
        # 存在该方法说明可通过 device 参数把 mel 提取搬到 GPU；属后续优化线索，本实验只记录。
        "has_torch_fbank_path": hasattr(extractor, "_torch_extract_fbank_features"),
        "has_numpy_fbank_path": hasattr(extractor, "_np_extract_fbank_features"),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 CUDA 设备")
    cuda_version = torch.version.cuda or ""
    if not cuda_version.startswith("12."):
        raise RuntimeError(
            f"需要 CUDA 12.x 构建的 PyTorch，当前为 {cuda_version}；"
            "目标 A10 宿主驱动为 535，不支持 CUDA 13.0"
        )

    args.model_dir = Path(args.model).resolve()
    identity = _model_identity(args.model_dir)
    audio_path = Path(args.audio).resolve()
    audio, duration = _load_audio(audio_path)
    text = (
        Path(args.text).read_text(encoding="utf-8").strip()
        if args.text_is_file
        else args.text
    )
    if not text:
        raise ValueError("对齐文本不能为空")

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "PERF-ALI-011",
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": cuda_version,
            "qwen_asr": importlib.metadata.version("qwen-asr"),
            "transformers": importlib.metadata.version("transformers"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "model": {"path": str(args.model_dir), **identity},
        "input": {
            "filename": audio_path.name,
            "sha256": _sha256(audio_path),
            "duration_seconds": duration,
            "language": args.language,
            "text_chars": len(text),
        },
        "parameters": {
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "dtype": args.dtype,
            "device": args.device,
            "attention": args.attention,
            "isolate_feature_extract": args.isolate_feature_extract,
        },
        "stage_order": list(_STAGES),
        "results": [],
    }

    aligner = _load_aligner(args)
    actual_attention = getattr(
        getattr(getattr(aligner, "model", None), "config", None),
        "_attn_implementation",
        None,
    )
    report["runtime"]["attention_actual"] = str(actual_attention)
    report["cpu_environment"] = _cpu_environment(aligner)
    print(f"模型已加载：attention={actual_attention}，dtype={args.dtype}")
    print(
        f"CPU 环境：核心 {report['cpu_environment']['cpu_count']}，"
        f"torch 线程 {report['cpu_environment']['torch_num_threads']}，"
        f"特征提取器 {report['cpu_environment']['feature_extractor_class']}"
        f"（torch 路径可用={report['cpu_environment']['has_torch_fbank_path']}）"
    )

    for batch_size in args.batch_sizes:
        align_audios = [(audio, 16000)] * batch_size
        texts = [text] * batch_size
        languages = [args.language] * batch_size

        official, official_results = _time_official_align(
            aligner, align_audios, texts, languages, args.warmup, args.iterations
        )
        stages, staged_total, staged_results, diag = _time_staged(
            aligner, align_audios, texts, languages, args.warmup, args.iterations
        )

        # 复刻正确性优先于任何性能数字：结果不一致则拆分出的占比毫无意义。
        _assert_same_units(_units(official_results), _units(staged_results))

        stage_sum = sum(stages[name]["mean"] for name in _STAGES)
        unexplained = staged_total["mean"] - stage_sum
        unexplained_percent = (
            unexplained / staged_total["mean"] * 100 if staged_total["mean"] > 0 else 0.0
        )
        cpu_stages = (
            "normalize_audio",
            "encode_timestamp",
            "feature_extract",
            "parse_timestamp",
        )
        cpu_sum = sum(stages[name]["mean"] for name in cpu_stages)

        units_per_sample = len(_units(staged_results)[0])
        item = {
            "batch_size": batch_size,
            "official_align_ms": official,
            "staged_total_ms": staged_total,
            "stages_ms": stages,
            "stage_share_percent": {
                name: round(stages[name]["mean"] / stage_sum * 100, 2)
                for name in _STAGES
            },
            "stage_sum_ms": round(stage_sum, 3),
            "unexplained_ms": round(unexplained, 3),
            "unexplained_percent": round(unexplained_percent, 2),
            # 与 PERF-ALI-010 的 256 ms 反推值对照：该值应落在同一量级。
            "cpu_total_ms": round(cpu_sum, 3),
            "cpu_share_percent": round(cpu_sum / stage_sum * 100, 2),
            "gpu_total_ms": round(stages["thinker_forward"]["mean"], 3),
            # 复刻与官方全程的偏差，用于确认插入计时点本身没有显著开销。
            "staged_overhead_percent": round(
                (staged_total["mean"] - official["mean"]) / official["mean"] * 100, 2
            ),
            # 按样本归一：首轮 feature_extract 为 3.96/11.21/17.61 ms/样本（batch 1/8/32），
            # 随批量上升说明其随批量超线性增长；thinker 同期为 39.98/22.12/21.09，是正常的
            # 亚线性。两者对照可判断超线性只出现在 CPU 段。
            "stage_ms_per_sample": {
                name: round(stages[name]["mean"] / batch_size, 3) for name in _STAGES
            },
            "stage_trend": {
                name: _trend(stages[name]["samples"]) for name in _STAGES
            },
            # 各段取 min 的下界估算。实测 feature_extract 随迭代恶化十倍，均值被污染，
            # 而 min 更接近未受干扰的真实成本；该估算用于与生产 model_call 780.82 ms 对照，
            # 判断退化是 spike 紧密循环的产物还是生产同样存在的问题。
            "stage_min_sum_ms": round(
                sum(stages[name]["min"] for name in _STAGES), 3
            ),
            "cpu_min_sum_ms": round(sum(stages[name]["min"] for name in cpu_stages), 3),
            "rss_mb_series": diag["rss_mb"],
            "elapsed_s_series": diag["elapsed_s"],
            "input_shapes": diag["input_shapes"],
            "units_per_sample": units_per_sample,
            # fix_timestamp 是 O(n²) DP，n 为对齐单元数的两倍；记录 n 便于验证平方关系。
            "fix_timestamp_n": units_per_sample * 2,
            "units_identical": True,
        }
        if args.isolate_feature_extract:
            from qwen_asr.inference.utils import normalize_audios

            raw_audios = normalize_audios(align_audios)
            aligner_input_texts = [
                aligner.aligner_processor.encode_timestamp(one_text, one_language)[1]
                for one_text, one_language in zip(texts, languages)
            ]
            item["feature_extract_isolated"] = _time_feature_extract_only(
                aligner, raw_audios, aligner_input_texts, args.warmup, args.iterations
            )

        # 稳定性判定需要统计量与趋势合并，因为漂移是判据之一。
        verdict_input: dict[str, dict[str, Any]] = {
            name: {**stages[name], "trend": item["stage_trend"][name]}
            for name in _STAGES
        }
        item["staged_total_trend"] = _trend(staged_total["samples"])
        item["official_trend"] = _trend(official["samples"])
        verdict_input["staged_total"] = {
            **staged_total, "trend": item["staged_total_trend"]
        }
        verdict_input["official"] = {**official, "trend": item["official_trend"]}
        unstable = _unstable(verdict_input)
        item["measurement_unstable"] = unstable or None
        item["stage_sum_mismatch"] = (
            abs(unexplained_percent) > _UNEXPLAINED_LIMIT_PERCENT
        )
        report["results"].append(item)

        print(
            f"\nbatch={batch_size}  官方 align {official['mean']:.2f} ms  "
            f"复刻 {staged_total['mean']:.2f} ms  "
            f"（复刻开销 {item['staged_overhead_percent']:+.2f}%）"
        )
        for name in _STAGES:
            drift = item["stage_trend"][name]["drift_percent"]
            drift_text = "漂移 n/a" if drift is None else f"漂移 {drift:+.1f}%"
            print(
                f"  {name:<18} {stages[name]['mean']:>9.2f} ms  "
                f"{item['stage_share_percent'][name]:>5.2f}%  "
                f"{item['stage_ms_per_sample'][name]:>8.2f} ms/样本  "
                f"min {stages[name]['min']:>8.2f}  p50 {stages[name]['p50']:>8.2f}  "
                f"max {stages[name]['max']:>8.2f}  "
                f"p95/p50={stages[name]['p95_over_p50']:<7}"
                f"max/min={stages[name]['max_over_min']:<8}{drift_text}"
            )
        print(
            f"  {'七段合计':<16} {stage_sum:>9.2f} ms  "
            f"未归因 {unexplained:+.2f} ms（{unexplained_percent:+.2f}%）"
        )
        print(
            f"  CPU 合计 {cpu_sum:.2f} ms（{item['cpu_share_percent']:.2f}%）  "
            f"GPU {item['gpu_total_ms']:.2f} ms  "
            f"对齐单元 {units_per_sample} 个 / fix_timestamp n={item['fix_timestamp_n']}"
        )
        print(
            f"  取各段 min 的下界：七段合计 {item['stage_min_sum_ms']:.2f} ms"
            f"（CPU {item['cpu_min_sum_ms']:.2f} ms）  "
            f"生产 aligner_batch_model_call_ms 为 780.82 ms，可据此判断退化是否存在于生产"
        )
        rss = [value for value in diag["rss_mb"] if value is not None]
        if rss:
            print(
                f"  RSS {rss[0]:.1f} → {rss[-1]:.1f} MiB"
                f"（峰值 {max(rss):.1f}，增长 {rss[-1] - rss[0]:+.1f}）"
            )
        if args.isolate_feature_extract:
            isolated = item["feature_extract_isolated"]
            drift = isolated["trend"]["drift_percent"]
            print(
                f"  隔离测量 feature_extract（循环内无 GPU 工作）："
                f"mean {isolated['latency_ms']['mean']:.2f} ms  "
                f"min {isolated['latency_ms']['min']:.2f}  "
                f"p50 {isolated['latency_ms']['p50']:.2f}  "
                f"max {isolated['latency_ms']['max']:.2f}  "
                f"max/min={isolated['latency_ms']['max_over_min']}  "
                f"漂移 {'n/a' if drift is None else f'{drift:+.1f}%'}"
            )
            print(
                "        隔离后仍恶化说明成因在 CPU 侧（分配器、内存增长、线程池状态）；"
                "隔离后稳定说明退化来自与 GPU 工作交替执行的交互。"
            )
        if item["stage_sum_mismatch"]:
            print(
                f"  警告：七段合计与复刻全程相差 {unexplained_percent:+.2f}%，"
                f"超过 {_UNEXPLAINED_LIMIT_PERCENT}%，说明存在未计时的段落，"
                "占比结论不可用。"
            )
        if unstable:
            print(
                f"  警告：以下段测量不稳定（{', '.join(unstable)}），"
                f"超过 p95/p50>{_DISPERSION_LIMIT} 或漂移>{_DRIFT_LIMIT_PERCENT}% 限制，"
                "该组的绝对值与占比不得用于定量判定。"
            )
            print(
                "        漂移超限说明随迭代恶化（进程内状态累积或外部负载渐增）；"
                "仅 p95/p50 超限说明主体分布本身离散。"
                "报告的 stages_ms[*].samples 保留了每次迭代原始值。"
            )
            print(
                "        注意：加大 --warmup 只能排除预热，对已排除预热的不稳定无效；"
                "先确认 CPU 侧无其他负载，再用 --isolate-feature-extract 区分"
                "CPU 侧成因与 GPU 交替执行的交互。"
            )
        else:
            print("  稳定性判定通过，本组绝对值与占比可用于判定。")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="拆分 ForcedAligner align() 内部七段耗时，定位非 GPU 时间去向"
    )
    parser.add_argument(
        "--model", default="models/qwen3-forced-aligner-0.6b/pt",
        help="ModelScope 固定 revision 的本地 Aligner 目录",
    )
    parser.add_argument("--audio", required=True, help="16 kHz 单声道、最长 32 秒音频")
    parser.add_argument("--language", required=True, help="官方语言名称，如 Chinese")
    parser.add_argument(
        "--text", required=True,
        help="与音频内容对应的参考文本；配合 --text-is-file 时为文本文件路径",
    )
    parser.add_argument(
        "--text-is-file", action="store_true",
        help="把 --text 当作文件路径读取，便于直接使用同名参考文本",
    )
    parser.add_argument(
        "--batch-sizes", type=_parse_batch_sizes,
        default=_parse_batch_sizes("1,8,32"),
        help="逗号分隔的 batch size，默认 1,8,32；生产对照值为 32",
    )
    # 默认与 PERF-ALI-010 一致：warmup=5 时 batch=32 的极差曾达 2.3 倍，均值虚高约 54%。
    parser.add_argument("--warmup", type=int, default=15, help="每组预热次数；不足会使均值虚高")
    parser.add_argument("--iterations", type=int, default=30, help="每组正式执行次数")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0", help="Aligner 执行设备")
    parser.add_argument(
        "--attention", choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto", help="与生产默认一致；目标机 auto 实际选择 sdpa",
    )
    parser.add_argument(
        "--isolate-feature-extract", action="store_true",
        help="额外只循环 processor 调用（循环内无 GPU 工作），"
             "用于判定 feature_extract 的随迭代恶化来自 CPU 侧还是与 GPU 交替执行的交互",
    )
    parser.add_argument("--output-json", help="原子写入实验报告")
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup 不能小于 0，--iterations 必须大于 0")
    if args.output_json and not args.output_json.endswith(".json"):
        parser.error("--output-json 路径必须以 .json 结尾")

    report = _run(args)
    if args.output_json:
        _write_json(Path(args.output_json), report)
        print(f"\n报告已写入：{args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
