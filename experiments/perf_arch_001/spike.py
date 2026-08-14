#!/usr/bin/env python3
"""PERF-ARCH-001：Qwen3 ForcedAligner vLLM token-classify 语义与吞吐 spike。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

# 候选只允许读取本地 ModelScope 固定 revision，禁止运行期联网回退。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import soundfile as sf

# 输入构造与时间桶提取参考 Apache-2.0 vLLM 官方示例（已按本实验改写）：
# https://github.com/vllm-project/vllm/blob/main/examples/pooling/token_classify/forced_alignment_offline.py
_ARCHITECTURE = "Qwen3ASRForcedAlignerForTokenClassification"
_EXPECTED_VLLM_VERSION = "0.19.1"
_EXPECTED_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
_EXPECTED_REVISION = "cf1c50164ea3ac48240d12bef5ead74aee0720cc"


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch size 必须是逗号分隔整数") from error
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("batch size 必须为不重复的正整数")
    return sizes


def _parse_words(value: str) -> list[str]:
    try:
        words = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("--words-json 必须是 JSON 字符串数组") from error
    if not isinstance(words, list) or not words or any(
        not isinstance(word, str) or not word for word in words
    ):
        raise argparse.ArgumentTypeError("--words-json 必须是非空字符串数组")
    return words


def _load_audio(path: Path) -> tuple[np.ndarray, int, float]:
    with sf.SoundFile(path) as source:
        if source.samplerate != 16000:
            raise ValueError(f"首轮只接受 16 kHz 音频，实际为 {source.samplerate} Hz")
        if source.channels != 1 or source.frames <= 0:
            raise ValueError("首轮只接受非空单声道音频")
        audio = source.read(dtype="float32", always_2d=False)
        duration = source.frames / source.samplerate
    if duration > 32.0:
        raise ValueError("Phase A/B 单片音频不得超过 32 秒")
    return np.ascontiguousarray(audio), 16000, duration


def _build_prompt(words: list[str]) -> str:
    body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


def _batch_audio(audio: np.ndarray, index: int, distinct: bool) -> np.ndarray:
    """为批内第 index 个请求生成音频。

    同一批重复完全相同的 prompt 与音频时，vLLM 的前缀与多模态缓存可能让实际计算次数
    远少于 batch size，使原始吞吐虚高。``distinct=True`` 时对单个采样点施加 1e-6 扰动：
    足以改变多模态哈希，但不改变时长、声道和波形形状，因此时间戳仍必须落在容差内。
    索引 0 始终保留原始音频，语义结果只从该请求提取。
    """
    if not distinct or index == 0:
        return audio
    perturbed = audio.copy()
    position = index % perturbed.shape[0]
    perturbed[position] = np.float32(perturbed[position] + 1e-6)
    return perturbed


def _fix_timestamp(values: list[float]) -> list[float]:
    """保持官方 Qwen3ForceAlignProcessor.fix_timestamp 的非递减修复语义。"""
    if not values:
        return []
    length = len(values)
    dp = [1] * length
    parent = [-1] * length
    for index in range(1, length):
        for previous in range(index):
            if values[previous] <= values[index] and dp[previous] + 1 > dp[index]:
                dp[index] = dp[previous] + 1
                parent[index] = previous
    cursor = max(range(length), key=dp.__getitem__)
    kept: set[int] = set()
    while cursor != -1:
        kept.add(cursor)
        cursor = parent[cursor]
    result = list(values)
    index = 0
    while index < length:
        if index in kept:
            index += 1
            continue
        end = index
        while end < length and end not in kept:
            end += 1
        left = next((result[pos] for pos in range(index - 1, -1, -1) if pos in kept), None)
        right = next((result[pos] for pos in range(end, length) if pos in kept), None)
        count = end - index
        if count <= 2:
            for pos in range(index, end):
                if left is None:
                    result[pos] = right
                elif right is None:
                    result[pos] = left
                else:
                    result[pos] = left if pos - index + 1 <= end - pos else right
        elif left is not None and right is not None:
            step = (right - left) / (count + 1)
            for pos in range(index, end):
                result[pos] = left + step * (pos - index + 1)
        else:
            fill = left if left is not None else right
            for pos in range(index, end):
                result[pos] = fill
        index = end
    if any(value is None or not math.isfinite(float(value)) for value in result):
        raise RuntimeError("时间桶修复后包含非法值")
    return [float(int(value)) for value in result]


def _units_from_output(
    output: Any,
    words: list[str],
    timestamp_token_id: int,
    timestamp_segment_time: float,
    duration: float,
) -> list[dict[str, Any]]:
    logits = output.outputs.data
    predictions = logits.argmax(dim=-1).tolist()
    bins = [
        float(prediction) * timestamp_segment_time
        for token_id, prediction in zip(output.prompt_token_ids, predictions)
        if int(token_id) == timestamp_token_id
    ]
    expected_count = len(words) * 2
    if len(bins) != expected_count:
        raise RuntimeError(
            f"timestamp 数量异常：期望 {expected_count}，实际 {len(bins)}"
        )
    fixed = _fix_timestamp(bins)
    units: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, word in enumerate(words):
        start = round(fixed[index * 2] / 1000.0, 3)
        end = round(fixed[index * 2 + 1] / 1000.0, 3)
        if start < previous_end or end < start or start < 0 or end > duration + 0.05:
            raise RuntimeError(f"第 {index} 个时间戳不单调或越界：{start}-{end}")
        units.append({"text": word, "start": start, "end": end})
        previous_end = end
    return units


def _validate_expected(
    units: list[dict[str, Any]], path: Path, tolerance_ms: float
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("units") if isinstance(payload, dict) else payload
    if not isinstance(expected, list) or len(expected) != len(units):
        raise RuntimeError("基线对齐单元数量不一致")
    tolerance = tolerance_ms / 1000.0
    for index, (actual, reference) in enumerate(zip(units, expected)):
        if not isinstance(reference, dict) or actual["text"] != reference.get("text"):
            raise RuntimeError(f"第 {index} 个基线对齐单元文本不一致")
        for field in ("start", "end"):
            try:
                delta = abs(float(actual[field]) - float(reference[field]))
            except (TypeError, ValueError, KeyError) as error:
                raise RuntimeError(f"第 {index} 个基线 {field} 非法") from error
            if delta > tolerance:
                raise RuntimeError(
                    f"第 {index} 个 {field} 偏差 {delta * 1000:.3f} ms "
                    f"超过 {tolerance_ms:.3f} ms"
                )


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    actual_vllm_version = importlib.metadata.version("vllm")
    if actual_vllm_version != _EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            "PERF-ARCH-001 必须使用项目统一环境："
            f"期望 vLLM {_EXPECTED_VLLM_VERSION}，实际 {actual_vllm_version}；"
            "不得在同一解释器安装第二个 vLLM 版本"
        )
    from vllm import LLM

    model_dir = Path(args.model).resolve()
    audio_path = Path(args.audio).resolve()
    identity = _model_identity(model_dir)
    audio, sample_rate, duration = _load_audio(audio_path)
    if args.max_num_seqs < max(args.batch_sizes):
        raise ValueError("--max-num-seqs 不能小于最大 batch size")
    llm = LLM(
        model=str(model_dir),
        runner="pooling",
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=False,
        hf_overrides={"architectures": [_ARCHITECTURE]},
    )
    config = llm.llm_engine.vllm_config.model_config.hf_config
    timestamp_token_id = int(config.timestamp_token_id)
    timestamp_segment_time = float(config.timestamp_segment_time)
    prompt = _build_prompt(args.words_json)
    benchmark: list[dict[str, Any]] = []
    semantic_units: list[dict[str, Any]] | None = None
    for batch_size in args.batch_sizes:
        requests = [
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "audio": _batch_audio(audio, index, args.distinct_batch_audio)
                },
            }
            for index in range(batch_size)
        ]
        for _ in range(args.warmup):
            llm.encode(requests, pooling_task="token_classify")
        latencies: list[float] = []
        started = perf_counter()
        last_outputs = None
        for _ in range(args.iterations):
            call_started = perf_counter()
            last_outputs = llm.encode(requests, pooling_task="token_classify")
            latencies.append((perf_counter() - call_started) * 1000)
        elapsed = perf_counter() - started
        if last_outputs is None or len(last_outputs) != batch_size:
            raise RuntimeError("vLLM 返回数量与 batch size 不一致")
        parsed = [
            _units_from_output(
                output, args.words_json, timestamp_token_id,
                timestamp_segment_time, duration,
            )
            for output in last_outputs
        ]
        if args.distinct_batch_audio:
            # 扰动只改变多模态哈希，不改变时长和波形形状，结果必须仍落在时间戳容差内。
            tolerance = args.timestamp_tolerance_ms / 1000.0
            for offset, units in enumerate(parsed[1:], start=1):
                if len(units) != len(parsed[0]):
                    raise RuntimeError(f"第 {offset} 个请求对齐单元数量不一致")
                for position, (actual, reference) in enumerate(zip(units, parsed[0])):
                    if actual["text"] != reference["text"]:
                        raise RuntimeError(
                            f"第 {offset} 个请求第 {position} 个单元文本不一致"
                        )
                    for field in ("start", "end"):
                        if abs(actual[field] - reference[field]) > tolerance:
                            raise RuntimeError(
                                f"第 {offset} 个请求第 {position} 个 {field} 超过容差"
                            )
        elif any(units != parsed[0] for units in parsed[1:]):
            raise RuntimeError("同批重复输入返回了不一致时间戳")
        if semantic_units is None:
            semantic_units = parsed[0]
            if args.expected_json:
                _validate_expected(
                    semantic_units, Path(args.expected_json),
                    args.timestamp_tolerance_ms,
                )
        processed_audio_seconds = batch_size * duration * args.iterations
        item = {
            "batch_size": batch_size,
            "iterations": args.iterations,
            "elapsed_seconds": elapsed,
            "audio_s_per_s": processed_audio_seconds / elapsed,
            "request_batches_per_second": args.iterations / elapsed,
            "call_latency_ms": {
                "mean": statistics.fmean(latencies),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
                "min": min(latencies),
                "max": max(latencies),
            },
        }
        benchmark.append(item)
        print(
            f"batch={batch_size} audio_s/s={item['audio_s_per_s']:.2f} "
            f"mean={item['call_latency_ms']['mean']:.2f}ms "
            f"p95={item['call_latency_ms']['p95']:.2f}ms"
        )
    return {
        "schema_version": 1,
        "experiment_id": "PERF-ARCH-001",
        "runtime": {
            "vllm": actual_vllm_version,
            "numpy": importlib.metadata.version("numpy"),
            "soundfile": importlib.metadata.version("soundfile"),
        },
        "model": {"path": str(model_dir), **identity},
        "input": {
            "filename": audio_path.name,
            "bytes": audio_path.stat().st_size,
            "sha256": _sha256(audio_path),
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "language": args.language,
            "word_count": len(args.words_json),
        },
        "parameters": {
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": args.enable_prefix_caching,
            "distinct_batch_audio": args.distinct_batch_audio,
        },
        "semantic_units": semantic_units,
        "expected_checked": bool(args.expected_json),
        "benchmark": benchmark,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="独立验证 vLLM Qwen3 ForcedAligner token-classify 语义与吞吐"
    )
    parser.add_argument(
        "--model", default="models/qwen3-forced-aligner-0.6b/vllm-spike",
        help="ModelScope 固定 revision 的本地模型目录",
    )
    parser.add_argument("--audio", required=True, help="16 kHz 单声道、最长 32 秒音频")
    parser.add_argument("--language", required=True, help="仅用于报告的官方语言名称")
    parser.add_argument(
        "--words-json", required=True, type=_parse_words,
        help='官方 processor 生成的单元 JSON，例如 ["你","好"]',
    )
    parser.add_argument(
        "--batch-sizes", type=_parse_batch_sizes, default=_parse_batch_sizes("1,8,16,32"),
        help="逗号分隔的 batch size，默认 1,8,16,32",
    )
    parser.add_argument("--warmup", type=int, default=3, help="每个 batch 的预热次数")
    parser.add_argument("--iterations", type=int, default=10, help="每个 batch 的正式执行次数")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True,
        help="首轮默认按官方示例启用；图模式必须另立单变量实验",
    )
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False,
        help="默认关闭：避免同批重复输入命中前缀缓存导致原始吞吐虚高",
    )
    parser.add_argument(
        "--distinct-batch-audio", action=argparse.BooleanOptionalAction, default=True,
        help="默认开启：对批内每个请求施加 1e-6 单点扰动，使多模态缓存不可复用",
    )
    parser.add_argument("--expected-json", help="可选 PyTorch 基线 units JSON")
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=1.0)
    parser.add_argument("--output-json", help="原子写入实验报告")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup 不能小于 0，--iterations 必须大于 0")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization 必须在 0～1 之间")
    if args.timestamp_tolerance_ms < 0:
        parser.error("--timestamp-tolerance-ms 不能小于 0")
    report = _run(args)
    print(json.dumps(report["semantic_units"], ensure_ascii=False, indent=2))
    if args.output_json:
        _write_json(Path(args.output_json), report)
        print(f"报告已写入：{args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
