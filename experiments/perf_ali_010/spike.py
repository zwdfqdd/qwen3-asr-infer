"""PERF-ALI-010：PyTorch ForcedAligner 的 torch.compile 收益与正确性 spike。

只在独立进程中包装 ``aligner.model.thinker``，不修改 src/aligner.py 与生产默认路径。

官方 ``Qwen3ForcedAligner.align()`` 的 GPU 计算集中在一行 ``self.model.thinker(**inputs)``，
其余为 CPU 前后处理（分词、特征提取、时间桶解析）。因此本 spike 分两段测量：

* Phase A：``align()`` 全程，口径与生产日志的 ``aligner_batch_model_call_ms`` 一致；
* Phase B：仅 thinker 前向，排除 CPU 前后处理，反映编译能影响的上限。

生产实测参考值：batch=32 满批时 ``model_call`` 约 780.82 ms，占 Aligner 阶段 83.7%。
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


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


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
    converted: list[list[dict[str, Any]]] = []
    for result in results:
        converted.append([
            {
                "text": item.text,
                "start": round(float(item.start_time), 3),
                "end": round(float(item.end_time), 3),
            }
            for item in result.items
        ])
    return converted


def _assert_same_units(
    baseline: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
    tolerance_ms: float,
) -> None:
    """编译只应改变执行方式，不应改变对齐结果。"""
    if len(baseline) != len(candidate):
        raise RuntimeError("编译前后样本数量不一致")
    tolerance = tolerance_ms / 1000.0
    for sample, (left, right) in enumerate(zip(baseline, candidate)):
        if len(left) != len(right):
            raise RuntimeError(f"第 {sample} 个样本对齐单元数量不一致")
        for index, (a, b) in enumerate(zip(left, right)):
            if a["text"] != b["text"]:
                raise RuntimeError(f"第 {sample} 个样本第 {index} 个单元文本不一致")
            for field in ("start", "end"):
                if abs(a[field] - b[field]) > tolerance:
                    raise RuntimeError(
                        f"第 {sample} 个样本第 {index} 个 {field} 偏差超过 "
                        f"{tolerance_ms} ms：{a[field]} 对 {b[field]}"
                    )


def _dynamo_graph_count() -> int:
    """读取 dynamo 已编译图数量；字段缺失时返回 -1 表示不可用。"""
    try:
        from torch._dynamo.utils import counters
    except Exception:  # noqa: BLE001 - 该计数器属内部接口，缺失不应中断实验。
        return -1
    stats = counters.get("stats", {})
    for key in ("unique_graphs", "calls_captured"):
        if key in stats:
            return int(stats[key])
    return -1


def _build_inputs(aligner: Any, audios: list[Any], texts: list[str], languages: list[str]):
    """复刻官方 align() 的 CPU 前处理，用于单独测量 thinker 前向。"""
    word_lists = []
    aligner_input_texts = []
    for text, language in zip(texts, languages):
        word_list, aligner_input_text = aligner.aligner_processor.encode_timestamp(
            text, language
        )
        word_lists.append(word_list)
        aligner_input_texts.append(aligner_input_text)
    inputs = aligner.processor(
        text=aligner_input_texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(aligner.model.device).to(aligner.model.dtype)
    return inputs, word_lists


def _time_align(
    aligner: Any,
    audios: list[Any],
    texts: list[str],
    languages: list[str],
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], Any]:
    """Phase A：测量 align() 全程，口径与生产 model_call 一致。"""
    import torch

    for _ in range(warmup):
        aligner.align(audio=audios, text=texts, language=languages)
    torch.cuda.synchronize()

    latencies: list[float] = []
    results = None
    for _ in range(iterations):
        started = perf_counter()
        results = aligner.align(audio=audios, text=texts, language=languages)
        torch.cuda.synchronize()
        latencies.append((perf_counter() - started) * 1000)
    return _stats(latencies), results


def _time_thinker(
    aligner: Any, inputs: Any, warmup: int, iterations: int
) -> dict[str, float]:
    """Phase B：只测 thinker 前向与 argmax，排除 CPU 前后处理。"""
    import torch

    thinker = aligner.model.thinker
    with torch.inference_mode():
        for _ in range(warmup):
            thinker(**inputs).logits.argmax(dim=-1)
        torch.cuda.synchronize()

        latencies: list[float] = []
        for _ in range(iterations):
            started = perf_counter()
            thinker(**inputs).logits.argmax(dim=-1)
            torch.cuda.synchronize()
            latencies.append((perf_counter() - started) * 1000)
    return _stats(latencies)


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
    text = Path(args.text).read_text(encoding="utf-8").strip() if args.text_is_file else args.text
    if not text:
        raise ValueError("对齐文本不能为空")

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "PERF-ALI-010",
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
            "compile_mode": args.compile_mode,
            "compile_dynamic": args.compile_dynamic,
            "compile_fullgraph": args.compile_fullgraph,
        },
        "results": [],
    }

    aligner = _load_aligner(args)
    actual_attention = getattr(
        getattr(getattr(aligner, "model", None), "config", None),
        "_attn_implementation",
        None,
    )
    report["runtime"]["attention_actual"] = str(actual_attention)
    print(f"模型已加载：attention={actual_attention}，dtype={args.dtype}")

    for batch_size in args.batch_sizes:
        audios = [audio] * batch_size
        texts = [text] * batch_size
        languages = [args.language] * batch_size

        eager_align, eager_results = _time_align(
            aligner, audios, texts, languages, args.warmup, args.iterations
        )
        inputs, _ = _build_inputs(aligner, audios, texts, languages)
        eager_thinker = _time_thinker(aligner, inputs, args.warmup, args.iterations)
        baseline_units = _units(eager_results)

        # 编译只替换本进程内的 thinker 引用，不触碰磁盘权重与生产代码。
        original_thinker = aligner.model.thinker
        graphs_before = _dynamo_graph_count()
        compile_kwargs: dict[str, Any] = {
            "dynamic": args.compile_dynamic,
            "fullgraph": args.compile_fullgraph,
        }
        if args.compile_mode != "default":
            compile_kwargs["mode"] = args.compile_mode
        compile_started = perf_counter()
        try:
            aligner.model.thinker = torch.compile(original_thinker, **compile_kwargs)
            compiled_align, compiled_results = _time_align(
                aligner, audios, texts, languages, args.warmup, args.iterations
            )
            compiled_thinker = _time_thinker(
                aligner, inputs, args.warmup, args.iterations
            )
            candidate_units = _units(compiled_results)
        finally:
            aligner.model.thinker = original_thinker
        compile_wall_ms = (perf_counter() - compile_started) * 1000
        graphs_after = _dynamo_graph_count()

        _assert_same_units(baseline_units, candidate_units, args.timestamp_tolerance_ms)

        audio_seconds = batch_size * duration
        item = {
            "batch_size": batch_size,
            "align_eager_ms": eager_align,
            "align_compiled_ms": compiled_align,
            "align_gain_percent": round(
                (eager_align["mean"] - compiled_align["mean"])
                / eager_align["mean"] * 100, 2
            ),
            "thinker_eager_ms": eager_thinker,
            "thinker_compiled_ms": compiled_thinker,
            "thinker_gain_percent": round(
                (eager_thinker["mean"] - compiled_thinker["mean"])
                / eager_thinker["mean"] * 100, 2
            ),
            "align_eager_audio_s_per_s": round(
                audio_seconds / (eager_align["mean"] / 1000), 2
            ),
            "align_compiled_audio_s_per_s": round(
                audio_seconds / (compiled_align["mean"] / 1000), 2
            ),
            "dynamo_graphs_before": graphs_before,
            "dynamo_graphs_after": graphs_after,
            "compile_and_measure_wall_ms": round(compile_wall_ms, 2),
            "units_identical": True,
        }
        report["results"].append(item)
        print(
            f"batch={batch_size} "
            f"align {eager_align['mean']:.2f}→{compiled_align['mean']:.2f} ms "
            f"({item['align_gain_percent']:+.2f}%)  "
            f"thinker {eager_thinker['mean']:.2f}→{compiled_thinker['mean']:.2f} ms "
            f"({item['thinker_gain_percent']:+.2f}%)"
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="独立验证 PyTorch ForcedAligner 的 torch.compile 收益与结果一致性"
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
    parser.add_argument("--warmup", type=int, default=5, help="每组预热次数；编译需要足够预热")
    parser.add_argument("--iterations", type=int, default=10, help="每组正式执行次数")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0", help="Aligner 执行设备")
    parser.add_argument(
        "--attention", choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto", help="与生产默认一致；目标机 auto 实际选择 sdpa",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default", help="torch.compile 的 mode；一次只改一个变量",
    )
    parser.add_argument(
        "--compile-dynamic", action=argparse.BooleanOptionalAction, default=None,
        help="torch.compile 的 dynamic；默认交由 PyTorch 自动判定",
    )
    parser.add_argument(
        "--compile-fullgraph", action=argparse.BooleanOptionalAction, default=False,
        help="默认关闭：允许图断裂，先取得可用基线再考虑收紧",
    )
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=1.0)
    parser.add_argument("--output-json", help="原子写入实验报告")
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup 不能小于 0，--iterations 必须大于 0")
    if args.timestamp_tolerance_ms < 0:
        parser.error("--timestamp-tolerance-ms 不能小于 0")
    if args.output_json and not args.output_json.endswith(".json"):
        parser.error("--output-json 路径必须以 .json 结尾")

    report = _run(args)
    if args.output_json:
        _write_json(Path(args.output_json), report)
        print(f"报告已写入：{args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
