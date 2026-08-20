#!/usr/bin/env python3
"""PERF-ARCH-001 Phase A：11 语种 PyTorch oracle 与 vLLM token-classify 语义对照。

必须分两个独立进程执行：

1. ``oracle`` 使用 qwen-asr 0.0.6 PyTorch ForcedAligner 生成逐单元基线；
2. ``compare`` 只加载 vLLM 0.19.1 token-classify，逐语言直接对照同一基线。

这样避免同进程同时加载两个 0.6B 模型占用显存，也避免把性能模式中的缓存扰动、批内间接
对照和容差传递带入正确性门禁。报告包含哈希和派生单元，不保存原始转写文本。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import soundfile as sf

_ARCHITECTURE = "Qwen3ASRForcedAlignerForTokenClassification"
_EXPECTED_VLLM_VERSION = "0.19.1"
_EXPECTED_QWEN_ASR_VERSION = "0.0.6"
_EXPECTED_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
_EXPECTED_REVISION = "cf1c50164ea3ac48240d12bef5ead74aee0720cc"
_SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
)
_LANGUAGE_MAP = {language.lower(): language for language in _SUPPORTED_LANGUAGES}


def _backend_name(value: Any) -> str:
    """把 vLLM attention 后端枚举规范化为稳定名称。"""
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(value).rsplit(".", 1)[-1].upper()


def _validate_mm_encoder_backend_request(
    requested: str,
    supported_names: tuple[str, ...],
    engine_arg_names: set[str],
) -> str | None:
    """校验显式 MM encoder 后端；auto 不覆盖 vLLM 自动选择。"""
    canonical = requested.strip().replace("-", "_").upper()
    if canonical in ("", "AUTO"):
        return None
    if canonical not in supported_names:
        raise ValueError(
            f"MM encoder attention 后端 {canonical} 不受当前平台支持；"
            f"可用值：{', '.join(supported_names)}"
        )
    if "mm_encoder_attn_backend" not in engine_arg_names:
        raise RuntimeError(
            "当前 vLLM EngineArgs 未暴露 mm_encoder_attn_backend；"
            "拒绝盲传或改用普通 attention_config.backend"
        )
    return canonical


def _resolve_mm_encoder_backend(requested: str) -> dict[str, Any]:
    """从当前 vLLM/平台解析受支持枚举，并确认正确的 EngineArgs 入口。"""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    supported_values = tuple(current_platform.get_supported_vit_attn_backends())
    supported_by_name = {_backend_name(value): value for value in supported_values}
    supported_names = tuple(supported_by_name)
    if not supported_names:
        raise RuntimeError("当前平台未报告任何可用的 ViT/MM encoder attention 后端")
    engine_arg_names = set(inspect.signature(EngineArgs).parameters)
    canonical = _validate_mm_encoder_backend_request(
        requested, supported_names, engine_arg_names
    )
    enum_value = None if canonical is None else supported_by_name[canonical]
    if enum_value is not None and not isinstance(enum_value, AttentionBackendEnum):
        raise RuntimeError(f"vLLM 返回了未知的 attention 后端类型：{type(enum_value).__name__}")
    return {
        "requested": canonical or "AUTO",
        "supported": list(supported_names),
        "engine_argument": "mm_encoder_attn_backend" if canonical else None,
        "enum_value": enum_value,
    }


def _enable_vllm_audio_cross_window_diagnostic() -> dict[str, Any]:
    """仅诊断：让 vLLM Torch SDPA 忽略 cu_seqlens，复刻生产 oracle 的跨窗语义。

    vLLM 默认在独立 EngineCore 进程执行模型，普通 monkeypatch 无法传入子进程。调用方必须先把
    ``VLLM_ENABLE_V1_MULTIPROCESSING`` 设为 0；每次推理还会检查调用计数，未真正命中补丁时
    直接失败，禁止把未生效结果写入报告。
    """
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    method_globals = MMEncoderAttention._forward_sdpa.__globals__
    original_wrapper = method_globals.get("vit_torch_sdpa_wrapper")
    if not callable(original_wrapper):
        raise RuntimeError(
            "MMEncoderAttention._forward_sdpa 未引用 vit_torch_sdpa_wrapper，"
            "当前 vLLM 源码与诊断假设不一致"
        )
    if getattr(original_wrapper, "_phase_a_cross_window_diagnostic", False):
        raise RuntimeError("vLLM audio cross-window 诊断补丁已存在，拒绝重复包装")

    state: dict[str, Any] = {
        "calls": 0,
        "discarded_cu_seqlens": 0,
        "target_class": (
            f"{MMEncoderAttention.__module__}.{MMEncoderAttention.__name__}"
        ),
        "wrapper_module": getattr(original_wrapper, "__module__", None),
        "wrapper_name": getattr(original_wrapper, "__name__", None),
    }

    def cross_window_wrapper(
        q: Any,
        k: Any,
        v: Any,
        scale: float | None = None,
        cu_seqlens: Any = None,
        enable_gqa: bool = False,
    ) -> Any:
        state["calls"] += 1
        if cu_seqlens is None:
            raise RuntimeError(
                "vLLM audio cross-window 诊断未收到 cu_seqlens，无法证明唯一变量是分窗语义"
            )
        state["discarded_cu_seqlens"] += 1
        return original_wrapper(
            q=q,
            k=k,
            v=v,
            scale=scale,
            cu_seqlens=None,
            enable_gqa=enable_gqa,
        )

    cross_window_wrapper._phase_a_cross_window_diagnostic = True  # type: ignore[attr-defined]
    method_globals["vit_torch_sdpa_wrapper"] = cross_window_wrapper
    return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _model_identity(model_dir: Path) -> dict[str, str]:
    manifest_path = model_dir / ".modelscope-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 ModelScope manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {
        "repo_id": str(manifest.get("repo_id", "")),
        "revision": str(manifest.get("revision", "")),
    }
    expected = {"repo_id": _EXPECTED_REPO, "revision": _EXPECTED_REVISION}
    if identity != expected:
        raise RuntimeError(f"Aligner 模型身份不符合固定值：{identity}")
    return identity


def _load_audio(path: Path) -> tuple[np.ndarray, float]:
    with sf.SoundFile(path) as source:
        if source.samplerate != 16000:
            raise ValueError(f"{path} 不是 16 kHz：{source.samplerate} Hz")
        if source.channels != 1 or source.frames <= 0:
            raise ValueError(f"{path} 必须是非空单声道音频")
        audio = source.read(dtype="float32", always_2d=False)
        duration = source.frames / source.samplerate
    if duration > 32.0:
        raise ValueError(f"{path} 超过 Phase A 单片 32 秒上限：{duration:.3f}s")
    return np.ascontiguousarray(audio), duration


def _canonical_language(value: Any) -> str:
    language = _LANGUAGE_MAP.get(str(value).strip().lower())
    if language is None:
        raise ValueError(f"不支持的 ForcedAligner 语言：{value!r}")
    return language


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Phase A manifest 的 schema_version 必须为 1")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != len(_SUPPORTED_LANGUAGES):
        raise ValueError("Phase A manifest 必须恰好包含 11 个语种样本")

    samples: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    languages: set[str] = set()
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest 第 {index} 项必须是对象")
        identifier = str(raw.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError(f"manifest 第 {index} 项 id 为空或重复：{identifier!r}")
        language = _canonical_language(raw.get("language"))
        if language in languages:
            raise ValueError(f"manifest 语种重复：{language}")
        audio_path = Path(str(raw.get("audio", ""))).resolve()
        text_path = Path(str(raw.get("text", ""))).resolve()
        if not audio_path.is_file() or not text_path.is_file():
            raise FileNotFoundError(
                f"{identifier} 缺少音频或文本：audio={audio_path}, text={text_path}"
            )
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{identifier} 的参考文本为空")
        identifiers.add(identifier)
        languages.add(language)
        samples.append({
            "id": identifier,
            "language": language,
            "audio_path": audio_path,
            "text_path": text_path,
            "text": text,
        })

    missing = set(_SUPPORTED_LANGUAGES) - languages
    if missing:
        raise ValueError(f"manifest 缺少语种：{sorted(missing)}")
    return samples


def _build_prompt(words: list[str]) -> str:
    body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


def _fix_timestamp(values: list[float]) -> list[float]:
    """逐行保持 qwen-asr 0.0.6 Qwen3ForceAlignProcessor.fix_timestamp 语义。"""
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


def _validate_units(units: list[dict[str, Any]], duration: float, sample_id: str) -> None:
    previous_end = 0.0
    if not units:
        raise RuntimeError(f"{sample_id} 未生成任何对齐单元")
    for index, unit in enumerate(units):
        try:
            text = unit["text"]
            start = float(unit["start"])
            end = float(unit["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{sample_id} 第 {index} 个单元格式非法") from error
        if not isinstance(text, str) or not text:
            raise RuntimeError(f"{sample_id} 第 {index} 个单元文本为空")
        # 官方 fix_timestamp 保证的是非降，允许相邻边界相等和零宽单元；不伪称严格递增。
        if start < previous_end or end < start or start < 0 or end > duration + 0.05:
            raise RuntimeError(
                f"{sample_id} 第 {index} 个时间戳非降/边界校验失败：{start}-{end}"
            )
        previous_end = end


def _units_from_vllm(
    output: Any,
    words: list[str],
    timestamp_token_id: int,
    timestamp_segment_time: float,
    duration: float,
    sample_id: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    """返回单元、原始桶下标和每个 timestamp 位置的 top-k 候选，后两项用于定因不参与门禁。"""
    logits = output.outputs.data
    predictions = logits.argmax(dim=-1).tolist()
    keep = [
        index for index, token_id in enumerate(output.prompt_token_ids)
        if int(token_id) == timestamp_token_id
    ]
    buckets = [int(predictions[index]) for index in keep]
    top_candidates: list[dict[str, Any]] = []
    if keep:
        width = int(logits.shape[-1])
        k = max(1, min(top_k, width))
        values, indices = logits[keep].float().topk(k, dim=-1)
        top_candidates = [
            {
                "buckets": [int(bucket) for bucket in index_row],
                "logits": [round(float(value), 5) for value in value_row],
            }
            for index_row, value_row in zip(indices.tolist(), values.tolist())
        ]

    expected_count = len(words) * 2
    if len(buckets) != expected_count:
        raise RuntimeError(
            f"{sample_id} timestamp 数量异常：期望 {expected_count}，实际 {len(buckets)}"
        )
    fixed = _fix_timestamp([float(bucket) * timestamp_segment_time for bucket in buckets])
    units = [
        {
            "text": word,
            "start": round(fixed[index * 2] / 1000.0, 3),
            "end": round(fixed[index * 2 + 1] / 1000.0, 3),
        }
        for index, word in enumerate(words)
    ]
    _validate_units(units, duration, sample_id)
    return units, buckets, top_candidates


def _margin(candidate: dict[str, Any]) -> float:
    """top1 与 top2 的 logit 间距；只有一个候选时视为无限大。"""
    values = candidate.get("logits") or []
    return float(values[0] - values[1]) if len(values) >= 2 else float("inf")


def _logit_at(candidate: dict[str, Any], bucket: int) -> float | None:
    """在已记录的 top-k 里查某个桶的 logit；不在 top-k 内返回 None。"""
    buckets = candidate.get("buckets") or []
    logits = candidate.get("logits") or []
    for index, value in enumerate(buckets):
        if int(value) == bucket and index < len(logits):
            return float(logits[index])
    return None


def _classify_mismatch(
    position: int,
    delta: int,
    actual_bucket: int,
    expected_bucket: int,
    actual_candidate: dict[str, Any] | None,
    expected_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    """按两侧候选集判定单个分歧位置的成因，判据显式写出，不做隐式归类。"""
    record: dict[str, Any] = {
        "position": position,
        "bucket_delta": delta,
        "vllm_bucket": actual_bucket,
        "pytorch_bucket": expected_bucket,
    }
    if not actual_candidate or not expected_candidate:
        record["cause"] = "候选缺失，无法判定"
        return record

    vllm_margin = _margin(actual_candidate)
    pytorch_margin = _margin(expected_candidate)
    record["vllm_top_candidates"] = actual_candidate
    record["pytorch_top_candidates"] = expected_candidate
    record["vllm_margin"] = round(vllm_margin, 5) if vllm_margin != float("inf") else None
    record["pytorch_margin"] = (
        round(pytorch_margin, 5) if pytorch_margin != float("inf") else None
    )
    # 交叉查询：本侧选中的桶在对侧候选里的 logit，差多少才会翻转。
    record["vllm_logit_at_pytorch_bucket"] = _logit_at(actual_candidate, expected_bucket)
    record["pytorch_logit_at_vllm_bucket"] = _logit_at(expected_candidate, actual_bucket)
    mutual = (
        record["vllm_logit_at_pytorch_bucket"] is not None
        and record["pytorch_logit_at_vllm_bucket"] is not None
    )
    record["mutually_in_top_k"] = mutual

    if mutual and max(vllm_margin, pytorch_margin) < 0.05:
        # 两侧都把对方的答案列为近邻候选且各自几乎平局：模型对该位置本身不确定。
        record["cause"] = "两侧近似平局，模型对该位置本身不确定"
    elif abs(delta) <= 1 and min(vllm_margin, pytorch_margin) < 0.05:
        record["cause"] = "相邻桶边界近似平局，处于 80 ms 分辨率下限"
    elif mutual:
        record["cause"] = "互为候选但间距不小，两侧对同一组假设排序不同"
    else:
        record["cause"] = "对侧答案未进入本侧 top-k，无法归因于已记录候选近似平局"
    return record


def _bucket_diagnosis(
    actual_buckets: list[int],
    expected_buckets: Any,
    actual_candidates: list[dict[str, Any]],
    expected_candidates: Any,
) -> dict[str, Any]:
    """比较两侧原始桶下标与候选集，区分复刻偏差、模糊性归属与真实实现差异。"""
    if not isinstance(expected_buckets, list) or len(expected_buckets) != len(actual_buckets):
        return {"available": False, "reason": "oracle 未记录可比较的原始桶下标"}
    deltas = [
        int(actual) - int(expected)
        for actual, expected in zip(actual_buckets, expected_buckets)
    ]
    positions = [index for index, delta in enumerate(deltas) if delta != 0]
    candidates_usable = (
        isinstance(expected_candidates, list)
        and len(expected_candidates) == len(actual_buckets)
        and len(actual_candidates) == len(actual_buckets)
    )
    mismatches = [
        _classify_mismatch(
            position,
            deltas[position],
            int(actual_buckets[position]),
            int(expected_buckets[position]),
            actual_candidates[position] if candidates_usable else None,
            expected_candidates[position] if candidates_usable else None,
        )
        for position in positions
    ]
    causes: dict[str, int] = {}
    for record in mismatches:
        causes[record["cause"]] = causes.get(record["cause"], 0) + 1
    return {
        "available": True,
        "bucket_count": len(deltas),
        "mismatch_count": len(positions),
        "max_abs_bucket_delta": max((abs(delta) for delta in deltas), default=0),
        "identical_buckets": not positions,
        "candidates_compared": candidates_usable,
        "cause_counts": causes,
        "mismatches": mismatches,
    }


def _compare_units(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    tolerance_ms: float,
    sample_id: str,
) -> dict[str, Any]:
    """结构不一致直接抛错；时间戳偏差全量收集后由调用方判定，便于一次看清全部分歧位置。"""
    if len(actual) != len(expected):
        raise RuntimeError(
            f"{sample_id} 单元数量不一致：vLLM={len(actual)}, PyTorch={len(expected)}"
        )
    exceeded: list[dict[str, Any]] = []
    max_delta_ms = 0.0
    for index, (candidate, oracle) in enumerate(zip(actual, expected)):
        if candidate["text"] != oracle.get("text"):
            raise RuntimeError(f"{sample_id} 第 {index} 个单元文本不一致")
        for field in ("start", "end"):
            try:
                delta_ms = abs(float(candidate[field]) - float(oracle[field])) * 1000.0
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"{sample_id} 第 {index} 个 oracle {field} 非法") from error
            max_delta_ms = max(max_delta_ms, delta_ms)
            if delta_ms > tolerance_ms:
                exceeded.append({
                    "unit_index": index,
                    "field": field,
                    "delta_ms": round(delta_ms, 3),
                    "vllm": float(candidate[field]),
                    "pytorch": float(oracle[field]),
                })
    return {
        "unit_count": len(actual),
        "max_timestamp_delta_ms": round(max_delta_ms, 3),
        "exceeded_count": len(exceeded),
        "exceeded": exceeded,
        "within_tolerance": not exceeded,
    }


def _print_diagnosis(
    comparison: dict[str, Any],
    diagnosis: dict[str, Any],
    timestamp_segment_time: float,
) -> None:
    """打印定因所需的最小信息：偏差是否为桶宽整数倍、桶下标差异、以及是否近似平局。"""
    for item in comparison["exceeded"][:8]:
        multiple = item["delta_ms"] / timestamp_segment_time if timestamp_segment_time else 0
        print(
            f"    单元 {item['unit_index']} {item['field']}："
            f"vLLM {item['vllm']:.3f}s 对 PyTorch {item['pytorch']:.3f}s，"
            f"差 {item['delta_ms']:.3f} ms = {multiple:.2f} 个桶宽"
        )
    if comparison["exceeded_count"] > 8:
        print(f"    ……其余 {comparison['exceeded_count'] - 8} 处见报告 exceeded 字段")
    if not diagnosis.get("available"):
        print(f"    桶级诊断不可用：{diagnosis.get('reason')}；请用新版 oracle 重新生成")
        return
    if diagnosis["identical_buckets"]:
        print(
            "    两侧原始桶下标完全一致，差异只能来自 fix_timestamp 复刻："
            "这是本工具的实现问题，不是后端分歧"
        )
        return
    print(
        f"    原始桶 {diagnosis['mismatch_count']}/{diagnosis['bucket_count']} 处不同，"
        f"最大差 {diagnosis['max_abs_bucket_delta']} 个桶"
    )
    if not diagnosis["candidates_compared"]:
        print("    两侧候选集不可比，无法判定成因；请用当前版本重新生成 oracle")
        return
    for cause, count in sorted(diagnosis["cause_counts"].items(), key=lambda item: -item[1]):
        print(f"    成因 {count} 处：{cause}")
    for record in diagnosis["mismatches"][:6]:
        print(
            f"    位置 {record['position']}：vLLM 桶 {record['vllm_bucket']}"
            f"（间距 {record.get('vllm_margin')}）对 PyTorch 桶 {record['pytorch_bucket']}"
            f"（间距 {record.get('pytorch_margin')}），"
            f"互为 top-k={record.get('mutually_in_top_k')}"
        )
    if diagnosis["mismatch_count"] > 6:
        print(f"    ……其余 {diagnosis['mismatch_count'] - 6} 处见报告 mismatches 字段")


def _sample_provenance(sample: dict[str, Any], duration: float) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "language": sample["language"],
        "audio_filename": sample["audio_path"].name,
        "audio_sha256": _sha256(sample["audio_path"]),
        "audio_bytes": sample["audio_path"].stat().st_size,
        "duration_seconds": duration,
        "text_filename": sample["text_path"].name,
        "text_sha256": _sha256(sample["text_path"]),
        "text_chars": len(sample["text"]),
    }


def _validate_oracle(oracle: Any, samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(oracle, dict) or oracle.get("schema_version") != 1:
        raise ValueError("oracle schema_version 必须为 1")
    experiment_id = oracle.get("experiment_id")
    valid_experiment_ids = {
        "PERF-ARCH-001-PHASE-A-ORACLE",
        "PERF-ARCH-001-PHASE-A-ORACLE-AUDIO-BLOCK-MASK-DIAGNOSTIC",
    }
    if experiment_id not in valid_experiment_ids:
        raise ValueError("expected JSON 不是 PERF-ARCH-001 Phase A oracle")
    if oracle.get("model", {}).get("repo_id") != _EXPECTED_REPO or oracle.get("model", {}).get(
        "revision"
    ) != _EXPECTED_REVISION:
        raise ValueError("oracle 模型身份不符合固定值")
    if oracle.get("runtime", {}).get("qwen_asr") != _EXPECTED_QWEN_ASR_VERSION:
        raise ValueError("oracle 必须由 qwen-asr 0.0.6 生成")
    rows = oracle.get("samples")
    if not isinstance(rows, list) or len(rows) != len(_SUPPORTED_LANGUAGES):
        raise ValueError("oracle 必须包含 11 个语种结果")
    by_id = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
    if len(by_id) != len(rows):
        raise ValueError("oracle 样本 id 重复或非法")
    for sample in samples:
        row = by_id.get(sample["id"])
        if row is None or row.get("language") != sample["language"]:
            raise ValueError(f"oracle 缺少样本或语种不一致：{sample['id']}")
        if row.get("audio_sha256") != _sha256(sample["audio_path"]):
            raise ValueError(f"oracle 音频哈希不一致：{sample['id']}")
        if row.get("text_sha256") != _sha256(sample["text_path"]):
            raise ValueError(f"oracle 文本哈希不一致：{sample['id']}")
    return by_id


def _topk_at_positions(logits: Any, mask: Any, top_k: int) -> list[dict[str, Any]]:
    """取每个 timestamp 位置的 top-k 桶下标与 logit，用于判断分歧是模糊性还是实现差异。"""
    selected = logits[mask].float()
    width = int(selected.shape[-1])
    k = max(1, min(top_k, width))
    values, indices = selected.topk(k, dim=-1)
    return [
        {
            "buckets": [int(bucket) for bucket in index_row],
            "logits": [round(float(value), 5) for value in value_row],
        }
        for index_row, value_row in zip(indices.tolist(), values.tolist())
    ]


def _oracle_buckets(
    aligner: Any,
    audio: np.ndarray,
    sample: dict[str, Any],
    words: list[str],
    units: list[dict[str, Any]],
    torch: Any,
    top_k: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """用官方组件复刻一次前向，取 argmax 原始桶下标；仅作定因用途，不替代 align()。

    复刻路径与 PERF-ALI-011 一致，并要求复刻出的单元与官方 ``align()`` 零容差一致，
    否则说明复刻已偏离官方语义，此时直接失败而不是产出可疑的诊断数据。
    """
    from qwen_asr.inference.utils import normalize_audios

    with torch.inference_mode():
        audios = normalize_audios([(audio, 16000)])
        word_list, aligner_input_text = aligner.aligner_processor.encode_timestamp(
            sample["text"], sample["language"]
        )
        inputs = aligner.processor(
            text=[aligner_input_text], audio=audios, return_tensors="pt", padding=True
        )
        inputs = inputs.to(aligner.model.device).to(aligner.model.dtype)
        logits = aligner.model.thinker(**inputs).logits
        output_ids = logits.argmax(dim=-1)
        input_id = inputs["input_ids"][0]
        output_id = output_ids[0]
        mask = input_id == aligner.timestamp_token_id
        masked = output_id[mask]
        buckets = [int(value) for value in masked.to("cpu").tolist()]
        top_candidates = _topk_at_positions(logits[0], mask, top_k)
        timestamp_ms = (
            masked * aligner.timestamp_segment_time
        ).to("cpu").numpy()
        replicated = aligner.aligner_processor.parse_timestamp(word_list, timestamp_ms)

    if len(buckets) != len(words) * 2:
        raise RuntimeError(
            f"{sample['id']} 复刻桶数量异常：期望 {len(words) * 2}，实际 {len(buckets)}"
        )
    for index, (item, unit) in enumerate(zip(replicated, units)):
        start = round(float(item["start_time"]) / 1000.0, 3)
        end = round(float(item["end_time"]) / 1000.0, 3)
        if item["text"] != unit["text"] or start != unit["start"] or end != unit["end"]:
            raise RuntimeError(
                f"{sample['id']} 第 {index} 个单元的复刻结果与官方 align() 不一致："
                f"复刻 {start}-{end}，官方 {unit['start']}-{unit['end']}；"
                "诊断路径已偏离官方语义"
            )
    return buckets, top_candidates


def _enable_oracle_audio_block_mask_diagnostic(aligner: Any) -> int:
    """仅诊断：让 PyTorch SDPA 使用源码中已有但生产路径未调用的块对角 mask。"""
    audio_tower = aligner.model.thinker.audio_tower
    implementation = getattr(audio_tower.config, "_attn_implementation", None)
    if implementation != "sdpa":
        raise RuntimeError(
            "audio block-mask 诊断要求 PyTorch 音频塔实际使用 sdpa，"
            f"当前为 {implementation!r}"
        )
    prepare_mask = getattr(audio_tower, "_prepare_attention_mask", None)
    if not callable(prepare_mask):
        raise RuntimeError("PyTorch 音频塔缺少 _prepare_attention_mask，无法执行诊断")

    patched = 0
    for layer in audio_tower.layers:
        attention = layer.self_attn
        original_forward = attention.forward

        def forward_with_block_mask(
            hidden_states: Any,
            cu_seqlens: Any = None,
            attention_mask: Any = None,
            _original_forward: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            if attention_mask is not None:
                raise RuntimeError("audio block-mask 诊断收到意外的既有 attention_mask")
            if cu_seqlens is None:
                raise RuntimeError("audio block-mask 诊断缺少 cu_seqlens")
            block_mask = prepare_mask(hidden_states, cu_seqlens)
            if block_mask is None:
                raise RuntimeError("audio block-mask 诊断未生成块对角 mask")
            return _original_forward(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=block_mask,
                **kwargs,
            )

        attention.forward = forward_with_block_mask
        patched += 1

    if patched == 0:
        raise RuntimeError("PyTorch 音频塔没有可包装的 attention 层")
    return patched


def _run_oracle(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from qwen_asr import Qwen3ForcedAligner

    actual_qwen_asr = importlib.metadata.version("qwen-asr")
    if actual_qwen_asr != _EXPECTED_QWEN_ASR_VERSION:
        raise RuntimeError(
            f"Phase A oracle 需要 qwen-asr {_EXPECTED_QWEN_ASR_VERSION}，实际 {actual_qwen_asr}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Phase A oracle 需要可用 CUDA")
    cuda_version = torch.version.cuda or ""
    if not cuda_version.startswith("12."):
        raise RuntimeError(f"需要 CUDA 12.x PyTorch，实际为 {cuda_version}")

    manifest_path = Path(args.manifest).resolve()
    model_dir = Path(args.model).resolve()
    samples = _load_manifest(manifest_path)
    identity = _model_identity(model_dir)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    aligner = Qwen3ForcedAligner.from_pretrained(
        str(model_dir), device_map=args.device, dtype=dtype, local_files_only=True
    )
    diagnostic_layers = 0
    if args.diagnostic_oracle_audio_block_mask:
        diagnostic_layers = _enable_oracle_audio_block_mask_diagnostic(aligner)
        print(
            "注意：已为 PyTorch oracle 的 "
            f"{diagnostic_layers} 层音频 attention 启用块对角 mask；"
            "该报告仅用于定因，不是生产 Phase A oracle"
        )
    supported = aligner.get_supported_languages()
    if supported is not None and set(supported) != set(_LANGUAGE_MAP):
        raise RuntimeError(f"模型支持语言与固定 11 语种不一致：{supported}")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        audio, duration = _load_audio(sample["audio_path"])
        words, prompt = aligner.aligner_processor.encode_timestamp(
            sample["text"], sample["language"]
        )
        if not words or prompt != _build_prompt(words):
            raise RuntimeError(f"{sample['id']} 官方 processor 未生成有效 prompt")
        aligned = aligner.align(
            audio=[(audio, 16000)],
            text=[sample["text"]],
            language=[sample["language"]],
        )
        if len(aligned) != 1:
            raise RuntimeError(f"{sample['id']} PyTorch 返回数量异常：{len(aligned)}")
        units = [
            {
                "text": item.text,
                "start": round(float(item.start_time), 3),
                "end": round(float(item.end_time), 3),
            }
            for item in aligned[0].items
        ]
        if [unit["text"] for unit in units] != words:
            raise RuntimeError(f"{sample['id']} align() 单元与 encode_timestamp() 不一致")
        _validate_units(units, duration, sample["id"])
        buckets, top_candidates = _oracle_buckets(
            aligner, audio, sample, words, units, torch, args.diagnostic_top_k
        )
        row = {
            **_sample_provenance(sample, duration),
            "words": words,
            "units": units,
            "buckets": buckets,
            "top_candidates": top_candidates,
        }
        rows.append(row)
        print(f"oracle {sample['language']}: {len(units)} 单元，通过")

    diagnostic_only = bool(args.diagnostic_oracle_audio_block_mask)
    return {
        "schema_version": 1,
        "experiment_id": (
            "PERF-ARCH-001-PHASE-A-ORACLE-AUDIO-BLOCK-MASK-DIAGNOSTIC"
            if diagnostic_only
            else "PERF-ARCH-001-PHASE-A-ORACLE"
        ),
        "diagnostic_only": diagnostic_only,
        "runtime": {
            "qwen_asr": actual_qwen_asr,
            "torch": torch.__version__,
            "torch_cuda": cuda_version,
            "transformers": importlib.metadata.version("transformers"),
        },
        "model": {"path": str(model_dir), **identity},
        "parameters": {
            "dtype": args.dtype,
            "device": args.device,
            "diagnostic_top_k": args.diagnostic_top_k,
            "audio_block_mask_diagnostic": diagnostic_only,
            "audio_block_mask_patched_layers": diagnostic_layers,
            "oracle_semantics": (
                "PyTorch SDPA 使用 cu_seqlens 块对角 mask，仅用于与 vLLM 分窗语义定因"
                if diagnostic_only
                else "qwen-asr 0.0.6 未修改生产语义"
            ),
        },
        "languages": list(_SUPPORTED_LANGUAGES),
        "samples": rows,
        "summary": {"passed": len(rows), "total": len(_SUPPORTED_LANGUAGES)},
    }


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    cross_window_diagnostic = bool(args.diagnostic_vllm_audio_cross_window)
    if cross_window_diagnostic:
        # 必须在首次导入 vLLM 前关闭独立 EngineCore，确保下面的 monkeypatch 与模型同进程。
        configured = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        if configured not in (None, "0"):
            raise RuntimeError(
                "--diagnostic-vllm-audio-cross-window 要求 "
                "VLLM_ENABLE_V1_MULTIPROCESSING=0；当前显式值为 "
                f"{configured!r}"
            )
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    actual_vllm = importlib.metadata.version("vllm")
    if actual_vllm != _EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"Phase A compare 需要 vLLM {_EXPECTED_VLLM_VERSION}，实际 {actual_vllm}；"
            "不得在同一解释器安装第二个 vLLM 版本"
        )
    from vllm import LLM

    backend = _resolve_mm_encoder_backend(args.mm_encoder_attn_backend)
    if cross_window_diagnostic and backend["requested"] != "TORCH_SDPA":
        raise ValueError(
            "--diagnostic-vllm-audio-cross-window 只允许配合 "
            "--mm-encoder-attn-backend TORCH_SDPA"
        )
    print(
        "MM encoder attention："
        f"请求={backend['requested']}，平台支持={','.join(backend['supported'])}"
    )

    manifest_path = Path(args.manifest).resolve()
    expected_path = Path(args.expected_json).resolve()
    model_dir = Path(args.model).resolve()
    samples = _load_manifest(manifest_path)
    oracle = json.loads(expected_path.read_text(encoding="utf-8"))
    oracle_by_id = _validate_oracle(oracle, samples)
    oracle_parameters = oracle.get("parameters", {})
    diagnostic_oracle = bool(oracle_parameters.get("audio_block_mask_diagnostic"))
    diagnostic_id = (
        oracle.get("experiment_id")
        == "PERF-ARCH-001-PHASE-A-ORACLE-AUDIO-BLOCK-MASK-DIAGNOSTIC"
    )
    if diagnostic_oracle != diagnostic_id:
        raise ValueError("诊断 oracle 的 experiment_id 与参数标记不一致")
    if diagnostic_oracle and not args.allow_audio_block_mask_diagnostic_oracle:
        raise ValueError(
            "该 oracle 修改了 PyTorch 音频窗口 mask，仅可用于定因；"
            "必须显式传 --allow-audio-block-mask-diagnostic-oracle"
        )
    if args.allow_audio_block_mask_diagnostic_oracle and not diagnostic_oracle:
        raise ValueError(
            "--allow-audio-block-mask-diagnostic-oracle 只能配合对应诊断 oracle 使用"
        )
    if diagnostic_oracle:
        print(
            "注意：本轮 compare 使用 audio block-mask 诊断 oracle；"
            "结果不得作为 Phase A 11/11 晋级证据"
        )
    if cross_window_diagnostic and diagnostic_oracle:
        raise ValueError(
            "vLLM 跨窗口诊断必须对照未修改的生产 PyTorch oracle，"
            "不得与 audio block-mask 诊断 oracle 叠加"
        )
    cross_window_state: dict[str, Any] | None = None
    if cross_window_diagnostic:
        cross_window_state = _enable_vllm_audio_cross_window_diagnostic()
        print(
            "注意：已启用 vLLM audio cross-window 诊断，Torch SDPA 将忽略 "
            "cu_seqlens；该报告只用于验证生产 oracle 兼容性，不是生产实现"
        )
    identity = _model_identity(model_dir)
    without_buckets = [
        identifier for identifier, row in oracle_by_id.items()
        if not isinstance(row.get("buckets"), list)
    ]
    if without_buckets:
        print(
            f"警告：oracle 有 {len(without_buckets)} 个样本没有原始桶下标，"
            "分歧将无法定因；建议用当前版本重新生成 oracle"
        )

    oracle_dtype = oracle.get("parameters", {}).get("dtype")
    if oracle_dtype != args.dtype:
        if not args.allow_oracle_dtype_mismatch:
            raise ValueError(
                f"oracle dtype={oracle_dtype}，compare dtype={args.dtype}；"
                "如确认要固定生产 oracle、只改变候选 dtype，必须显式传 "
                "--allow-oracle-dtype-mismatch"
            )
        print(
            f"注意：固定 oracle dtype={oracle_dtype}，仅测试 vLLM candidate dtype={args.dtype}；"
            "这是显式单变量实验"
        )
    without_candidates = [
        identifier for identifier, row in oracle_by_id.items()
        if not isinstance(row.get("top_candidates"), list)
    ]
    if without_candidates:
        print(
            f"警告：oracle 有 {len(without_candidates)} 个样本没有 PyTorch top-k 候选，"
            "只能比较原始桶，无法区分模糊性归属与实现差异；请用当前版本重新生成 oracle"
        )
    else:
        oracle_top_k = oracle.get("parameters", {}).get("diagnostic_top_k")
        if oracle_top_k != args.diagnostic_top_k:
            raise ValueError(
                f"oracle diagnostic_top_k={oracle_top_k}，compare={args.diagnostic_top_k}；"
                "双侧候选必须使用相同 top-k，请重新生成 oracle 或统一参数"
            )

    llm_arguments: dict[str, Any] = {
        "model": str(model_dir),
        "runner": "pooling",
        "enforce_eager": args.enforce_eager,
        "dtype": args.dtype,
        "max_num_seqs": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
        "trust_remote_code": False,
        "hf_overrides": {"architectures": [_ARCHITECTURE]},
    }
    if backend["enum_value"] is not None:
        llm_arguments["mm_encoder_attn_backend"] = backend["enum_value"]
    llm = LLM(**llm_arguments)

    model_config = llm.llm_engine.vllm_config.model_config
    multimodal_config = model_config.multimodal_config
    configured_value = (
        multimodal_config.mm_encoder_attn_backend
        if multimodal_config is not None
        else None
    )
    configured_backend = (
        "AUTO" if configured_value is None else _backend_name(configured_value)
    )
    explicit_backend = backend["enum_value"] is not None
    if explicit_backend and configured_backend != backend["requested"]:
        raise RuntimeError(
            "MM encoder attention 后端未生效："
            f"请求={backend['requested']}，配置回读={configured_backend}；拒绝继续比较"
        )
    if explicit_backend:
        print(
            f"MM encoder attention 已严格回读为 {configured_backend}；"
            "启动日志还必须出现相同后端，否则本轮结果作废"
        )
    else:
        print("MM encoder attention 使用 AUTO；实际自动选择仅以启动日志为准")

    config = model_config.hf_config
    timestamp_token_id = int(config.timestamp_token_id)
    timestamp_segment_time = float(config.timestamp_segment_time)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        try:
            oracle_row = oracle_by_id[sample["id"]]
            audio, duration = _load_audio(sample["audio_path"])
            words = oracle_row.get("words")
            if not isinstance(words, list) or not words or any(
                not isinstance(word, str) or not word for word in words
            ):
                raise RuntimeError(f"{sample['id']} oracle words 非法")
            request = {
                "prompt": _build_prompt(words),
                "multi_modal_data": {"audio": audio},
            }
            calls_before = (
                int(cross_window_state["calls"])
                if cross_window_state is not None
                else 0
            )
            started = perf_counter()
            outputs = llm.encode([request], pooling_task="token_classify")
            elapsed_seconds = perf_counter() - started
            if cross_window_state is not None:
                calls_after = int(cross_window_state["calls"])
                if calls_after <= calls_before:
                    raise RuntimeError(
                        "vLLM audio cross-window 诊断补丁未命中模型执行；"
                        "拒绝把未生效结果写入报告"
                    )
            if len(outputs) != 1:
                raise RuntimeError(f"{sample['id']} vLLM 返回数量异常：{len(outputs)}")
            units, buckets, candidates = _units_from_vllm(
                outputs[0], words, timestamp_token_id, timestamp_segment_time,
                duration, sample["id"], args.diagnostic_top_k,
            )
            comparison = _compare_units(
                units, oracle_row.get("units"), args.timestamp_tolerance_ms, sample["id"]
            )
            diagnosis = _bucket_diagnosis(
                buckets, oracle_row.get("buckets"),
                candidates, oracle_row.get("top_candidates"),
            )
            rows.append({
                **_sample_provenance(sample, duration),
                **comparison,
                "bucket_diagnosis": diagnosis,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "passed": comparison["within_tolerance"],
            })
            if comparison["within_tolerance"]:
                print(
                    f"compare {sample['language']}: {comparison['unit_count']} 单元，"
                    f"最大偏差 {comparison['max_timestamp_delta_ms']:.3f} ms，通过"
                )
            else:
                print(
                    f"compare {sample['language']}：不通过："
                    f"{comparison['exceeded_count']} 处超过 {args.timestamp_tolerance_ms:.3f} ms，"
                    f"最大 {comparison['max_timestamp_delta_ms']:.3f} ms"
                )
                _print_diagnosis(comparison, diagnosis, timestamp_segment_time)
        except Exception as error:  # noqa: BLE001 - 必须汇总全部 11 语种，不在首错停止。
            if cross_window_diagnostic and (
                "cross-window 诊断补丁未命中模型执行" in str(error)
                or (
                    cross_window_state is not None
                    and int(cross_window_state["calls"]) == 0
                )
            ):
                raise RuntimeError(
                    "vLLM audio cross-window 诊断补丁未进入模型执行进程；"
                    "本轮立即作废，不写比较报告"
                ) from error
            rows.append({
                "id": sample["id"],
                "language": sample["language"],
                "audio_filename": sample["audio_path"].name,
                "text_filename": sample["text_path"].name,
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            print(f"compare {sample['language']}：失败：{type(error).__name__}: {error}")

    passed_rows = [row for row in rows if row["passed"]]
    diagnostic_only = diagnostic_oracle or cross_window_diagnostic
    return {
        "schema_version": 1,
        "experiment_id": (
            "PERF-ARCH-001-PHASE-A-COMPARE-VLLM-AUDIO-CROSS-WINDOW-DIAGNOSTIC"
            if cross_window_diagnostic
            else "PERF-ARCH-001-PHASE-A-COMPARE"
        ),
        "diagnostic_only": diagnostic_only,
        "runtime": {
            "vllm": actual_vllm,
            "numpy": importlib.metadata.version("numpy"),
            "soundfile": importlib.metadata.version("soundfile"),
        },
        "model": {"path": str(model_dir), **identity},
        "oracle": {
            "filename": expected_path.name,
            "sha256": _sha256(expected_path),
            "qwen_asr": oracle["runtime"]["qwen_asr"],
            "torch": oracle["runtime"]["torch"],
            "dtype": oracle_dtype,
            "audio_block_mask_diagnostic": diagnostic_oracle,
            "semantics": oracle_parameters.get("oracle_semantics"),
        },
        "parameters": {
            "dtype": args.dtype,
            "oracle_dtype": oracle_dtype,
            "dtype_mismatch_explicitly_allowed": args.allow_oracle_dtype_mismatch,
            "audio_block_mask_diagnostic_oracle": diagnostic_oracle,
            "diagnostic_oracle_explicitly_allowed": (
                args.allow_audio_block_mask_diagnostic_oracle
            ),
            "vllm_audio_cross_window_diagnostic": cross_window_diagnostic,
            "vllm_enable_v1_multiprocessing": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ),
            "vllm_audio_cross_window_patch": (
                dict(cross_window_state) if cross_window_state is not None else None
            ),
            "enforce_eager": args.enforce_eager,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "timestamp_tolerance_ms": args.timestamp_tolerance_ms,
            "diagnostic_top_k": args.diagnostic_top_k,
            "max_num_seqs": 1,
            "enable_prefix_caching": False,
            "mm_encoder_attn_backend_requested": backend["requested"],
            "mm_encoder_attn_backend_supported": backend["supported"],
            "mm_encoder_attn_backend_configured": configured_backend,
            "mm_encoder_attn_backend_actual": (
                configured_backend if explicit_backend else None
            ),
            "mm_encoder_attn_backend_verified": explicit_backend,
            "mm_encoder_attn_backend_evidence": (
                "model_config.multimodal_config 回读；平台显式后端不支持时初始化失败"
                if explicit_backend
                else "AUTO 无法从控制进程确定，必须核对启动日志"
            ),
        },
        "languages": list(_SUPPORTED_LANGUAGES),
        "samples": rows,
        "summary": {
            "passed": len(passed_rows),
            "failed": len(rows) - len(passed_rows),
            "total": len(_SUPPORTED_LANGUAGES),
            "all_passed": len(passed_rows) == len(_SUPPORTED_LANGUAGES),
            "max_timestamp_delta_ms": max(
                (row["max_timestamp_delta_ms"] for row in rows
                 if "max_timestamp_delta_ms" in row),
                default=None,
            ),
            "max_abs_bucket_delta": max(
                (row["bucket_diagnosis"]["max_abs_bucket_delta"] for row in rows
                 if row.get("bucket_diagnosis", {}).get("available")),
                default=None,
            ),
            "languages_with_bucket_mismatch": [
                row["language"] for row in rows
                if row.get("bucket_diagnosis", {}).get("available")
                and not row["bucket_diagnosis"]["identical_buckets"]
            ],
        },
    }


def _check_one(raw: Any, index: int, seen_ids: set[str], seen_languages: set[str]) -> dict[str, Any]:
    """逐项检查 manifest 条目，把问题收集为 problems 而不是首错抛出。"""
    row: dict[str, Any] = {"index": index, "problems": []}
    if not isinstance(raw, dict):
        row["problems"].append("条目必须是对象")
        return row

    identifier = str(raw.get("id", "")).strip()
    row["id"] = identifier or None
    if not identifier:
        row["problems"].append("id 为空")
    elif identifier in seen_ids:
        row["problems"].append(f"id 重复：{identifier}")
    else:
        seen_ids.add(identifier)

    try:
        language = _canonical_language(raw.get("language"))
        row["language"] = language
        if language in seen_languages:
            row["problems"].append(f"语种重复：{language}")
        else:
            seen_languages.add(language)
    except ValueError as error:
        row["language"] = None
        row["problems"].append(str(error))

    for key, field in (("audio", "audio_path"), ("text", "text_path")):
        value = str(raw.get(key, "")).strip()
        if not value:
            row["problems"].append(f"{key} 路径为空")
            row[field] = None
            continue
        resolved = Path(value).resolve()
        row[field] = resolved
        if not resolved.is_file():
            row["problems"].append(f"{key} 不存在：{resolved}")

    audio_path = row.get("audio_path")
    if isinstance(audio_path, Path) and audio_path.is_file():
        try:
            _, duration = _load_audio(audio_path)
            row["duration_seconds"] = round(duration, 3)
            row["audio_sha256"] = _sha256(audio_path)
        except (ValueError, RuntimeError) as error:
            row["problems"].append(str(error))
        except Exception as error:  # noqa: BLE001 - 解码失败也要归到该样本而不是中断全表。
            row["problems"].append(f"音频无法解码：{type(error).__name__}: {error}")

    text_path = row.get("text_path")
    if isinstance(text_path, Path) and text_path.is_file():
        try:
            text = text_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as error:
            row["problems"].append(f"参考文本不是 UTF-8：{error}")
        else:
            if not text:
                row["problems"].append("参考文本为空")
            else:
                row["text"] = text
                row["text_chars"] = len(text)
                row["text_sha256"] = _sha256(text_path)
    return row


def _run_check(args: argparse.Namespace) -> dict[str, Any]:
    """只做 CPU 侧预检：manifest 结构、音频格式、文本可读性和官方分词，不加载模型权重。"""
    manifest_path = Path(args.manifest).resolve()
    global_problems: list[str] = []
    payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        global_problems.append("manifest 的 schema_version 必须为 1")
    raw_samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(raw_samples, list):
        global_problems.append("manifest 的 samples 必须是数组")
        raw_samples = []
    elif len(raw_samples) != len(_SUPPORTED_LANGUAGES):
        global_problems.append(
            f"manifest 必须恰好包含 {len(_SUPPORTED_LANGUAGES)} 个语种样本，实际 {len(raw_samples)}"
        )

    processor = None
    if args.processor:
        try:
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor

            actual_qwen_asr = importlib.metadata.version("qwen-asr")
            if actual_qwen_asr != _EXPECTED_QWEN_ASR_VERSION:
                global_problems.append(
                    f"需要 qwen-asr {_EXPECTED_QWEN_ASR_VERSION}，实际 {actual_qwen_asr}"
                )
            processor = Qwen3ForceAlignProcessor()
        except Exception as error:  # noqa: BLE001 - 缺依赖必须显式报错，不静默跳过分词预检。
            global_problems.append(
                f"无法加载官方 Qwen3ForceAlignProcessor（{type(error).__name__}: {error}）；"
                "确认在目标环境执行，或显式传 --no-processor 跳过分词预检"
            )

    seen_ids: set[str] = set()
    seen_languages: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_samples):
        row = _check_one(raw, index, seen_ids, seen_languages)
        if processor is not None and row.get("text") and row.get("language"):
            try:
                words, prompt = processor.encode_timestamp(row["text"], row["language"])
                if not words:
                    row["problems"].append("官方 processor 未切出任何单元")
                elif prompt != _build_prompt(words):
                    row["problems"].append("官方 prompt 与本工具构造不一致")
                else:
                    row["unit_count"] = len(words)
                    duration = row.get("duration_seconds")
                    if duration:
                        density = len(words) / float(duration)
                        row["units_per_second"] = round(density, 2)
                        # 仅提示：密度异常通常意味着文本与音频内容不对应，但这不构成门禁。
                        if density > 12.0 or density < 0.3:
                            row["advisories"] = [
                                f"单元密度 {density:.2f}/秒 偏离常见范围，请人工确认文本与音频内容对应"
                            ]
            except Exception as error:  # noqa: BLE001 - 分词失败按样本归因。
                row["problems"].append(
                    f"官方 encode_timestamp 失败：{type(error).__name__}: {error}"
                )
        row.pop("text", None)
        rows.append(row)

    missing = sorted(set(_SUPPORTED_LANGUAGES) - seen_languages)
    if missing:
        global_problems.append(f"manifest 缺少语种：{missing}")

    for row in rows:
        label = row.get("language") or row.get("id") or f"#{row['index']}"
        if row["problems"]:
            print(f"check {label}：不通过")
            for problem in row["problems"]:
                print(f"    - {problem}")
            continue
        details = [f"{row.get('duration_seconds')}s", f"{row.get('text_chars')} 字符"]
        if row.get("unit_count") is not None:
            details.append(f"{row['unit_count']} 单元 / {row.get('units_per_second')} 每秒")
        print(f"check {label}：通过（{'，'.join(str(item) for item in details)}）")
        for advisory in row.get("advisories", ()):
            print(f"    提示：{advisory}")
    for problem in global_problems:
        print(f"manifest 级问题：{problem}")

    passed = [row for row in rows if not row["problems"]]
    all_passed = not global_problems and len(passed) == len(_SUPPORTED_LANGUAGES)
    for row in rows:
        for key in ("audio_path", "text_path"):
            if isinstance(row.get(key), Path):
                row[key] = str(row[key])
    return {
        "schema_version": 1,
        "experiment_id": "PERF-ARCH-001-PHASE-A-CHECK",
        "runtime": {
            "numpy": importlib.metadata.version("numpy"),
            "soundfile": importlib.metadata.version("soundfile"),
            "processor_checked": processor is not None,
        },
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "languages": list(_SUPPORTED_LANGUAGES),
        "manifest_problems": global_problems,
        "samples": rows,
        "summary": {
            "passed": len(passed),
            "failed": len(rows) - len(passed),
            "total": len(_SUPPORTED_LANGUAGES),
            "all_passed": all_passed,
            "missing_languages": missing,
        },
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, help="11 语种样本 manifest；路径按项目根解析")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--diagnostic-top-k", type=int, default=5,
        help="每个 timestamp 位置记录的候选桶数量（2～50），仅用于定因，不参与门禁",
    )
    parser.add_argument("--output-json", required=True, help="原子写入报告；不得提交含正文的 oracle")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PERF-ARCH-001 Phase A：11 语种 PyTorch/vLLM 时间戳语义对照"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="不加载模型的 CPU 预检：manifest、音频格式与官方分词"
    )
    check_parser.add_argument("--manifest", required=True, help="11 语种样本 manifest")
    check_parser.add_argument("--output-json", default=None, help="可选，原子写入预检报告")
    check_parser.add_argument(
        "--processor", action=argparse.BooleanOptionalAction, default=True,
        help="默认调用官方 encode_timestamp() 预检分词；缺依赖时必须显式 --no-processor",
    )

    oracle_parser = subparsers.add_parser("oracle", help="生成 qwen-asr 0.0.6 PyTorch oracle")
    _add_common_arguments(oracle_parser)
    oracle_parser.add_argument(
        "--model", default="models/qwen3-forced-aligner-0.6b/pt",
        help="ModelScope 固定 revision 的 PyTorch Aligner 目录",
    )
    oracle_parser.add_argument("--device", default="cuda:0")
    oracle_parser.add_argument(
        "--diagnostic-oracle-audio-block-mask", action="store_true",
        help=(
            "仅用于定因：让 PyTorch SDPA 调用源码已有的音频窗口块对角 mask；"
            "输出不得作为生产 Phase A oracle"
        ),
    )

    compare_parser = subparsers.add_parser("compare", help="vLLM token-classify 对照 oracle")
    _add_common_arguments(compare_parser)
    compare_parser.add_argument(
        "--model", default="models/qwen3-forced-aligner-0.6b/vllm-spike",
        help="ModelScope 固定 revision 的 vLLM Aligner 目录",
    )
    compare_parser.add_argument("--expected-json", required=True, help="oracle 子命令生成的 JSON")
    compare_parser.add_argument(
        "--allow-oracle-dtype-mismatch", action="store_true",
        help="显式固定生产 oracle dtype、只改变 vLLM 候选 dtype；默认拒绝误混口径",
    )
    compare_parser.add_argument(
        "--allow-audio-block-mask-diagnostic-oracle", action="store_true",
        help=(
            "显式允许只用于定因的 PyTorch 音频块对角 mask oracle；"
            "该 compare 永不构成 Phase A 晋级证据"
        ),
    )
    compare_parser.add_argument(
        "--diagnostic-vllm-audio-cross-window", action="store_true",
        help=(
            "仅用于定因：关闭 V1 独立 EngineCore，并让 vLLM Torch SDPA 忽略 "
            "cu_seqlens 分窗，以复刻生产 PyTorch oracle 的跨窗口语义；"
            "该 compare 永不构成 Phase A 晋级证据"
        ),
    )
    compare_parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True,
        help="Phase A 默认 eager，图模式属于独立性能变量",
    )
    compare_parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    compare_parser.add_argument("--timestamp-tolerance-ms", type=float, default=1.0)
    compare_parser.add_argument(
        "--mm-encoder-attn-backend", default="auto",
        help=(
            "MM encoder/音频塔 attention 后端；auto 保留自动选择，显式值按当前平台 "
            "get_supported_vit_attn_backends() 校验，不支持或未生效时直接失败"
        ),
    )

    args = parser.parse_args()
    output_path = Path(args.output_json) if args.output_json else None
    if output_path is not None and output_path.suffix.lower() != ".json":
        parser.error("--output-json 必须以 .json 结尾")
    if args.command in ("oracle", "compare") and not 2 <= args.diagnostic_top_k <= 50:
        parser.error("--diagnostic-top-k 必须在 2～50 之间")
    if args.command == "compare":
        if not 0 < args.gpu_memory_utilization < 1:
            parser.error("--gpu-memory-utilization 必须在 0～1 之间")
        if args.timestamp_tolerance_ms < 0:
            parser.error("--timestamp-tolerance-ms 不能小于 0")
        report = _run_compare(args)
    elif args.command == "check":
        report = _run_check(args)
    else:
        report = _run_oracle(args)

    if output_path is not None:
        _write_json(output_path, report)
    summary = report["summary"]
    location = f"，报告 {output_path}" if output_path is not None else ""
    print(f"Phase A {args.command}：{summary['passed']}/{summary['total']}{location}")
    if report.get("diagnostic_only"):
        print("诊断模式：该报告修改了 oracle 或 candidate 语义，不得作为 Phase A 晋级证据")
        if args.command == "compare":
            return 2
    passed = summary.get("all_passed", summary["passed"] == summary["total"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
