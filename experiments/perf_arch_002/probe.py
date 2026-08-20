#!/usr/bin/env python3
"""PERF-ARCH-002：TensorRT FP16 路线的最小导出可行性探针。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import traceback
from pathlib import Path
from types import MethodType
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_EXPECTED_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
_EXPECTED_REVISION = "cf1c50164ea3ac48240d12bef5ead74aee0720cc"
_EXPECTED_QWEN_ASR = "0.0.6"
_DYNAMIC_AUDIO_WINDOW = 100
_DYNAMIC_AUDIO_CHUNKS_MIN = 2
_DYNAMIC_AUDIO_CHUNKS_MAX = 32


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


def _class_node(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise RuntimeError(f"源码缺少类 {name}")


def _method_node(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"{class_node.name} 缺少方法 {name}")


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _assigned_self_members(method: ast.FunctionDef) -> set[str]:
    members: set[str] = set()
    for node in ast.walk(method):
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            name = _dotted_name(target)
            if name and name.startswith("self."):
                members.add(name.removeprefix("self."))
    return members


def _called_names(method: ast.FunctionDef) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name:
                calls.append((name, node.lineno))
    return calls


def _run_static(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.source_root).resolve()
    modeling = root / "qwen_asr/core/transformers_backend/modeling_qwen3_asr.py"
    aligner = root / "qwen_asr/inference/qwen3_forced_aligner.py"
    for path in (modeling, aligner):
        if not path.is_file():
            raise FileNotFoundError(f"缺少官方源码：{path}")

    modeling_tree = ast.parse(modeling.read_text(encoding="utf-8"), filename=str(modeling))
    aligner_tree = ast.parse(aligner.read_text(encoding="utf-8"), filename=str(aligner))
    thinker_class = _class_node(modeling_tree, "Qwen3ASRThinkerForConditionalGeneration")
    thinker_init = _method_node(thinker_class, "__init__")
    thinker_audio = _method_node(thinker_class, "get_audio_features")
    thinker_forward = _method_node(thinker_class, "forward")
    aligner_class = _class_node(aligner_tree, "Qwen3ForcedAligner")
    align_method = _method_node(aligner_class, "align")

    members = _assigned_self_members(thinker_init)
    audio_calls = _called_names(thinker_audio)
    forward_calls = _called_names(thinker_forward)
    align_calls = _called_names(align_method)
    forward_order = {
        name: min(line for called, line in forward_calls if called == name)
        for name in ("self.get_audio_features", "self.model", "self.lm_head")
        if any(called == name for called, _ in forward_calls)
    }
    checks = {
        "thinker_members": {"audio_tower", "model", "lm_head"}.issubset(members),
        "audio_helper_calls_tower": any(
            name == "self.audio_tower" for name, _ in audio_calls
        ),
        "forward_calls_all_segments": len(forward_order) == 3,
        "forward_order": (
            len(forward_order) == 3
            and forward_order["self.get_audio_features"]
            < forward_order["self.model"]
            < forward_order["self.lm_head"]
        ),
        "align_calls_thinker": any(name == "self.model.thinker" for name, _ in align_calls),
    }
    return {
        "schema_version": 1,
        "experiment_id": "PERF-ARCH-002-STATIC-PROBE",
        "source": {
            "root": str(root),
            "modeling": str(modeling),
            "modeling_sha256": _sha256(modeling),
            "aligner": str(aligner),
            "aligner_sha256": _sha256(aligner),
        },
        "observed": {
            "thinker_members": sorted(members),
            "audio_tower_calls": [
                line for name, line in audio_calls if name == "self.audio_tower"
            ],
            "forward_order": forward_order,
            "align_thinker_calls": [line for name, line in align_calls if name == "self.model.thinker"],
        },
        "checks": checks,
        "summary": {"passed": all(checks.values()), "total": len(checks)},
    }


def _model_identity(model_dir: Path) -> dict[str, str]:
    manifest_path = model_dir / ".modelscope-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 ModelScope manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {
        "repo_id": manifest.get("repo_id"),
        "revision": manifest.get("revision"),
    }
    expected = {"repo_id": _EXPECTED_REPO, "revision": _EXPECTED_REVISION}
    if identity != expected:
        raise RuntimeError(f"Aligner 模型身份不符合固定值：{identity}")
    return identity


def _tensor_meta(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(getattr(value, "device", "unknown")),
        }
    if hasattr(value, "last_hidden_state"):
        return _tensor_meta(value.last_hidden_state)
    if isinstance(value, (tuple, list)):
        return [_tensor_meta(item) for item in value[:3]]
    return {"type": type(value).__name__}


def _extract_logits(value: Any, torch: Any) -> Any:
    logits = getattr(value, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, torch.Tensor) and item.ndim == 3]
        if tensors:
            return tensors[-1]
    raise RuntimeError(f"无法从导出回放结果提取 logits：{type(value).__name__}")


def _compare_logits(
    reference_logits: Any,
    candidate_logits: Any,
    timestamp_mask: Any,
    reference_buckets: Any,
    torch: Any,
) -> dict[str, Any]:
    same_shape = tuple(reference_logits.shape) == tuple(candidate_logits.shape)
    finite = bool(torch.isfinite(candidate_logits).all().item())
    logits_identical = same_shape and bool(torch.equal(reference_logits, candidate_logits))
    max_delta = None
    if same_shape:
        max_delta = float((reference_logits - candidate_logits).abs().max().item())

    buckets_identical = False
    bucket_count = None
    if candidate_logits.ndim == 3 and tuple(candidate_logits.shape[:2]) == tuple(
        timestamp_mask.shape
    ):
        candidate_buckets = candidate_logits.argmax(dim=-1)[timestamp_mask].detach().cpu()
        bucket_count = int(candidate_buckets.numel())
        buckets_identical = bool(torch.equal(reference_buckets, candidate_buckets))

    return {
        "logits_shape": list(candidate_logits.shape),
        "same_shape": same_shape,
        "finite": finite,
        "logits_identical": logits_identical,
        "max_abs_logit_delta": max_delta,
        "timestamp_bucket_count": bucket_count,
        "timestamp_buckets_identical": buckets_identical,
    }


def _comparison_passed(comparison: dict[str, Any] | None, require_exact_logits: bool) -> bool:
    return bool(
        comparison
        and comparison["same_shape"]
        and comparison["finite"]
        and comparison["timestamp_buckets_identical"]
        and (comparison["logits_identical"] or not require_exact_logits)
    )


def _validate_dynamic_shape_input(
    tensor_inputs: dict[str, Any], label: str, torch: Any, allow_short: bool = False
) -> dict[str, int]:
    required = {
        "input_ids",
        "attention_mask",
        "feature_attention_mask",
        "input_features",
    }
    missing = sorted(required - set(tensor_inputs))
    if missing:
        raise RuntimeError(f"{label}缺少动态 shape 必需 tensor：{missing}")

    input_ids = tensor_inputs["input_ids"]
    attention_mask = tensor_inputs["attention_mask"]
    feature_mask = tensor_inputs["feature_attention_mask"]
    input_features = tensor_inputs["input_features"]
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise RuntimeError(f"{label}文本 tensor 必须为二维")
    if feature_mask.ndim != 2 or input_features.ndim != 3:
        raise RuntimeError(f"{label}音频 tensor rank 不符合预期")
    if not all(
        value.shape[0] == 1
        for value in (input_ids, attention_mask, feature_mask, input_features)
    ):
        raise RuntimeError(f"{label}动态 shape 当前只允许 batch=1")
    if tuple(input_ids.shape) != tuple(attention_mask.shape):
        raise RuntimeError(f"{label}input_ids 与 attention_mask shape 不一致")
    if int(feature_mask.shape[1]) != int(input_features.shape[2]):
        raise RuntimeError(f"{label}音频 mask 与 mel 帧数不一致")
    if not bool(torch.all(attention_mask == 1).item()):
        raise RuntimeError(f"{label}动态文本改写只接受无 padding 的 attention_mask")
    if not bool(torch.all(feature_mask == 1).item()):
        raise RuntimeError(f"{label}动态音频改写只接受无 padding 的 feature_attention_mask")
    audio_frames = int(input_features.shape[2])
    if not 0 < audio_frames <= _DYNAMIC_AUDIO_WINDOW * _DYNAMIC_AUDIO_CHUNKS_MAX:
        raise RuntimeError(
            f"{label}动态音频帧数必须位于 "
            f"[1, {_DYNAMIC_AUDIO_WINDOW * _DYNAMIC_AUDIO_CHUNKS_MAX}]，"
            f"实际为 {audio_frames}"
        )
    audio_full_chunks, audio_tail_frames = divmod(
        audio_frames, _DYNAMIC_AUDIO_WINDOW
    )
    audio_padded_chunks = audio_full_chunks + int(audio_tail_frames > 0)
    if audio_padded_chunks < _DYNAMIC_AUDIO_CHUNKS_MIN and not allow_short:
        raise RuntimeError(
            f"{label}当前动态 T1 至少需要 {_DYNAMIC_AUDIO_CHUNKS_MIN} 个卷积 chunk；"
            "更短输入必须使用显式短音频规范化"
        )
    return {
        "batch_size": 1,
        "text_tokens": int(input_ids.shape[1]),
        "audio_frames": audio_frames,
        "audio_full_chunks": audio_full_chunks,
        "audio_tail_frames": audio_tail_frames,
        "audio_padded_chunks": audio_padded_chunks,
        "mel_bins": int(input_features.shape[1]),
    }


def _tail_output_bucket(tail_frames: int) -> dict[str, int]:
    if not 0 < tail_frames < _DYNAMIC_AUDIO_WINDOW:
        raise RuntimeError(f"尾帧必须位于 [1, 99]，实际为 {tail_frames}")
    output_length = _feature_output_length(tail_frames)
    canonical_tail = min(output_length * 8, _DYNAMIC_AUDIO_WINDOW - 1)
    return {
        "tail_frames": tail_frames,
        "output_length": output_length,
        "canonical_tail_frames": canonical_tail,
        "canonical_padding": canonical_tail - tail_frames,
    }


def _canonicalize_tail_bucket_inputs(
    tensor_inputs: dict[str, Any], torch: Any
) -> tuple[dict[str, Any], dict[str, int]]:
    from torch.nn import functional as functional

    contract = _validate_dynamic_shape_input(tensor_inputs, "尾帧桶输入", torch)
    plan = _tail_output_bucket(contract["audio_tail_frames"])
    padding = plan["canonical_padding"]
    canonical_inputs = dict(tensor_inputs)
    if padding:
        canonical_inputs["input_features"] = functional.pad(
            tensor_inputs["input_features"], (0, padding)
        )
        canonical_inputs["feature_attention_mask"] = functional.pad(
            tensor_inputs["feature_attention_mask"], (0, padding), value=1
        )
    canonical_contract = _validate_dynamic_shape_input(
        canonical_inputs, "尾帧桶规范化输入", torch
    )
    if _feature_output_length(canonical_contract["audio_tail_frames"]) != plan[
        "output_length"
    ]:
        raise RuntimeError("尾帧桶规范化改变了 CNN 输出长度")
    return canonical_inputs, {**plan, "canonical_audio_frames": canonical_contract["audio_frames"]}


def _normalize_short_tail_to_full_chunks_inputs(
    tensor_inputs: dict[str, Any], torch: Any
) -> tuple[dict[str, Any], dict[str, int]]:
    from torch.nn import functional as functional

    contract = _validate_dynamic_shape_input(
        tensor_inputs, "短音频原始输入", torch, allow_short=True
    )
    if contract["audio_full_chunks"] not in {0, 1}:
        raise RuntimeError(
            "短音频规范化只接受 audio_full_chunks=0/1，"
            f"实际为 {contract['audio_full_chunks']}"
        )
    tail_frames = contract["audio_tail_frames"]
    if tail_frames == 0:
        raise RuntimeError("短音频规范化当前只处理带非零尾块的输入")
    bucket = _tail_output_bucket(tail_frames)
    tail_completion_padding = _DYNAMIC_AUDIO_WINDOW - tail_frames
    guard_chunk_padding = _DYNAMIC_AUDIO_WINDOW
    total_padding = tail_completion_padding + guard_chunk_padding
    normalized_inputs = dict(tensor_inputs)
    normalized_inputs["input_features"] = functional.pad(
        tensor_inputs["input_features"], (0, total_padding)
    )
    normalized_inputs["feature_attention_mask"] = functional.pad(
        tensor_inputs["feature_attention_mask"], (0, total_padding), value=1
    )
    normalized_contract = _validate_dynamic_shape_input(
        normalized_inputs, "短音频完整 chunk 候选", torch
    )
    if normalized_contract["audio_tail_frames"] != 0:
        raise RuntimeError("短音频规范化后仍存在尾帧")
    if normalized_contract["audio_full_chunks"] != contract["audio_full_chunks"] + 2:
        raise RuntimeError("短音频规范化后的完整 chunk 数不符合预期")
    output_per_full_chunk = _feature_output_length(_DYNAMIC_AUDIO_WINDOW)
    trailing_output_drop = (
        output_per_full_chunk - bucket["output_length"]
        + output_per_full_chunk
    )
    return normalized_inputs, {
        **bucket,
        "tail_completion_padding": tail_completion_padding,
        "guard_chunk_padding": guard_chunk_padding,
        "total_padding": total_padding,
        "normalized_audio_frames": normalized_contract["audio_frames"],
        "normalized_full_chunks": normalized_contract["audio_full_chunks"],
        "trailing_output_drop": trailing_output_drop,
    }


def _load_audio(path: Path) -> tuple[Any, float]:
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(path) as source:
        if source.samplerate != 16000 or source.channels != 1 or source.frames <= 0:
            raise ValueError("目标机探针只接受非空 16 kHz 单声道音频")
        duration = source.frames / source.samplerate
        if duration > 32.0:
            raise ValueError("目标机探针音频不得超过 32 秒")
        audio = source.read(dtype="float32", always_2d=False)
    return np.ascontiguousarray(audio), duration


def _feature_output_length(length: int) -> int:
    remainder = length % 100
    feature_length = (remainder - 1) // 2 + 1
    return ((feature_length - 1) // 2 + 1 - 1) // 2 + 1 + (length // 100) * 13


def _install_fixed_audio_shape_specialization(
    thinker: Any, tensor_inputs: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from torch.nn import functional as functional
    from torch.nn.utils.rnn import pad_sequence

    feature_mask = tensor_inputs.get("feature_attention_mask")
    input_features = tensor_inputs.get("input_features")
    if feature_mask is None or input_features is None:
        raise RuntimeError("固定音频 shape 专门化需要 input_features 和 feature_attention_mask")
    if feature_mask.ndim != 2 or feature_mask.shape[0] != 1 or input_features.shape[0] != 1:
        raise RuntimeError("固定音频 shape 专门化当前只允许 batch=1、单音频")

    feature_length = int(feature_mask[0].sum().item())
    if feature_length <= 0 or feature_length > int(input_features.shape[-1]):
        raise RuntimeError(f"非法有效 mel 长度：{feature_length}")

    audio_tower = thinker.audio_tower
    window = int(audio_tower.n_window) * 2
    chunk_count = (feature_length + window - 1) // window
    tail = feature_length % window or window
    chunk_lengths = [window] * (chunk_count - 1) + [tail]
    chunk_aftercnn_lengths = [_feature_output_length(length) for length in chunk_lengths]
    aftercnn_length = _feature_output_length(feature_length)
    window_aftercnn = max(chunk_aftercnn_lengths) * (
        int(audio_tower.n_window_infer) // window
    )
    if window_aftercnn <= 0:
        raise RuntimeError(f"非法 attention 窗口长度：{window_aftercnn}")
    cu_segments = [window_aftercnn] * (aftercnn_length // window_aftercnn)
    remainder = aftercnn_length % window_aftercnn
    if remainder:
        cu_segments.append(remainder)
    if sum(cu_segments) != sum(chunk_aftercnn_lengths):
        raise RuntimeError(
            "固定 shape 计划与官方 chunk/CNN 长度不闭合："
            f"cu={sum(cu_segments)}, hidden={sum(chunk_aftercnn_lengths)}"
        )

    def fixed_audio_forward(
        module: Any,
        current_features: Any,
        feature_lens: Any = None,
        aftercnn_lens: Any = None,
    ) -> Any:
        del feature_lens, aftercnn_lens
        chunk_list = current_features.transpose(0, 1).split(tuple(chunk_lengths), dim=0)
        padded_feature = pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(int(module.conv_chunksize), dim=0):
            padded_embed = functional.gelu(module.conv2d1(chunk))
            padded_embed = functional.gelu(module.conv2d2(padded_embed))
            padded_embed = functional.gelu(module.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        batch, channels, frequency, time = padded_embed.size()
        padded_embed = module.conv_out(
            padded_embed.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch, time, channels * frequency)
        )
        positional_embedding = (
            module.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = torch.cat(
            [
                padded_embed[index, :length]
                for index, length in enumerate(chunk_aftercnn_lengths)
            ],
            dim=0,
        )
        cu_seqlens = torch.tensor(
            [0, *cu_segments], dtype=torch.int32, device=hidden_states.device
        ).cumsum(-1, dtype=torch.int32)
        for encoder_layer in module.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)[0]
        hidden_states = module.ln_post(hidden_states)
        hidden_states = module.proj1(hidden_states)
        hidden_states = module.act(hidden_states)
        return module.proj2(hidden_states)

    def fixed_get_audio_features(
        module: Any,
        current_features: Any,
        feature_attention_mask: Any = None,
        audio_feature_lengths: Any = None,
    ) -> Any:
        del feature_attention_mask, audio_feature_lengths
        return module.audio_tower(current_features[0, :, :feature_length])

    audio_tower.forward = MethodType(fixed_audio_forward, audio_tower)
    thinker.get_audio_features = MethodType(fixed_get_audio_features, thinker)
    return {
        "enabled": True,
        "batch_size": 1,
        "feature_length": feature_length,
        "window": window,
        "chunk_lengths": chunk_lengths,
        "chunk_aftercnn_lengths": chunk_aftercnn_lengths,
        "aftercnn_length": aftercnn_length,
        "cu_segments": cu_segments,
    }


def _install_dynamic_audio_shape_rewrite(
    thinker: Any,
    tensor_inputs: dict[str, Any],
    torch: Any,
    trailing_output_drop_override: int | None = None,
) -> dict[str, Any]:
    from torch.nn import functional as functional

    feature_mask = tensor_inputs.get("feature_attention_mask")
    input_features = tensor_inputs.get("input_features")
    if feature_mask is None or input_features is None:
        raise RuntimeError("动态音频 shape 改写需要 input_features 和 feature_attention_mask")
    if feature_mask.ndim != 2 or feature_mask.shape[0] != 1 or input_features.shape[0] != 1:
        raise RuntimeError("动态音频 shape 改写当前只允许 batch=1、单音频")
    if not bool(torch.all(feature_mask == 1).item()):
        raise RuntimeError("动态音频 shape 改写只接受无 padding 的 feature_attention_mask")

    audio_tower = thinker.audio_tower
    attention_implementation = str(audio_tower.config._attn_implementation)
    if attention_implementation != "sdpa":
        raise RuntimeError(
            "动态音频 shape 改写依赖生产 SDPA 忽略 cu_seqlens 的语义，"
            f"实际为 {attention_implementation}"
        )
    window = int(audio_tower.n_window) * 2
    if window != _DYNAMIC_AUDIO_WINDOW:
        raise RuntimeError(
            f"动态音频契约要求 window={_DYNAMIC_AUDIO_WINDOW}，实际为 {window}"
        )
    feature_length = int(input_features.shape[-1])
    full_chunks, tail_frames = divmod(feature_length, window)
    tail_padding = (window - tail_frames) % window
    output_per_full_chunk = _feature_output_length(window)
    tail_output_length = _feature_output_length(tail_frames) if tail_frames else 0
    natural_tail_output_drop = (
        output_per_full_chunk - tail_output_length if tail_frames else 0
    )
    tail_output_drop = (
        natural_tail_output_drop
        if trailing_output_drop_override is None
        else int(trailing_output_drop_override)
    )
    if tail_output_drop < 0:
        raise RuntimeError("动态音频输出裁剪量不得为负数")
    if trailing_output_drop_override is not None and tail_frames != 0:
        raise RuntimeError("显式尾部输出裁剪只允许用于完整 chunk 候选输入")

    def dynamic_audio_forward(
        module: Any,
        current_features: Any,
        feature_lens: Any = None,
        aftercnn_lens: Any = None,
    ) -> Any:
        del feature_lens, aftercnn_lens
        padded_features = (
            functional.pad(current_features, (0, tail_padding))
            if tail_padding
            else current_features
        )
        chunks = (
            padded_features.transpose(0, 1)
            .reshape(-1, window, padded_features.shape[0])
            .transpose(1, 2)
        )
        padded_feature = chunks.unsqueeze(1)
        padded_embed = functional.gelu(module.conv2d1(padded_feature))
        padded_embed = functional.gelu(module.conv2d2(padded_embed))
        padded_embed = functional.gelu(module.conv2d3(padded_embed))
        batch, channels, frequency, time = padded_embed.size()
        padded_embed = module.conv_out(
            padded_embed.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch, time, channels * frequency)
        )
        positional_embedding = (
            module.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed.reshape(-1, padded_embed.shape[-1])
        if tail_output_drop:
            hidden_states = hidden_states[:-tail_output_drop]
        cu_seqlens = torch.arange(
            2, dtype=torch.int32, device=hidden_states.device
        ) * hidden_states.shape[0]
        for encoder_layer in module.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)[0]
        hidden_states = module.ln_post(hidden_states)
        hidden_states = module.proj1(hidden_states)
        hidden_states = module.act(hidden_states)
        return module.proj2(hidden_states)

    def dynamic_get_audio_features(
        module: Any,
        current_features: Any,
        feature_attention_mask: Any = None,
        audio_feature_lengths: Any = None,
    ) -> Any:
        del feature_attention_mask, audio_feature_lengths
        return module.audio_tower(current_features[0])

    audio_tower.forward = MethodType(dynamic_audio_forward, audio_tower)
    thinker.get_audio_features = MethodType(dynamic_get_audio_features, thinker)
    return {
        "enabled": True,
        "batch_size": 1,
        "window": window,
        "output_per_full_chunk": output_per_full_chunk,
        "audio_frame_domain": (
            f"{window} * audio_full_chunks + {tail_frames}"
        ),
        "audio_full_chunks_example": full_chunks,
        "audio_tail_frames": tail_frames,
        "tail_padding": tail_padding,
        "tail_output_length": tail_output_length,
        "natural_tail_output_drop": natural_tail_output_drop,
        "tail_output_drop": tail_output_drop,
        "trailing_output_drop_override": trailing_output_drop_override,
        "tail_frames_dynamic": False,
        "attention_implementation": attention_implementation,
        "cu_seqlens_strategy": "single_full_sequence_for_sdpa",
    }


def _install_single_chunk_dynamic_audio_rewrite(
    thinker: Any, tensor_inputs: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from torch.nn import functional as functional

    contract = _validate_dynamic_shape_input(
        tensor_inputs, "原生单 chunk 动态输入", torch, allow_short=True
    )
    if contract["audio_full_chunks"] != 0:
        raise RuntimeError(
            "原生单 chunk 动态改写只接受 audio_full_chunks=0，"
            f"实际为 {contract['audio_full_chunks']}"
        )
    if tensor_inputs.get("feature_attention_mask") is None:
        raise RuntimeError("原生单 chunk 动态改写缺少 feature_attention_mask")

    audio_tower = thinker.audio_tower
    attention_implementation = str(audio_tower.config._attn_implementation)
    if attention_implementation != "sdpa":
        raise RuntimeError(
            "原生单 chunk 动态改写依赖生产 SDPA 单序列语义，"
            f"实际为 {attention_implementation}"
        )
    window = int(audio_tower.n_window) * 2
    if window != _DYNAMIC_AUDIO_WINDOW:
        raise RuntimeError(
            f"原生单 chunk 契约要求 window={_DYNAMIC_AUDIO_WINDOW}，实际为 {window}"
        )
    bucket = _tail_output_bucket(contract["audio_frames"])
    bucket_start = (bucket["output_length"] - 1) * 8 + 1
    bucket_end = min(bucket["output_length"] * 8, window - 1)

    def single_chunk_audio_forward(
        module: Any,
        current_features: Any,
        feature_lens: Any = None,
        aftercnn_lens: Any = None,
    ) -> Any:
        del feature_lens, aftercnn_lens
        padded_embed = functional.gelu(
            module.conv2d1(current_features.unsqueeze(0).unsqueeze(0))
        )
        padded_embed = functional.gelu(module.conv2d2(padded_embed))
        padded_embed = functional.gelu(module.conv2d3(padded_embed))
        batch, channels, frequency, time = padded_embed.size()
        padded_embed = module.conv_out(
            padded_embed.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch, time, channels * frequency)
        )
        positional_embedding = (
            module.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        hidden_states = (padded_embed + positional_embedding).reshape(
            -1, padded_embed.shape[-1]
        )
        cu_seqlens = torch.arange(
            2, dtype=torch.int32, device=hidden_states.device
        ) * hidden_states.shape[0]
        for encoder_layer in module.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)[0]
        hidden_states = module.ln_post(hidden_states)
        hidden_states = module.proj1(hidden_states)
        hidden_states = module.act(hidden_states)
        return module.proj2(hidden_states)

    def single_chunk_get_audio_features(
        module: Any,
        current_features: Any,
        feature_attention_mask: Any = None,
        audio_feature_lengths: Any = None,
    ) -> Any:
        del feature_attention_mask, audio_feature_lengths
        return module.audio_tower(current_features[0])

    audio_tower.forward = MethodType(single_chunk_audio_forward, audio_tower)
    thinker.get_audio_features = MethodType(single_chunk_get_audio_features, thinker)
    return {
        "enabled": True,
        "batch_size": 1,
        "window": window,
        "audio_frame_domain": f"[{bucket_start}, {bucket_end}]",
        "audio_full_chunks_example": 0,
        "audio_tail_frames_example": contract["audio_frames"],
        "output_length": bucket["output_length"],
        "bucket_start": bucket_start,
        "bucket_end": bucket_end,
        "padding": 0,
        "tail_output_drop": 0,
        "tail_frames_dynamic": True,
        "attention_implementation": attention_implementation,
        "cu_seqlens_strategy": "single_native_sequence_for_sdpa",
    }


def _install_fixed_text_mask_specialization(
    thinker: Any, tensor_inputs: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast

    attention_mask = tensor_inputs.get("attention_mask")
    input_ids = tensor_inputs.get("input_ids")
    if attention_mask is None or input_ids is None:
        raise RuntimeError("固定文本 mask 专门化需要 input_ids 和 attention_mask")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise RuntimeError("固定文本 mask 专门化当前只允许 batch=1")

    text_model = thinker.model
    with torch.inference_mode():
        input_embeds = thinker.get_input_embeddings()(input_ids)
        cache_position = torch.arange(
            input_embeds.shape[1], device=input_embeds.device
        )
        position_ids, _ = thinker.get_rope_index(attention_mask)
        fixed_causal_mask = create_causal_mask(
            config=text_model.config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids[0],
        )
    if fixed_causal_mask is None:
        text_model._arch002_fixed_causal_mask = None
        causal_mask_meta = None
    else:
        fixed_causal_mask = fixed_causal_mask.detach().clone()
        text_model.register_buffer(
            "_arch002_fixed_causal_mask", fixed_causal_mask, persistent=False
        )
        causal_mask_meta = _tensor_meta(fixed_causal_mask)

    def fixed_text_forward(
        module: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: Any = None,
        cache_position: Any = None,
        **kwargs: Any,
    ) -> Any:
        del input_ids, attention_mask
        if inputs_embeds is None:
            raise RuntimeError("固定文本 mask 专门化只接受 inputs_embeds")
        if past_key_values is not None or use_cache:
            raise RuntimeError("固定文本 mask 专门化不支持 KV cache")
        if cache_position is None:
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                3, position_ids.shape[0], -1
            )
        if position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        hidden_states = inputs_embeds
        position_embeddings = module.rotary_emb(hidden_states, position_ids)
        for decoder_layer in module.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=module._arch002_fixed_causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = module.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )

    text_model.forward = MethodType(fixed_text_forward, text_model)
    return {
        "enabled": True,
        "batch_size": int(attention_mask.shape[0]),
        "sequence_length": int(attention_mask.shape[1]),
        "attention_implementation": str(text_model.config._attn_implementation),
        "causal_mask": causal_mask_meta,
        "uses_native_causal_attention": fixed_causal_mask is None,
    }


def _run_target(args: argparse.Namespace) -> dict[str, Any]:
    import warnings

    import torch
    from qwen_asr import Qwen3ForcedAligner

    if importlib.metadata.version("qwen-asr") != _EXPECTED_QWEN_ASR:
        raise RuntimeError(f"需要 qwen-asr=={_EXPECTED_QWEN_ASR}")
    if not torch.cuda.is_available() or not (torch.version.cuda or "").startswith("12."):
        raise RuntimeError("目标机探针需要 CUDA 12.x PyTorch 和可用 GPU")
    if args.dynamic_shapes and args.specialize_audio_shapes:
        raise RuntimeError("--dynamic-shapes 与 --specialize-audio-shapes 不能同时使用")
    dynamic_audio_modes = {
        "--canonicalize-tail-bucket": args.canonicalize_tail_bucket,
        "--normalize-short-tail-to-full-chunks": (
            args.normalize_short_tail_to_full_chunks
        ),
        "--direct-short-single-chunk-bucket": (
            args.direct_short_single_chunk_bucket
        ),
    }
    for option, enabled in dynamic_audio_modes.items():
        if enabled and not args.dynamic_shapes:
            raise RuntimeError(f"{option} 必须与 --dynamic-shapes 同时使用")
    enabled_audio_modes = [
        option for option, enabled in dynamic_audio_modes.items() if enabled
    ]
    if len(enabled_audio_modes) > 1:
        raise RuntimeError(
            "动态音频实验模式不能同时使用：" + ", ".join(enabled_audio_modes)
        )
    if args.specialize_text_mask and not (
        args.specialize_audio_shapes or args.dynamic_shapes
    ):
        raise RuntimeError(
            "--specialize-text-mask 必须与 --specialize-audio-shapes 或 --dynamic-shapes 同时使用"
        )
    if bool(args.replay_audio) != bool(args.replay_text):
        raise RuntimeError("--replay-audio 与 --replay-text 必须同时提供")
    if args.dynamic_shapes and not args.replay_audio:
        raise RuntimeError("--dynamic-shapes 必须提供异 shape 的 --replay-audio/--replay-text")
    if args.replay_audio and not args.dynamic_shapes:
        raise RuntimeError("--replay-audio/--replay-text 当前只用于 --dynamic-shapes")

    model_dir = Path(args.model).resolve()
    audio_path = Path(args.audio).resolve()
    text_path = Path(args.text).resolve()
    identity = _model_identity(model_dir)
    audio, duration = _load_audio(audio_path)
    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("参考文本不能为空")

    aligner = Qwen3ForcedAligner.from_pretrained(
        str(model_dir), device_map=args.device, dtype=torch.float16, local_files_only=True
    )
    thinker = aligner.model.thinker
    required_members = ("audio_tower", "model", "lm_head")
    if any(not hasattr(thinker, name) for name in required_members):
        raise RuntimeError("thinker 缺少 audio_tower/model/lm_head，拒绝猜测导出边界")
    if int(thinker.lm_head.out_features) != int(thinker.config.classify_num):
        raise RuntimeError("lm_head.out_features 与 classify_num 不一致")

    words, prompt = aligner.aligner_processor.encode_timestamp(text, args.language)
    inputs = aligner.processor(
        text=[prompt], audio=[audio], return_tensors="pt", padding=True
    )
    inputs = inputs.to(aligner.model.device).to(aligner.model.dtype)
    reference_tensor_inputs = {
        name: value for name, value in inputs.items() if isinstance(value, torch.Tensor)
    }
    tensor_inputs = reference_tensor_inputs

    dynamic_input_contract = None
    reference_dynamic_input_contract = None
    if args.dynamic_shapes:
        reference_dynamic_input_contract = _validate_dynamic_shape_input(
            reference_tensor_inputs,
            "主输入",
            torch,
            allow_short=(
                args.normalize_short_tail_to_full_chunks
                or args.direct_short_single_chunk_bucket
            ),
        )
        dynamic_input_contract = reference_dynamic_input_contract

    shape_replay: dict[str, Any] | None = None
    if args.replay_audio:
        replay_audio_path = Path(args.replay_audio).resolve()
        replay_text_path = Path(args.replay_text).resolve()
        replay_audio, replay_duration = _load_audio(replay_audio_path)
        replay_text = replay_text_path.read_text(encoding="utf-8").strip()
        if not replay_text:
            raise ValueError("异 shape 回放参考文本不能为空")
        replay_words, replay_prompt = aligner.aligner_processor.encode_timestamp(
            replay_text, args.language
        )
        replay_inputs = aligner.processor(
            text=[replay_prompt], audio=[replay_audio], return_tensors="pt", padding=True
        )
        replay_inputs = replay_inputs.to(aligner.model.device).to(aligner.model.dtype)
        replay_tensor_inputs = {
            name: value
            for name, value in replay_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        main_ranks = {name: value.ndim for name, value in tensor_inputs.items()}
        replay_ranks = {name: value.ndim for name, value in replay_tensor_inputs.items()}
        if replay_ranks != main_ranks:
            raise RuntimeError(
                "主输入与异 shape 回放输入的 tensor 字段或 rank 不一致："
                f"main={main_ranks}, replay={replay_ranks}"
            )
        replay_contract = _validate_dynamic_shape_input(
            replay_tensor_inputs,
            "异 shape 回放输入",
            torch,
            allow_short=(
                args.normalize_short_tail_to_full_chunks
                or args.direct_short_single_chunk_bucket
            ),
        )
        if replay_contract["mel_bins"] != dynamic_input_contract["mel_bins"]:
            raise RuntimeError("主输入与异 shape 回放输入的 mel bins 不一致")
        if args.direct_short_single_chunk_bucket:
            main_bucket = _tail_output_bucket(dynamic_input_contract["audio_frames"])
            replay_bucket = _tail_output_bucket(replay_contract["audio_frames"])
            if (
                dynamic_input_contract["audio_full_chunks"] != 0
                or replay_contract["audio_full_chunks"] != 0
            ):
                raise RuntimeError(
                    "原生单 chunk 动态模式要求主输入与回放的 "
                    "audio_full_chunks 均为 0"
                )
            if dynamic_input_contract["audio_frames"] == replay_contract["audio_frames"]:
                raise RuntimeError("原生单 chunk 动态模式要求主输入与回放帧数不同")
            if main_bucket["output_length"] != replay_bucket["output_length"]:
                raise RuntimeError(
                    "原生单 chunk 动态模式要求主输入与回放位于同一 CNN 输出桶："
                    f"main={main_bucket}, replay={replay_bucket}"
                )
        elif args.normalize_short_tail_to_full_chunks:
            main_bucket = _tail_output_bucket(
                dynamic_input_contract["audio_tail_frames"]
            )
            replay_bucket = _tail_output_bucket(replay_contract["audio_tail_frames"])
            if {
                dynamic_input_contract["audio_full_chunks"],
                replay_contract["audio_full_chunks"],
            } != {0, 1}:
                raise RuntimeError(
                    "短音频规范化要求主输入与回放分别覆盖 "
                    "audio_full_chunks=0 和 1"
                )
            if main_bucket["output_length"] != replay_bucket["output_length"]:
                raise RuntimeError(
                    "短音频规范化要求主输入与回放具有相同 CNN 尾块输出长度："
                    f"main={main_bucket}, replay={replay_bucket}"
                )
        elif args.canonicalize_tail_bucket:
            main_bucket = _tail_output_bucket(
                dynamic_input_contract["audio_tail_frames"]
            )
            replay_bucket = _tail_output_bucket(replay_contract["audio_tail_frames"])
            if (
                main_bucket["tail_frames"]
                == replay_bucket["tail_frames"]
            ):
                raise RuntimeError(
                    "T1.2 尾帧桶规范化要求主输入与回放使用不同的非零尾帧余数"
                )
            if main_bucket["output_length"] != replay_bucket["output_length"]:
                raise RuntimeError(
                    "尾帧桶规范化要求主输入与回放具有相同 CNN 尾块输出长度："
                    f"main={main_bucket}, replay={replay_bucket}"
                )
        elif (
            replay_contract["audio_tail_frames"]
            != dynamic_input_contract["audio_tail_frames"]
        ):
            raise RuntimeError(
                "当前动态 T1.1 要求主输入与异 shape 回放具有相同固定尾帧余数："
                f"main={dynamic_input_contract['audio_tail_frames']}, "
                f"replay={replay_contract['audio_tail_frames']}"
            )
        if (
            replay_contract["audio_frames"] == dynamic_input_contract["audio_frames"]
            and replay_contract["text_tokens"] == dynamic_input_contract["text_tokens"]
        ):
            raise RuntimeError("异 shape 回放的音频帧数和文本长度均未变化")
        shape_replay = {
            "audio_path": replay_audio_path,
            "text_path": replay_text_path,
            "duration": replay_duration,
            "unit_count": len(replay_words),
            "reference_tensor_inputs": replay_tensor_inputs,
            "tensor_inputs": replay_tensor_inputs,
            "reference_contract": replay_contract,
            "contract": replay_contract,
        }

    tail_bucket_canonicalization = None
    short_tail_normalization = None
    direct_short_bucket_plan = None
    trailing_output_drop_override = None
    if args.direct_short_single_chunk_bucket:
        main_bucket = _tail_output_bucket(dynamic_input_contract["audio_frames"])
        replay_bucket = _tail_output_bucket(shape_replay["contract"]["audio_frames"])
        output_length = main_bucket["output_length"]
        direct_short_bucket_plan = {
            "output_length": output_length,
            "bucket_start": (output_length - 1) * 8 + 1,
            "bucket_end": min(output_length * 8, _DYNAMIC_AUDIO_WINDOW - 1),
            "main_audio_frames": dynamic_input_contract["audio_frames"],
            "shape_replay_audio_frames": shape_replay["contract"]["audio_frames"],
            "padding": 0,
            "output_drop": 0,
        }
        if replay_bucket["output_length"] != output_length:
            raise RuntimeError("原生单 chunk 主输入与回放的 CNN 输出桶不一致")
    elif args.normalize_short_tail_to_full_chunks:
        tensor_inputs, main_short_plan = _normalize_short_tail_to_full_chunks_inputs(
            reference_tensor_inputs, torch
        )
        replay_candidate_inputs, replay_short_plan = (
            _normalize_short_tail_to_full_chunks_inputs(
                shape_replay["reference_tensor_inputs"], torch
            )
        )
        if (
            main_short_plan["output_length"]
            != replay_short_plan["output_length"]
            or main_short_plan["trailing_output_drop"]
            != replay_short_plan["trailing_output_drop"]
        ):
            raise RuntimeError(
                "主输入与回放短音频规范化计划不一致："
                f"main={main_short_plan}, replay={replay_short_plan}"
            )
        dynamic_input_contract = _validate_dynamic_shape_input(
            tensor_inputs, "主输入短音频候选", torch
        )
        shape_replay["tensor_inputs"] = replay_candidate_inputs
        shape_replay["contract"] = _validate_dynamic_shape_input(
            replay_candidate_inputs, "回放短音频候选", torch
        )
        if (
            shape_replay["contract"]["audio_frames"]
            == dynamic_input_contract["audio_frames"]
            and shape_replay["contract"]["text_tokens"]
            == dynamic_input_contract["text_tokens"]
        ):
            raise RuntimeError(
                "短音频规范化后的主输入与回放 shape 完全相同，"
                "不能作为 strict 动态图异 shape 回放证据"
            )
        trailing_output_drop_override = main_short_plan["trailing_output_drop"]
        short_tail_normalization = {
            "main": main_short_plan,
            "shape_replay": replay_short_plan,
        }
    elif args.canonicalize_tail_bucket:
        tensor_inputs, main_tail_plan = _canonicalize_tail_bucket_inputs(
            reference_tensor_inputs, torch
        )
        replay_candidate_inputs, replay_tail_plan = _canonicalize_tail_bucket_inputs(
            shape_replay["reference_tensor_inputs"], torch
        )
        if (
            main_tail_plan["canonical_tail_frames"]
            != replay_tail_plan["canonical_tail_frames"]
        ):
            raise RuntimeError(
                "主输入与回放规范化后的尾帧不一致："
                f"main={main_tail_plan}, replay={replay_tail_plan}"
            )
        dynamic_input_contract = _validate_dynamic_shape_input(
            tensor_inputs, "主输入尾帧桶候选", torch
        )
        shape_replay["tensor_inputs"] = replay_candidate_inputs
        shape_replay["contract"] = _validate_dynamic_shape_input(
            replay_candidate_inputs, "回放尾帧桶候选", torch
        )
        if (
            shape_replay["contract"]["audio_frames"]
            == dynamic_input_contract["audio_frames"]
            and shape_replay["contract"]["text_tokens"]
            == dynamic_input_contract["text_tokens"]
        ):
            raise RuntimeError(
                "尾帧桶规范化后的主输入与回放 shape 完全相同，"
                "不能作为 strict 动态图异 shape 回放证据"
            )
        tail_bucket_canonicalization = {
            "main": main_tail_plan,
            "shape_replay": replay_tail_plan,
        }

    stage_outputs: dict[str, Any] = {}
    hooks = []
    for name in required_members:
        def capture(_module: Any, _inputs: Any, output: Any, stage: str = name) -> None:
            stage_outputs[stage] = _tensor_meta(output)
        hooks.append(getattr(thinker, name).register_forward_hook(capture))

    try:
        with torch.inference_mode():
            eager_output = thinker(**reference_tensor_inputs)
            eager_logits = _extract_logits(eager_output, torch)
            torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()

    timestamp_mask = (
        reference_tensor_inputs["input_ids"] == aligner.timestamp_token_id
    )
    eager_buckets = eager_logits.argmax(dim=-1)[timestamp_mask].detach().cpu()
    if shape_replay is not None:
        with torch.inference_mode():
            replay_eager_output = thinker(**shape_replay["reference_tensor_inputs"])
            replay_eager_logits = _extract_logits(replay_eager_output, torch)
            torch.cuda.synchronize()
        replay_timestamp_mask = (
            shape_replay["reference_tensor_inputs"]["input_ids"]
            == aligner.timestamp_token_id
        )
        replay_eager_buckets = replay_eager_logits.argmax(dim=-1)[
            replay_timestamp_mask
        ].detach().cpu()
        shape_replay.update(
            {
                "eager_logits": replay_eager_logits,
                "eager_buckets": replay_eager_buckets,
                "timestamp_mask": replay_timestamp_mask,
            }
        )

    specialization = None
    specialized_eager = None
    replay_specialized_eager = None
    text_mask_specialization = None
    text_mask_eager = None
    replay_text_mask_eager = None
    if args.direct_short_single_chunk_bucket:
        specialization = _install_single_chunk_dynamic_audio_rewrite(
            thinker, tensor_inputs, torch
        )
    elif args.dynamic_shapes:
        specialization = _install_dynamic_audio_shape_rewrite(
            thinker,
            tensor_inputs,
            torch,
            trailing_output_drop_override=trailing_output_drop_override,
        )
    elif args.specialize_audio_shapes:
        specialization = _install_fixed_audio_shape_specialization(
            thinker, tensor_inputs, torch
        )

    if specialization is not None:
        with torch.inference_mode():
            specialized_output = thinker(**tensor_inputs)
            specialized_logits = _extract_logits(specialized_output, torch)
            torch.cuda.synchronize()
        specialized_eager = _compare_logits(
            eager_logits, specialized_logits, timestamp_mask, eager_buckets, torch
        )
        if shape_replay is not None:
            with torch.inference_mode():
                replay_specialized_output = thinker(**shape_replay["tensor_inputs"])
                replay_specialized_logits = _extract_logits(
                    replay_specialized_output, torch
                )
                torch.cuda.synchronize()
            replay_specialized_eager = _compare_logits(
                shape_replay["eager_logits"],
                replay_specialized_logits,
                shape_replay["timestamp_mask"],
                shape_replay["eager_buckets"],
                torch,
            )

    if args.specialize_text_mask:
        text_mask_specialization = _install_fixed_text_mask_specialization(
            thinker, tensor_inputs, torch
        )
        with torch.inference_mode():
            text_mask_output = thinker(**tensor_inputs)
            text_mask_logits = _extract_logits(text_mask_output, torch)
            torch.cuda.synchronize()
        text_mask_eager = _compare_logits(
            eager_logits, text_mask_logits, timestamp_mask, eager_buckets, torch
        )
        if shape_replay is not None:
            with torch.inference_mode():
                replay_text_mask_output = thinker(**shape_replay["tensor_inputs"])
                replay_text_mask_logits = _extract_logits(
                    replay_text_mask_output, torch
                )
                torch.cuda.synchronize()
            replay_text_mask_eager = _compare_logits(
                shape_replay["eager_logits"],
                replay_text_mask_logits,
                shape_replay["timestamp_mask"],
                shape_replay["eager_buckets"],
                torch,
            )

    dynamic_shape_spec = None
    dynamic_shape_constraints = None
    if args.dynamic_shapes:
        text_min = 4
        text_max = int(
            getattr(thinker.model.config, "max_position_embeddings", 32768)
        )
        observed_audio = {
            dynamic_input_contract["audio_frames"],
            shape_replay["contract"]["audio_frames"],
        }
        observed_full_chunks = {
            dynamic_input_contract["audio_full_chunks"],
            shape_replay["contract"]["audio_full_chunks"],
        }
        observed_text = {
            dynamic_input_contract["text_tokens"],
            shape_replay["contract"]["text_tokens"],
        }
        if min(observed_text) < text_min or max(observed_text) > text_max:
            raise RuntimeError(
                f"动态文本长度超出 [{text_min}, {text_max}]：{sorted(observed_text)}"
            )
        text_tokens = torch.export.Dim(
            "text_tokens", min=text_min, max=text_max
        )
        if args.direct_short_single_chunk_bucket:
            output_length = direct_short_bucket_plan["output_length"]
            audio_min = direct_short_bucket_plan["bucket_start"]
            audio_max = direct_short_bucket_plan["bucket_end"]
            if observed_full_chunks != {0}:
                raise RuntimeError(
                    "原生单 chunk 动态帧维只允许 audio_full_chunks=0："
                    f"{sorted(observed_full_chunks)}"
                )
            if min(observed_audio) < audio_min or max(observed_audio) > audio_max:
                raise RuntimeError(
                    "原生单 chunk 音频帧数超出输出桶 "
                    f"[{audio_min}, {audio_max}]：{sorted(observed_audio)}"
                )
            audio_frames = torch.export.Dim(
                "short_audio_frames", min=audio_min, max=audio_max
            )
            dynamic_shape_constraints = {
                "batch_size": {"static": 1},
                "mel_bins": {"static": dynamic_input_contract["mel_bins"]},
                "audio_full_chunks": {"static": 0},
                "audio_tail_frames": {
                    "min": audio_min,
                    "max": audio_max,
                    "dynamic": True,
                },
                "audio_frames": {
                    "min": audio_min,
                    "max": audio_max,
                    "output_length": output_length,
                    "tail_frames_dynamic": True,
                },
                "text_tokens": {"min": text_min, "max": text_max},
            }
        else:
            audio_tail_frames = dynamic_input_contract["audio_tail_frames"]
            audio_full_chunks_min = _DYNAMIC_AUDIO_CHUNKS_MIN
            audio_full_chunks_max = (
                _DYNAMIC_AUDIO_CHUNKS_MAX
                if audio_tail_frames == 0
                else _DYNAMIC_AUDIO_CHUNKS_MAX - 1
            )
            audio_min = (
                _DYNAMIC_AUDIO_WINDOW * audio_full_chunks_min + audio_tail_frames
            )
            audio_max = (
                _DYNAMIC_AUDIO_WINDOW * audio_full_chunks_max + audio_tail_frames
            )
            if (
                min(observed_full_chunks) < audio_full_chunks_min
                or max(observed_full_chunks) > audio_full_chunks_max
            ):
                raise RuntimeError(
                    "动态音频完整 chunk 数超出 "
                    f"[{audio_full_chunks_min}, {audio_full_chunks_max}]："
                    f"{sorted(observed_full_chunks)}"
                )
            if min(observed_audio) < audio_min or max(observed_audio) > audio_max:
                raise RuntimeError(
                    f"动态音频帧数超出 [{audio_min}, {audio_max}]："
                    f"{sorted(observed_audio)}"
                )
            audio_full_chunks = torch.export.Dim(
                "audio_full_chunks",
                min=audio_full_chunks_min,
                max=audio_full_chunks_max,
            )
            audio_frames = _DYNAMIC_AUDIO_WINDOW * audio_full_chunks
            if audio_tail_frames:
                audio_frames = audio_frames + audio_tail_frames
            dynamic_shape_constraints = {
                "batch_size": {"static": 1},
                "mel_bins": {"static": dynamic_input_contract["mel_bins"]},
                "audio_full_chunks": {
                    "min": audio_full_chunks_min,
                    "max": audio_full_chunks_max,
                },
                "audio_tail_frames": {"static": audio_tail_frames},
                "audio_frames": {
                    "expression": (
                        f"{_DYNAMIC_AUDIO_WINDOW} * audio_full_chunks"
                        f" + {audio_tail_frames}"
                    ),
                    "min": audio_min,
                    "max": audio_max,
                    "tail_frames_dynamic": False,
                },
                "text_tokens": {"min": text_min, "max": text_max},
            }
        dynamic_shape_spec = {name: None for name in tensor_inputs}
        dynamic_shape_spec.update(
            {
                "input_ids": {1: text_tokens},
                "attention_mask": {1: text_tokens},
                "feature_attention_mask": {1: audio_frames},
                "input_features": {2: audio_frames},
            }
        )

    export_error = None
    main_export_replay = None
    shape_export_replay = None
    export_warnings: list[dict[str, str]] = []
    mode_name = "动态" if args.dynamic_shapes else "固定"
    require_exact_logits = bool(
        args.canonicalize_tail_bucket
        or args.normalize_short_tail_to_full_chunks
        or args.direct_short_single_chunk_bucket
    )
    if specialization is not None and not _comparison_passed(
        specialized_eager, require_exact_logits=require_exact_logits
    ):
        export_error = {
            "type": "AudioShapeSpecializationMismatch",
            "message": f"{mode_name}音频 shape 改写未通过主输入 eager 等价门禁，拒绝导出",
            "traceback": None,
        }
    elif shape_replay is not None and not _comparison_passed(
        replay_specialized_eager, require_exact_logits=require_exact_logits
    ):
        export_error = {
            "type": "AudioShapeReplayMismatch",
            "message": "动态音频 shape 改写未通过异 shape eager 等价门禁，拒绝导出",
            "traceback": None,
        }
    elif args.dynamic_shapes and text_mask_specialization is not None and not (
        text_mask_specialization["uses_native_causal_attention"]
    ):
        export_error = {
            "type": "DynamicTextMaskUnsupported",
            "message": "动态文本长度不能复用固定显式 causal mask，拒绝导出",
            "traceback": None,
        }
    elif text_mask_eager is not None and not _comparison_passed(
        text_mask_eager, require_exact_logits=require_exact_logits
    ):
        export_error = {
            "type": "TextMaskSpecializationMismatch",
            "message": "文本 causal mask 改写未通过主输入 eager 等价门禁，拒绝导出",
            "traceback": None,
        }
    elif shape_replay is not None and args.specialize_text_mask and not _comparison_passed(
        replay_text_mask_eager, require_exact_logits=require_exact_logits
    ):
        export_error = {
            "type": "TextMaskReplayMismatch",
            "message": "文本 causal mask 改写未通过异 shape eager 等价门禁，拒绝导出",
            "traceback": None,
        }
    else:
        observed_warnings: list[Any] = []
        try:
            with warnings.catch_warnings(record=True) as observed_warnings:
                warnings.simplefilter("always")
                export_kwargs: dict[str, Any] = {
                    "args": (),
                    "kwargs": tensor_inputs,
                    "strict": True,
                }
                if dynamic_shape_spec is not None:
                    export_kwargs["dynamic_shapes"] = dynamic_shape_spec
                exported = torch.export.export(thinker, **export_kwargs)
            with torch.inference_mode():
                replay_output = exported.module()(**tensor_inputs)
                replay_logits = _extract_logits(replay_output, torch)
                torch.cuda.synchronize()
            main_export_replay = _compare_logits(
                eager_logits, replay_logits, timestamp_mask, eager_buckets, torch
            )
            if not _comparison_passed(
                main_export_replay, require_exact_logits=require_exact_logits
            ):
                export_error = {
                    "type": "ExportReplayMismatch",
                    "message": "导出图未通过主输入回放等价门禁",
                    "traceback": None,
                }
            elif shape_replay is not None:
                with torch.inference_mode():
                    shape_replay_output = exported.module()(
                        **shape_replay["tensor_inputs"]
                    )
                    shape_replay_logits = _extract_logits(
                        shape_replay_output, torch
                    )
                    torch.cuda.synchronize()
                shape_export_replay = _compare_logits(
                    shape_replay["eager_logits"],
                    shape_replay_logits,
                    shape_replay["timestamp_mask"],
                    shape_replay["eager_buckets"],
                    torch,
                )
                if not _comparison_passed(
                    shape_export_replay, require_exact_logits=require_exact_logits
                ):
                    export_error = {
                        "type": "DynamicShapeReplayMismatch",
                        "message": "同一导出图未通过异 shape 回放等价门禁",
                        "traceback": None,
                    }
        except Exception as error:  # noqa: BLE001 - 报告首个导出阻塞而不是吞掉。
            export_error = {
                "type": type(error).__name__,
                "message": str(error)[:4000],
                "traceback": traceback.format_exc(limit=20)[-12000:],
            }
        finally:
            export_warnings = [
                {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                }
                for warning in observed_warnings
            ]

    source_path = Path(inspect.getsourcefile(type(thinker)) or "")
    passed = export_error is None and _comparison_passed(
        main_export_replay, require_exact_logits=require_exact_logits
    )
    if args.dynamic_shapes:
        passed = passed and _comparison_passed(
            shape_export_replay, require_exact_logits=require_exact_logits
        )
    export_root_warning = any(
        "_export_root" in warning["message"] for warning in export_warnings
    )
    shape_replay_report = None
    if shape_replay is not None:
        shape_replay_report = {
            "input": {
                "audio_filename": shape_replay["audio_path"].name,
                "audio_sha256": _sha256(shape_replay["audio_path"]),
                "duration_seconds": shape_replay["duration"],
                "text_filename": shape_replay["text_path"].name,
                "text_sha256": _sha256(shape_replay["text_path"]),
                "language": args.language,
                "unit_count": shape_replay["unit_count"],
                "tensors": {
                    name: _tensor_meta(value)
                    for name, value in shape_replay["reference_tensor_inputs"].items()
                },
                "candidate_tensors": {
                    name: _tensor_meta(value)
                    for name, value in shape_replay["tensor_inputs"].items()
                },
            },
            "reference_contract": shape_replay["reference_contract"],
            "contract": shape_replay["contract"],
            "eager": {
                "logits_shape": list(shape_replay["eager_logits"].shape),
                "timestamp_bucket_count": int(
                    shape_replay["eager_buckets"].numel()
                ),
            },
            "audio_rewrite_comparison": replay_specialized_eager,
            "text_mask_comparison": replay_text_mask_eager,
        }

    return {
        "schema_version": 1,
        "experiment_id": "PERF-ARCH-002-TARGET-EXPORT-PROBE",
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "qwen_asr": importlib.metadata.version("qwen-asr"),
            "transformers": importlib.metadata.version("transformers"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "model": {
            "path": str(model_dir),
            **identity,
            "thinker_class": f"{type(thinker).__module__}.{type(thinker).__name__}",
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path) if source_path.is_file() else None,
            "classify_num": int(thinker.config.classify_num),
            "lm_head_shape": list(thinker.lm_head.weight.shape),
        },
        "input": {
            "audio_filename": audio_path.name,
            "audio_sha256": _sha256(audio_path),
            "duration_seconds": duration,
            "text_filename": text_path.name,
            "text_sha256": _sha256(text_path),
            "language": args.language,
            "unit_count": len(words),
            "tensors": {
                name: _tensor_meta(value)
                for name, value in reference_tensor_inputs.items()
            },
            "candidate_tensors": {
                name: _tensor_meta(value) for name, value in tensor_inputs.items()
            },
        },
        "parameters": {
            "dtype": "float16",
            "device": args.device,
            "strict": True,
            "dynamic_shapes": args.dynamic_shapes,
            "specialize_audio_shapes": args.specialize_audio_shapes,
            "specialize_text_mask": args.specialize_text_mask,
            "canonicalize_tail_bucket": args.canonicalize_tail_bucket,
            "normalize_short_tail_to_full_chunks": (
                args.normalize_short_tail_to_full_chunks
            ),
            "direct_short_single_chunk_bucket": (
                args.direct_short_single_chunk_bucket
            ),
        },
        "dynamic_shape": {
            "enabled": args.dynamic_shapes,
            "constraints": dynamic_shape_constraints,
            "reference_main_contract": reference_dynamic_input_contract,
            "main_contract": dynamic_input_contract,
            "tail_bucket_canonicalization": tail_bucket_canonicalization,
            "short_tail_normalization": short_tail_normalization,
            "direct_short_bucket_plan": direct_short_bucket_plan,
            "shape_replay": shape_replay_report,
        },
        "eager": {
            "logits_shape": list(eager_logits.shape),
            "timestamp_bucket_count": int(eager_buckets.numel()),
            "stages": stage_outputs,
        },
        "audio_shape_specialization": {
            "plan": specialization,
            "eager_comparison": specialized_eager,
        },
        "text_mask_specialization": {
            "plan": text_mask_specialization,
            "eager_comparison": text_mask_eager,
        },
        "export": {
            "passed": export_error is None,
            "replay": main_export_replay,
            "shape_replay": shape_export_replay,
            "warnings": export_warnings,
            "export_root_side_effect_warning": export_root_warning,
            "error": export_error,
        },
        "summary": {
            "passed": bool(passed),
            "fixed_shape_export": bool(passed) if not args.dynamic_shapes else None,
            "dynamic_shape_export": bool(passed) if args.dynamic_shapes else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PERF-ARCH-002 TensorRT FP16 路线的导出可行性探针"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    static_parser = subparsers.add_parser("static", help="无 GPU 官方源码契约检查")
    static_parser.add_argument(
        "--source-root", default=".tmp-qwen3-asr-official",
        help="qwen-asr 官方源码根目录",
    )
    static_parser.add_argument("--output-json", required=True)

    target_parser = subparsers.add_parser(
        "target", help="目标 GPU 固定或动态 shape torch.export 探针"
    )
    target_parser.add_argument(
        "--model", default="models/qwen3-forced-aligner-0.6b/pt",
        help="ModelScope 固定 revision 的 PyTorch Aligner 目录",
    )
    target_parser.add_argument("--audio", required=True)
    target_parser.add_argument("--text", required=True)
    target_parser.add_argument("--language", default="Chinese")
    target_parser.add_argument("--device", default="cuda:0")
    target_parser.add_argument(
        "--specialize-audio-shapes",
        action="store_true",
        help="仅为当前 batch=1 输入专门化音频 chunk/CNN/cu_seqlens，仍使用 strict export",
    )
    target_parser.add_argument(
        "--specialize-text-mask",
        action="store_true",
        help="绕过 Transformers mask vmap；SDPA 无 padding 路径保留原生 causal attention",
    )
    target_parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="固定 batch=1，动态音频帧数和文本长度，并执行异 shape 回放",
    )
    target_parser.add_argument(
        "--canonicalize-tail-bucket",
        action="store_true",
        help="仅用于动态 shape：把同 CNN 输出桶的非零尾帧补到同一 canonical tail",
    )
    target_parser.add_argument(
        "--normalize-short-tail-to-full-chunks",
        action="store_true",
        help=(
            "仅用于复现已拒绝的 T1.3 候选：把 0/1 个完整 chunk 补成完整块；"
            "full0 原生卷积边界不等价，不得作为可用方案"
        ),
    )
    target_parser.add_argument(
        "--direct-short-single-chunk-bucket",
        action="store_true",
        help=(
            "仅用于动态 shape：full0 单 chunk 保持原始卷积宽度，"
            "在同一 CNN 输出桶内直接动态化 1～99 帧"
        ),
    )
    target_parser.add_argument(
        "--replay-audio",
        help="动态导出后用于同一图异 shape 回放的 16 kHz 单声道音频",
    )
    target_parser.add_argument(
        "--replay-text",
        help="异 shape 回放音频对应的参考文本",
    )
    target_parser.add_argument("--output-json", required=True)

    args = parser.parse_args()
    output_path = Path(args.output_json)
    if output_path.suffix.lower() != ".json":
        parser.error("--output-json 必须以 .json 结尾")
    report = _run_static(args) if args.command == "static" else _run_target(args)
    _write_json(output_path, report)
    passed = bool(report["summary"]["passed"])
    print(
        f"PERF-ARCH-002 {args.command}：{'通过' if passed else '不通过'}，"
        f"报告 {output_path}"
    )
    if not passed and args.command == "target":
        error = report.get("export", {}).get("error") or {}
        message = str(error.get("message") or "未记录错误").splitlines()[0]
        print(f"首个阻塞：{error.get('type', 'UnknownError')}：{message}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())