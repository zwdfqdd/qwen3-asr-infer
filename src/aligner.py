"""Qwen3 ForcedAligner 可选后处理：生成字/词级及句级时间戳。"""

import asyncio
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import soundfile as sf

try:
    from .config import settings
except ImportError:  # 兼容 python src/gateway.py 直接执行。
    from config import settings

_LANGUAGE_ALIASES = {
    "zh": "Chinese", "chinese": "Chinese",
    "en": "English", "english": "English",
    "yue": "Cantonese", "cantonese": "Cantonese",
    "fr": "French", "french": "French",
    "de": "German", "german": "German",
    "it": "Italian", "italian": "Italian",
    "ja": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "korean": "Korean",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "ru": "Russian", "russian": "Russian",
    "es": "Spanish", "spanish": "Spanish",
}
_SUPPORTED_LANGUAGES = set(_LANGUAGE_ALIASES.values())
_SENTENCE_PATTERN = re.compile(r".*?(?:[。！？!?；;]+(?:[”’\"']+)?|$)", re.S)


@dataclass(frozen=True)
class AlignedUnit:
    """一个真实对齐单元；起止时间为相对整条原音频的秒，语种来自对应 ASR 分片。"""

    text: str
    start: float
    end: float
    language: str = ""


@dataclass(frozen=True)
class AlignmentSkip:
    """未执行对齐的物理分片索引及模型语种；用于生成显式部分对齐告警。"""

    chunk_index: int
    language: str


@dataclass(frozen=True)
class AlignmentBatchResult:
    """批量对齐结果，并区分信号量排队与模型推理墙钟耗时。"""

    units: list[list[AlignedUnit]]
    skipped: list[AlignmentSkip]
    queue_wait_ms: float = 0.0
    inference_ms: float = 0.0


def normalize_language(value: Any) -> str:
    """把 ISO/英文语种别名规范化为 Aligner 名称；空值或不支持语种抛错。"""
    if value is None or not str(value).strip():
        raise ValueError("ForcedAligner 语种为空，不能可靠对齐")
    raw = str(value).strip().lower()
    language = _LANGUAGE_ALIASES.get(raw, str(value).strip().capitalize())
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"ForcedAligner 不支持语种 {language}；支持: "
            + ", ".join(sorted(_SUPPORTED_LANGUAGES))
        )
    return language


def _comparable(text: str) -> str:
    return "".join(
        char.casefold() for char in unicodedata.normalize("NFKC", text)
        if char == "'" or unicodedata.category(char)[:1] in {"L", "N"}
    )


def _decode_audio(audio: bytes) -> tuple[Any, int]:
    with sf.SoundFile(io.BytesIO(audio)) as source:
        samples = source.read(dtype="float32", always_2d=False)
        return samples, int(source.samplerate)


def _validate_units(units: list[AlignedUnit], duration: float) -> None:
    previous_end = 0.0
    for index, unit in enumerate(units):
        if not unit.text or unit.start < previous_end or unit.end < unit.start:
            raise ValueError(f"ForcedAligner 第 {index} 个结果无效")
        if unit.start < -0.001 or unit.end > duration + 0.05:
            raise ValueError(f"ForcedAligner 第 {index} 个结果超出音频边界")
        previous_end = unit.end


def _sentence_ranges(text: str) -> list[tuple[str, int, int]]:
    ranges = []
    cursor = 0
    for match in _SENTENCE_PATTERN.finditer(text):
        sentence = match.group(0)
        if not sentence:
            continue
        comparable = _comparable(sentence)
        if not comparable:
            continue
        start = cursor
        cursor += len(comparable)
        ranges.append((sentence.strip(), start, cursor))
    return ranges


def build_sentences(text: str, units: list[AlignedUnit]) -> list[dict[str, Any]]:
    """按主模型标点把真实对齐单元聚合为句子。

    ``text`` 必须与全部 ``units`` 归一化后严格一致；返回时间仍是整条原音频的秒级坐标。
    无法可靠映射时抛错，绝不按字符长度估算边界。
    """
    if not text.strip() or not units:
        return []
    full_comparable = _comparable(text)
    units_comparable = "".join(_comparable(unit.text) for unit in units)
    if units_comparable != full_comparable:
        raise ValueError("ForcedAligner 单元与 ASR 文本不一致，无法可靠生成句级时间戳")
    unit_ranges = []
    cursor = 0
    for unit in units:
        value = _comparable(unit.text)
        if not value:
            continue
        start = cursor
        cursor += len(value)
        unit_ranges.append((unit, start, cursor))
    sentences = []
    for sentence, start, end in _sentence_ranges(text):
        matched = [unit for unit, left, right in unit_ranges if right > start and left < end]
        if not matched:
            continue
        languages = []
        for unit in matched:
            if unit.language and unit.language not in languages:
                languages.append(unit.language)
        sentences.append({
            "idx": len(sentences),
            "slid": ",".join(languages),
            "text": sentence,
            "speaker": "",
            "timestamp": [round(matched[0].start, 3), round(matched[-1].end, 3)],
            "words": [
                {"text": unit.text, "timestamp": [round(unit.start, 3), round(unit.end, 3)]}
                for unit in matched
            ] if settings.enable_word_timestamp else [],
        })
    if not sentences:
        raise ValueError("无法按标点将 ForcedAligner 结果映射到句子")
    return sentences


class ForcedAlignerEngine:
    def __init__(self) -> None:
        self._model = None
        self._slots: asyncio.Semaphore | None = None

    @property
    def enabled(self) -> bool:
        return settings.timestamp_enabled

    def load(self) -> None:
        if not self.enabled or self._model is not None:
            return
        model_dir = Path(settings.aligner_model_dir).resolve()
        if not (model_dir / "config.json").is_file():
            raise RuntimeError(f"ForcedAligner 模型目录不完整: {model_dir}")
        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as error:
            raise RuntimeError(
                "时间戳功能需要 qwen-asr==0.0.6；请安装 requirements-aligner.txt"
            ) from error
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[settings.aligner_dtype]
        self._model = Qwen3ForcedAligner.from_pretrained(
            str(model_dir),
            device_map=settings.aligner_device,
            dtype=dtype,
            local_files_only=True,
        )

    def _align_batch_sync(
        self,
        chunks: list[Any],
        texts: list[str],
        languages: list[str],
    ) -> list[list[AlignedUnit]]:
        if self._model is None:
            raise RuntimeError("ForcedAligner 尚未加载")
        results: list[list[AlignedUnit]] = [[] for _ in chunks]
        active = [
            index for index, (text, language) in enumerate(zip(texts, languages))
            if text.strip() and language
        ]
        for start in range(0, len(active), settings.aligner_batch_size):
            indices = active[start:start + settings.aligner_batch_size]
            audios = [_decode_audio(chunks[index].audio) for index in indices]
            aligned = self._model.align(
                audio=audios,
                text=[texts[index] for index in indices],
                language=[languages[index] for index in indices],
            )
            if len(aligned) != len(indices):
                raise RuntimeError("ForcedAligner 返回数量与输入不一致")
            for index, result in zip(indices, aligned):
                local = [
                    AlignedUnit(
                        text=str(item.text),
                        start=float(item.start_time),
                        end=float(item.end_time),
                        language=languages[index],
                    )
                    for item in result
                ]
                _validate_units(local, chunks[index].end - chunks[index].start)
                results[index] = [
                    AlignedUnit(
                        unit.text,
                        round(unit.start + chunks[index].start, 3),
                        round(unit.end + chunks[index].start, 3),
                        unit.language,
                    )
                    for unit in local
                ]
        return results

    async def align_chunks(
        self,
        chunks: list[Any],
        texts: list[str],
        languages: list[str],
    ) -> AlignmentBatchResult:
        """逐物理分片执行对齐。

        ``chunks``、``texts``、``languages`` 必须等长；空文本不对齐，不支持的语种写入
        ``skipped``，受支持分片仍返回真实结果。模型执行受 ``ALIGNER_CONCURRENCY`` 限制。
        """
        empty = [[] for _ in chunks]
        if not self.enabled:
            return AlignmentBatchResult(empty, [], 0.0, 0.0)
        if len(chunks) != len(texts) or len(chunks) != len(languages):
            raise ValueError("ForcedAligner 分片、文本与语种数量不一致")
        normalized = [""] * len(chunks)
        skipped: list[AlignmentSkip] = []
        for index, (text, language) in enumerate(zip(texts, languages)):
            if not text.strip():
                continue
            try:
                normalized[index] = normalize_language(language)
            except ValueError:
                skipped.append(AlignmentSkip(index, str(language).strip()))
        if self._slots is None:
            self._slots = asyncio.Semaphore(settings.aligner_concurrency)
        queue_started = perf_counter()
        async with self._slots:
            queue_wait_ms = (perf_counter() - queue_started) * 1000
            inference_started = perf_counter()
            units = await asyncio.to_thread(
                self._align_batch_sync, chunks, texts, normalized
            )
            inference_ms = (perf_counter() - inference_started) * 1000
        return AlignmentBatchResult(
            units, skipped, round(queue_wait_ms, 2), round(inference_ms, 2)
        )


forced_aligner = ForcedAlignerEngine()