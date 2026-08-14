"""Qwen3 ForcedAligner 可选后处理：生成字/词级及句级时间戳。"""

import asyncio
import io
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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
    """对齐结果及单请求在共享动态微批中的墙钟耗时与批次观测。"""

    units: list[list[AlignedUnit]]
    skipped: list[AlignmentSkip]
    queue_wait_ms: float = 0.0
    inference_ms: float = 0.0
    batch_count: int = 0
    batch_size_max: int = 0
    batch_size_mean: float = 0.0
    queue_depth_max: int = 0
    batch_audio_decode_ms: float = 0.0
    batch_model_call_ms: float = 0.0
    batch_result_build_ms: float = 0.0
    predecode_wait_ms: float = 0.0
    predecode_audio_decode_ms: float = 0.0
    predecode_count: int = 0


@dataclass(frozen=True)
class _AlignmentWorkResult:
    """一个物理分片在共享模型批次中的结果和时间边界。"""

    units: list[AlignedUnit]
    batch_id: int
    batch_size: int
    batch_started_at: float
    batch_finished_at: float
    audio_decode_ms: float
    model_call_ms: float
    result_build_ms: float


@dataclass(frozen=True)
class _PreparedAudio:
    """预解码 PCM、实际 CPU 解码耗时和占用的全局字节预算。"""

    audio: tuple[Any, int]
    decode_ms: float
    reserved_bytes: int


class _AsyncByteBudget:
    """事件循环内的加权字节信号量，避免已解码 PCM 无界驻留。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.used = 0
        self._condition = asyncio.Condition()

    async def acquire(self, amount: int) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self.used + amount <= self.capacity
            )
            self.used += amount

    async def release(self, amount: int) -> None:
        async with self._condition:
            self.used -= amount
            if self.used < 0:
                raise RuntimeError("ForcedAligner 预解码字节预算释放失衡")
            self._condition.notify_all()


class _AlignmentPreparation:
    """一次请求的预解码任务集合；关闭时取消任务并归还全部 PCM 预算。"""

    def __init__(
        self,
        tasks: list[asyncio.Task[_PreparedAudio | None]],
        budget: _AsyncByteBudget,
    ) -> None:
        self._tasks = tasks
        self._budget = budget
        self._closed = False

    async def resolve(self, indices: list[int]) -> dict[int, _PreparedAudio]:
        outcomes = await asyncio.gather(*(self._tasks[index] for index in indices))
        return {
            index: outcome
            for index, outcome in zip(indices, outcomes)
            if outcome is not None
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            if not task.done():
                task.cancel()
        outcomes = await asyncio.gather(*self._tasks, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, _PreparedAudio):
                await self._budget.release(outcome.reserved_bytes)


@dataclass
class _PredecodeWorkItem:
    """进入有界预解码队列的单个物理分片。"""

    audio: bytes
    future: asyncio.Future[tuple[Any, int, float]]


@dataclass
class _AlignmentWorkItem:
    """进入有界动态微批队列的一个物理分片。"""

    chunk: Any
    text: str
    language: str
    future: asyncio.Future[_AlignmentWorkResult]
    decoded_audio: tuple[Any, int] | None = None


_STOP_WORKER = object()


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
        self._attention_backend_actual = "disabled"
        self._queue: asyncio.Queue[Any] | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._decode_executor: ThreadPoolExecutor | None = None
        self._predecode_queue: asyncio.Queue[Any] | None = None
        self._predecode_workers: list[asyncio.Task[None]] = []
        self._predecode_budget: _AsyncByteBudget | None = None
        self._submit_lock: asyncio.Lock | None = None
        self._accepting = False
        self._batch_id = 0

    @property
    def enabled(self) -> bool:
        return settings.timestamp_enabled

    @property
    def attention_backend_actual(self) -> str:
        """返回 Transformers 最终采用的 attention 后端。"""
        return self._attention_backend_actual

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
        model_kwargs: dict[str, Any] = {
            "device_map": settings.aligner_device,
            "dtype": dtype,
            "local_files_only": True,
        }
        if settings.aligner_attention_backend != "auto":
            model_kwargs["attn_implementation"] = settings.aligner_attention_backend
        self._model = Qwen3ForcedAligner.from_pretrained(
            str(model_dir),
            **model_kwargs,
        )
        model = getattr(self._model, "model", None)
        config = getattr(model, "config", None)
        actual = getattr(config, "_attn_implementation", None)
        self._attention_backend_actual = str(actual or "unknown")

    async def start(self) -> None:
        """在当前事件循环创建有界队列和动态微批 worker。"""
        if not self.enabled:
            return
        if self._model is None:
            raise RuntimeError("ForcedAligner 尚未加载")
        if self._workers:
            return
        self._queue = asyncio.Queue(maxsize=settings.aligner_queue_size)
        self._submit_lock = asyncio.Lock()
        self._accepting = True
        self._batch_id = 0
        if settings.aligner_decode_workers > 1:
            self._decode_executor = ThreadPoolExecutor(
                max_workers=settings.aligner_decode_workers,
                thread_name_prefix="forced-aligner-decode",
            )
        if settings.aligner_predecode_enabled:
            self._predecode_queue = asyncio.Queue(
                maxsize=settings.aligner_queue_size
            )
            self._predecode_budget = _AsyncByteBudget(
                settings.aligner_predecode_max_mb * 1024**2
            )
            self._predecode_workers = [
                asyncio.create_task(
                    self._predecode_worker(worker_index),
                    name=f"forced-aligner-predecode-{worker_index}",
                )
                for worker_index in range(settings.aligner_decode_workers)
            ]
        self._workers = [
            asyncio.create_task(
                self._batch_worker(worker_index),
                name=f"forced-aligner-worker-{worker_index}",
            )
            for worker_index in range(settings.aligner_concurrency)
        ]

    async def close(self) -> None:
        """停止接收新任务，排空预解码和动态微批队列并回收线程池。"""
        lock = self._submit_lock
        if lock is not None:
            async with lock:
                self._accepting = False
        else:
            self._accepting = False

        predecode_queue = self._predecode_queue
        predecode_workers = list(self._predecode_workers)
        if predecode_queue is not None:
            for worker in predecode_workers:
                if not worker.done():
                    await predecode_queue.put(_STOP_WORKER)
        predecode_outcomes = await asyncio.gather(
            *predecode_workers, return_exceptions=True
        )
        self._predecode_workers.clear()
        self._predecode_queue = None
        self._predecode_budget = None

        queue = self._queue
        workers = list(self._workers)
        if queue is not None:
            for worker in workers:
                if not worker.done():
                    await queue.put(_STOP_WORKER)
        outcomes = await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._queue = None
        self._submit_lock = None
        decode_executor = self._decode_executor
        self._decode_executor = None
        if decode_executor is not None:
            decode_executor.shutdown(wait=True, cancel_futures=False)
        failures = [
            outcome for outcome in [*predecode_outcomes, *outcomes]
            if isinstance(outcome, BaseException)
            and not isinstance(outcome, asyncio.CancelledError)
        ]
        if failures:
            raise RuntimeError("ForcedAligner worker 异常退出") from failures[0]

    async def _predecode_one(self, chunk: Any) -> _PreparedAudio | None:
        queue = self._predecode_queue
        budget = self._predecode_budget
        estimated_bytes = int(getattr(chunk, "pcm_bytes_estimate", 0))
        if (
            queue is None
            or budget is None
            or estimated_bytes <= 0
            or estimated_bytes > budget.capacity
        ):
            return None
        await budget.acquire(estimated_bytes)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[Any, int, float]] = loop.create_future()
        try:
            if not self._accepting:
                raise RuntimeError("ForcedAligner 预解码调度器正在关闭")
            await queue.put(_PredecodeWorkItem(chunk.audio, future))
            samples, samplerate, decode_ms = await future
            return _PreparedAudio(
                audio=(samples, samplerate),
                decode_ms=decode_ms,
                reserved_bytes=estimated_bytes,
            )
        except BaseException:
            if not future.done():
                future.cancel()
            await budget.release(estimated_bytes)
            raise

    async def _predecode_worker(self, worker_index: int) -> None:
        del worker_index
        queue = self._predecode_queue
        if queue is None:
            raise RuntimeError("ForcedAligner 预解码队列未初始化")
        while True:
            entry = await queue.get()
            if entry is _STOP_WORKER:
                queue.task_done()
                break
            try:
                if entry.future.cancelled():
                    continue
                started = perf_counter()
                samples, samplerate = await asyncio.to_thread(
                    _decode_audio, entry.audio
                )
                decode_ms = (perf_counter() - started) * 1000
                if not entry.future.done():
                    entry.future.set_result((samples, samplerate, decode_ms))
            except Exception as error:  # noqa: BLE001 - 失败必须回传所属请求。
                if not entry.future.done():
                    entry.future.set_exception(self._work_failure(error))
            finally:
                queue.task_done()

    @asynccontextmanager
    async def prepare_chunks(self, chunks: list[Any]):
        """在 ASR 执行期间有界预解码；上下文退出时释放全部 PCM。"""
        queue = self._predecode_queue
        budget = self._predecode_budget
        if not settings.aligner_predecode_enabled or queue is None or budget is None:
            yield None
            return
        preparation = _AlignmentPreparation(
            [
                asyncio.create_task(
                    self._predecode_one(chunk),
                    name=f"forced-aligner-prepare-{index}",
                )
                for index, chunk in enumerate(chunks)
            ],
            budget,
        )
        try:
            yield preparation
        finally:
            await preparation.close()

    def _align_items_sync(
        self,
        items: list[_AlignmentWorkItem],
    ) -> tuple[list[list[AlignedUnit]], float, float, float]:
        """执行一批对齐，并分离批内音频解码、官方调用和结果构建耗时。"""
        if self._model is None:
            raise RuntimeError("ForcedAligner 尚未加载")
        started = perf_counter()
        missing_indices = [
            index for index, item in enumerate(items)
            if item.decoded_audio is None
        ]
        missing_inputs = [items[index].chunk.audio for index in missing_indices]
        if self._decode_executor is None:
            decoded = [_decode_audio(audio) for audio in missing_inputs]
        else:
            # 持久化固定大小线程池只并行独立内存 WAV 解码；map 保持批次输入顺序。
            decoded = list(self._decode_executor.map(_decode_audio, missing_inputs))
        decoded_by_index = dict(zip(missing_indices, decoded))
        audios = [
            item.decoded_audio
            if item.decoded_audio is not None
            else decoded_by_index[index]
            for index, item in enumerate(items)
        ]
        audio_decode_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        aligned = self._model.align(
            audio=audios,
            text=[item.text for item in items],
            language=[item.language for item in items],
        )
        model_call_ms = (perf_counter() - started) * 1000
        if len(aligned) != len(items):
            raise RuntimeError("ForcedAligner 返回数量与动态微批输入不一致")

        started = perf_counter()
        batch_units: list[list[AlignedUnit]] = []
        for item, result in zip(items, aligned):
            local = [
                AlignedUnit(
                    text=str(unit.text),
                    start=float(unit.start_time),
                    end=float(unit.end_time),
                    language=item.language,
                )
                for unit in result
            ]
            _validate_units(local, item.chunk.end - item.chunk.start)
            batch_units.append([
                AlignedUnit(
                    unit.text,
                    round(unit.start + item.chunk.start, 3),
                    round(unit.end + item.chunk.start, 3),
                    unit.language,
                )
                for unit in local
            ])
        result_build_ms = (perf_counter() - started) * 1000
        return batch_units, audio_decode_ms, model_call_ms, result_build_ms

    async def _collect_batch(
        self,
        queue: asyncio.Queue[Any],
        first: _AlignmentWorkItem,
    ) -> tuple[list[_AlignmentWorkItem], bool]:
        """从首条任务起按固定截止时间收集；队列已有任务时立即取满。"""
        batch = [first]
        stop_after_batch = False
        deadline = (
            asyncio.get_running_loop().time()
            + settings.aligner_batch_wait_ms / 1000
        )
        while len(batch) < settings.aligner_batch_size:
            try:
                entry = queue.get_nowait()
            except asyncio.QueueEmpty:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
            if entry is _STOP_WORKER:
                queue.task_done()
                stop_after_batch = True
                break
            batch.append(entry)
        return batch, stop_after_batch

    @staticmethod
    def _work_failure(error: Exception) -> Exception:
        if isinstance(error, ValueError):
            return ValueError(str(error))
        if isinstance(error, RuntimeError):
            return RuntimeError(str(error))
        return RuntimeError(
            f"ForcedAligner 动态微批执行失败: {type(error).__name__}: {error}"
        )

    async def _batch_worker(self, worker_index: int) -> None:
        del worker_index  # worker 名称已保留索引，执行逻辑相同。
        queue = self._queue
        if queue is None:
            raise RuntimeError("ForcedAligner 动态微批队列未初始化")
        while True:
            entry = await queue.get()
            if entry is _STOP_WORKER:
                queue.task_done()
                break
            batch, stop_after_batch = await self._collect_batch(queue, entry)
            active = [item for item in batch if not item.future.cancelled()]
            try:
                if active:
                    self._batch_id += 1
                    batch_id = self._batch_id
                    batch_started_at = perf_counter()
                    try:
                        (
                            units,
                            audio_decode_ms,
                            model_call_ms,
                            result_build_ms,
                        ) = await asyncio.to_thread(self._align_items_sync, active)
                    except Exception as error:  # noqa: BLE001 - 必须广播整批失败。
                        for item in active:
                            if not item.future.done():
                                item.future.set_exception(self._work_failure(error))
                    else:
                        batch_finished_at = perf_counter()
                        batch_size = len(active)
                        for item, item_units in zip(active, units):
                            if not item.future.done():
                                item.future.set_result(_AlignmentWorkResult(
                                    units=item_units,
                                    batch_id=batch_id,
                                    batch_size=batch_size,
                                    batch_started_at=batch_started_at,
                                    batch_finished_at=batch_finished_at,
                                    audio_decode_ms=audio_decode_ms,
                                    model_call_ms=model_call_ms,
                                    result_build_ms=result_build_ms,
                                ))
            finally:
                for _ in batch:
                    queue.task_done()
            if stop_after_batch:
                break

    async def align_chunks(
        self,
        chunks: list[Any],
        texts: list[str],
        languages: list[str],
        preparation: _AlignmentPreparation | None = None,
    ) -> AlignmentBatchResult:
        """将受支持物理分片送入跨请求有界动态微批，并按原索引组装结果。"""
        empty = [[] for _ in chunks]
        if not self.enabled:
            return AlignmentBatchResult(empty, [])
        if len(chunks) != len(texts) or len(chunks) != len(languages):
            raise ValueError("ForcedAligner 分片、文本与语种数量不一致")
        queue = self._queue
        lock = self._submit_lock
        if queue is None or lock is None or not self._accepting:
            raise RuntimeError("ForcedAligner 动态微批调度器未启动或正在关闭")

        normalized = [""] * len(chunks)
        skipped: list[AlignmentSkip] = []
        active_indices: list[int] = []
        for index, (text, language) in enumerate(zip(texts, languages)):
            if not text.strip():
                continue
            try:
                normalized[index] = normalize_language(language)
                active_indices.append(index)
            except ValueError:
                skipped.append(AlignmentSkip(index, str(language).strip()))
        if not active_indices:
            return AlignmentBatchResult(empty, skipped)

        queue_started_at = perf_counter()
        predecode_wait_started = perf_counter()
        prepared: dict[int, _PreparedAudio] = {}
        if preparation is not None:
            prepared = await preparation.resolve(active_indices)
        predecode_wait_ms = (perf_counter() - predecode_wait_started) * 1000

        loop = asyncio.get_running_loop()
        items = [
            _AlignmentWorkItem(
                chunk=chunks[index],
                text=texts[index],
                language=normalized[index],
                decoded_audio=(
                    prepared[index].audio if index in prepared else None
                ),
                future=loop.create_future(),
            )
            for index in active_indices
        ]
        queue_depth_max = 0
        try:
            async with lock:
                if not self._accepting:
                    raise RuntimeError("ForcedAligner 动态微批调度器正在关闭")
                for item in items:
                    await queue.put(item)
                    queue_depth_max = max(queue_depth_max, queue.qsize())
            outcomes = await asyncio.gather(
                *(item.future for item in items),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            for item in items:
                if not item.future.done():
                    item.future.cancel()
            raise
        except Exception:
            for item in items:
                if not item.future.done():
                    item.future.cancel()
            raise

        work_results: list[_AlignmentWorkResult] = []
        for outcome in outcomes:
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, Exception):
                raise outcome
            work_results.append(outcome)
        units = [[] for _ in chunks]
        for index, result in zip(active_indices, work_results):
            units[index] = result.units

        first_started_at = min(result.batch_started_at for result in work_results)
        last_finished_at = max(result.batch_finished_at for result in work_results)
        batch_results: dict[int, _AlignmentWorkResult] = {}
        for result in work_results:
            batch_results.setdefault(result.batch_id, result)
        batches = {
            batch_id: result.batch_size
            for batch_id, result in batch_results.items()
        }
        return AlignmentBatchResult(
            units=units,
            skipped=skipped,
            queue_wait_ms=round(
                (first_started_at - queue_started_at) * 1000, 2
            ),
            inference_ms=round(
                (last_finished_at - first_started_at) * 1000, 2
            ),
            batch_count=len(batches),
            batch_size_max=max(batches.values()),
            batch_size_mean=round(sum(batches.values()) / len(batches), 2),
            queue_depth_max=queue_depth_max,
            batch_audio_decode_ms=round(sum(
                result.audio_decode_ms for result in batch_results.values()
            ), 2),
            batch_model_call_ms=round(sum(
                result.model_call_ms for result in batch_results.values()
            ), 2),
            batch_result_build_ms=round(sum(
                result.result_build_ms for result in batch_results.values()
            ), 2),
            predecode_wait_ms=round(predecode_wait_ms, 2),
            predecode_audio_decode_ms=round(sum(
                result.decode_ms for result in prepared.values()
            ), 2),
            predecode_count=len(prepared),
        )


forced_aligner = ForcedAlignerEngine()