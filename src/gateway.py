"""Qwen3-ASR CPU 网关：兼容 /chinese_asr，并代理原生转写接口。"""

import asyncio
import base64
import binascii
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import soundfile as sf
from aiohttp import web

try:
    from .aligner import build_sentences, forced_aligner
    from .config import settings
    from .logger import (
        generate_request_id, log_event, logger, request_id_var,
        setup_logger, shutdown_logging,
    )
except ImportError:  # 兼容 python src/gateway.py 直接执行。
    from aligner import build_sentences, forced_aligner
    from config import settings
    from logger import (
        generate_request_id, log_event, logger, request_id_var,
        setup_logger, shutdown_logging,
    )


class ErrorCode:
    INPUT_PARAM_FAILED = (1000, "INPUT_PARAM_FAILED", 400)
    DECODE_FAILED = (1001, "DECODE_FAILED", 400)
    VAD_SEGMENT_ERROR = (1002, "VAD_SEGMENT_ERROR", 500)
    AUDIO_SEGMENT_ERROR = (1003, "AUDIO_SEGMENT_ERROR", 500)
    ASR_INFER_FAILED = (1004, "ASR_INFER_FAILED", 500)
    AUDIO_TOO_LONG = (1005, "AUDIO_TOO_LONG", 400)
    MODEL_LOAD_FAILED = (1006, "MODEL_LOAD_FAILED", 500)
    SERVICE_BUSY = (1007, "SERVICE_BUSY", 503)
    HOTWORD_VERSION_CONFLICT = (1008, "HOTWORD_VERSION_CONFLICT", 409)
    ALIGNER_INFER_FAILED = (1009, "ALIGNER_INFER_FAILED", 500)


class APIError(Exception):
    """可转换为 HTTP 响应的业务异常。

    ``definition`` 为 ``(业务码, 错误标识, 默认 HTTP 状态码)``；``message`` 为中文
    客户端提示；``native_status`` 仅覆盖 multipart 原生接口的 HTTP 状态码。
    """

    def __init__(
        self,
        definition: tuple[int, str, int],
        message: str,
        native_status: int | None = None,
    ):
        """保存业务错误定义；``native_status`` 只覆盖非兼容接口的 HTTP 状态码。"""
        self.code, self.error, self.status = definition
        self.native_status = native_status or self.status
        self.message = message
        super().__init__(message)


@dataclass
class AudioChunk:
    """一个物理音频分片。

    ``audio`` 是待提交后端的完整文件字节；``filename``/``content_type`` 用于 HTTP 上传；
    ``start``/``end`` 是该分片相对整条原音频的秒级边界。
    """

    audio: bytes
    filename: str
    content_type: str
    start: float
    end: float
    pcm_bytes_estimate: int = 0


@dataclass(frozen=True)
class ASRChunkResult:
    """单个物理分片的 ASR 结果；``language`` 为空表示模型未给出可靠语种标签。"""

    text: str
    language: str


_ASR_LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}
_ASR_LANGUAGE_NAMES = {name.casefold(): (name, code) for code, name in _ASR_LANGUAGES.items()}
_ASR_DIALECTS = (
    "Anhui", "Dongbei", "Fujian", "Gansu", "Guizhou", "Hebei", "Henan",
    "Hubei", "Hunan", "Jiangxi", "Ningxia", "Shandong", "Shaanxi",
    "Shanxi", "Sichuan", "Tianjin", "Yunnan", "Zhejiang",
    "Cantonese (Hong Kong accent)", "Cantonese (Guangdong accent)",
    "Wu language", "Minnan language",
)
_MODEL_LANGUAGE_LABELS = {
    **{name.casefold(): name for name in _ASR_LANGUAGES.values()},
    **{name.casefold(): name for name in _ASR_DIALECTS},
}
_ASR_TEXT_TAG = "<asr_text>"
_LOGGED_METHODS = {"GET", "POST"}
_LOGGED_PATHS = {
    "/chinese_asr", "/v1/audio/transcriptions", "/health", "/metrics", "/v1/models",
}
_LOGGED_MEDIA_TYPES = {
    "application/json", "application/octet-stream", "audio/flac", "audio/mpeg",
    "audio/ogg", "audio/wav", "audio/x-wav", "multipart/form-data",
}
_STAGE_NAMES = (
    "request-parse", "base64-decode", "audio-split",
    "chunk-local-queue", "chunk-local-queue-max",
    "chunk-global-queue", "chunk-global-queue-max",
    "asr-backend", "asr", "aligner-queue", "aligner",
    "postprocess", "serialize", "total",
)


def _new_stage_timings() -> dict[str, float]:
    """建立顺序固定的阶段耗时字典，未经过的可选阶段保持为零。"""
    return {name: 0.0 for name in _STAGE_NAMES}


def _set_stage(request: web.Request, name: str, elapsed_ms: float) -> None:
    request["stage_timings_ms"][name] = round(elapsed_ms, 2)


def _finish_stage(request: web.Request, name: str, started: float) -> None:
    _set_stage(request, name, (perf_counter() - started) * 1000)


def _server_timing_header(timings: dict[str, float]) -> str:
    return ", ".join(f"{name};dur={timings[name]:.2f}" for name in _STAGE_NAMES)


def _json_response(
    request: web.Request,
    data: Any,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> web.Response:
    """统一构造 JSON 响应，并把同步 JSON 编码计入固定 serialize 阶段。"""
    started = perf_counter()
    response = web.json_response(
        data, status=status, headers=headers, dumps=_json_dumps
    )
    timings = request["stage_timings_ms"]
    timings["serialize"] = round(
        timings["serialize"] + (perf_counter() - started) * 1000, 2
    )
    return response


def _logged_path(value: str) -> str:
    """日志只保留固定路由，未知路径统一归类，避免路径携带秘密或放大日志。"""
    return value if value in _LOGGED_PATHS else "<unmatched>"


def _logged_media_type(value: str | None) -> str:
    """媒体类型只记录允许值，不记录未受信任参数或扩展文本。"""
    normalized = (value or "").split(";", 1)[0].strip().lower()
    return normalized if normalized in _LOGGED_MEDIA_TYPES else "other"


def _normalize_requested_language(value: Any) -> tuple[str, str] | None:
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip()
    if raw.casefold() in _ASR_LANGUAGE_NAMES:
        return _ASR_LANGUAGE_NAMES[raw.casefold()]
    code = raw.lower()
    if code in _ASR_LANGUAGES:
        return _ASR_LANGUAGES[code], code
    raise APIError(
        ErrorCode.INPUT_PARAM_FAILED,
        "language 仅支持 Qwen3-ASR 的 30 种可显式指定语言名称或 ISO 代码",
    )


def _canonical_model_language(value: str) -> str:
    label = value.strip()
    return _MODEL_LANGUAGE_LABELS.get(label.casefold(), label)


def _log_language_label(value: str) -> str:
    """日志仅保留官方语种标签；未知模型元数据统一记为 Other。"""
    label = _canonical_model_language(value)
    if not label:
        return ""
    if label.casefold() in _MODEL_LANGUAGE_LABELS:
        return label
    return "Other"


def _parse_raw_asr_output(raw: str) -> ASRChunkResult:
    value = raw.strip()
    if not value:
        return ASRChunkResult("", "")
    if _ASR_TEXT_TAG not in value:
        # 后端未返回结构化语言标签时保留文本，但不猜测语种。
        return ASRChunkResult(value, "")
    metadata, text = value.split(_ASR_TEXT_TAG, 1)
    language = ""
    for line in metadata.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            language = _canonical_model_language(line[len("language "):])
            break
    text = text.strip()
    if language.casefold() == "none" and not text:
        return ASRChunkResult("", "")
    return ASRChunkResult(text, "" if language.casefold() == "none" else language)


def _logged_chinese_asr_input(payload: Any) -> Any:
    """复制业务输入供日志使用，并将音频 Base64 严格限制为前 64 个 ASCII 字节。"""
    if not isinstance(payload, dict):
        return payload
    logged = dict(payload)
    encoded = logged.get("base64")
    if isinstance(encoded, str):
        logged["base64"] = encoded.encode(
            "ascii", errors="replace"
        )[:64].decode("ascii")
    return logged


def _legacy_payload(article_url: str | None, **values) -> dict[str, Any]:
    payload = {"code": 0, "article_url": article_url, "istar_asr": "", "asr": [], "message": ""}
    payload.update(values)
    return payload


def _error_payload(error: APIError) -> dict[str, Any]:
    return {
        "code": error.code,
        "article_url": None,
        "istar_asr": "",
        "asr": [],
        "error": error.error,
        "message": error.message,
    }


def _native_error_payload(message: str, code: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "gateway_error", "code": code}}


def _error_response(request: web.Request, error: APIError) -> web.Response:
    if request.path == "/chinese_asr":
        payload = _error_payload(error)
        status = error.status
    else:
        payload = _native_error_payload(error.message, error.error)
        status = error.native_status
    request["log_fields"].update(
        business_code=error.code,
        error_code=error.error,
        output_fields=list(payload),
    )
    if request.path == "/chinese_asr":
        request["log_fields"]["result_json"] = payload
    return _json_response(request, payload, status=status)


@web.middleware
async def request_log_middleware(request: web.Request, handler):
    """建立请求上下文，并把同一阶段耗时写入日志和 Server-Timing。"""
    request_id = generate_request_id(request.headers.get("X-Request-ID"))
    token = request_id_var.set(request_id)
    started = perf_counter()
    stage_timings_ms = _new_stage_timings()
    request["stage_timings_ms"] = stage_timings_ms
    request["log_fields"] = {
        "method": request.method if request.method in _LOGGED_METHODS else "OTHER",
        "path": _logged_path(request.path),
        "content_type": _logged_media_type(request.content_type),
        "content_length": request.content_length,
        "input_fields": [],
        "output_fields": [],
        "stage_timings_ms": stage_timings_ms,
    }
    status = 500
    outcome = "completed"
    response: web.StreamResponse | None = None
    try:
        response = await handler(request)
        status = response.status
        stage_timings_ms["total"] = round((perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = _server_timing_header(stage_timings_ms)
        body = getattr(response, "body", None)
        if isinstance(body, (bytes, bytearray, memoryview)):
            request["log_fields"]["response_bytes"] = len(body)
        return response
    except asyncio.CancelledError:
        status = 499
        outcome = "cancelled"
        request["log_fields"].update(error_code="REQUEST_CANCELLED")
        raise
    except (ConnectionResetError, BrokenPipeError):
        status = 499
        outcome = "client_disconnected"
        request["log_fields"].update(error_code="CLIENT_DISCONNECTED")
        raise
    finally:
        if response is None:
            stage_timings_ms["total"] = round((perf_counter() - started) * 1000, 2)
        fields = request["log_fields"]
        fields.update(
            status=status,
            outcome=outcome,
            request_latency_ms=round((perf_counter() - started) * 1000, 2),
        )
        level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
        if request.path in {"/health", "/metrics", "/v1/models"} and status < 400:
            level = logging.DEBUG
        log_event(level, "http_request_completed", "HTTP 请求处理完成", **fields)
        request_id_var.reset(token)


@web.middleware
async def error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except APIError as error:
        return _error_response(request, error)
    except web.HTTPException as exception:
        if request.path == "/chinese_asr":
            definition = (
                ErrorCode.AUDIO_TOO_LONG if exception.status == 413
                else ErrorCode.INPUT_PARAM_FAILED
            )
            message = (
                "请求体或音频超过大小限制" if exception.status == 413
                else exception.text or "请求参数错误"
            )
            return _error_response(request, APIError(definition, message))
        payload = _native_error_payload(
            exception.text or exception.reason, exception.reason
        )
        request["log_fields"].update(
            error_code=type(exception).__name__,
            output_fields=list(payload),
        )
        return _json_response(
            request,
            payload,
            status=exception.status,
        )
    except asyncio.TimeoutError:
        error = APIError(ErrorCode.ASR_INFER_FAILED, "ASR 推理超时", native_status=504)
        return _error_response(request, error)
    except (ConnectionResetError, BrokenPipeError):
        request["log_fields"].update(error_code="CLIENT_DISCONNECTED")
        raise
    except Exception as exception:  # noqa: BLE001
        request["log_fields"].update(
            business_code=ErrorCode.ASR_INFER_FAILED[0],
            error_code=ErrorCode.ASR_INFER_FAILED[1],
            exception_type=type(exception).__name__,
        )
        logger.error(
            "网关未处理异常",
            extra={
                "event": "http_request_failed",
                "extra_fields": request["log_fields"],
            },
            exc_info=True,
        )
        error = APIError(ErrorCode.ASR_INFER_FAILED, "ASR 推理失败")
        return _error_response(request, error)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _split_audio(audio: bytes, filename: str) -> list[AudioChunk]:
    try:
        source = sf.SoundFile(io.BytesIO(audio))
    except (sf.LibsndfileError, RuntimeError) as error:
        raise APIError(
            ErrorCode.DECODE_FAILED, "音频解码失败，请确认为有效音频格式"
        ) from error
    with source:
        if source.samplerate <= 0 or source.frames <= 0:
            raise APIError(ErrorCode.DECODE_FAILED, "音频内容为空或采样率无效")
        duration = source.frames / source.samplerate
        if duration > settings.max_audio_seconds:
            raise APIError(
                ErrorCode.AUDIO_TOO_LONG,
                f"音频超过最大时长 {settings.max_audio_seconds:g} 秒",
            )
        frames_per_chunk = int(source.samplerate * settings.chunk_seconds)
        if frames_per_chunk <= 0:
            raise APIError(ErrorCode.AUDIO_SEGMENT_ERROR, "音频切段参数无效")
        if source.frames <= frames_per_chunk:
            return [AudioChunk(
                audio,
                filename,
                "application/octet-stream",
                0.0,
                duration,
                int(source.frames * source.channels * 4),
            )]
        chunks: list[AudioChunk] = []
        stem = Path(filename).stem or "audio"
        index = 0
        while source.tell() < source.frames:
            start_frame = source.tell()
            samples = source.read(frames_per_chunk, dtype="float32", always_2d=True)
            if not len(samples):
                break
            end_frame = source.tell()
            buffer = io.BytesIO()
            try:
                sf.write(buffer, samples, source.samplerate, format="WAV", subtype="PCM_16")
            except (sf.LibsndfileError, RuntimeError, ValueError) as error:
                raise APIError(ErrorCode.AUDIO_SEGMENT_ERROR, "音频切段处理失败") from error
            chunks.append(AudioChunk(
                buffer.getvalue(), f"{stem}_{index:03d}.wav", "audio/wav",
                start_frame / source.samplerate, end_frame / source.samplerate,
                int(samples.nbytes),
            ))
            index += 1
        if not chunks:
            raise APIError(ErrorCode.AUDIO_SEGMENT_ERROR, "音频切段结果为空")
        return chunks


def _merge_text(parts: list[str]) -> str:
    merged = ""
    for part in (text.strip() for text in parts):
        if not part:
            continue
        if (merged and merged[-1].isascii() and merged[-1].isalnum()
                and part[0].isascii() and part[0].isalnum()):
            merged += " "
        merged += part
    return merged


def _normalize_hotwords(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(word, str) for word in value):
        raise APIError(ErrorCode.INPUT_PARAM_FAILED, "hotwords 必须是字符串数组或 null")
    result: list[str] = []
    seen: set[str] = set()
    for raw in value:
        word = raw.strip()
        if not word or word in seen:
            continue
        if len(word) > settings.max_hotword_length:
            raise APIError(ErrorCode.INPUT_PARAM_FAILED, "单个热词长度超过限制")
        seen.add(word)
        result.append(word)
    if (len(result) > settings.max_hotwords
            or sum(map(len, result)) > settings.max_hotword_chars):
        raise APIError(ErrorCode.INPUT_PARAM_FAILED, "热词数量或总长度超过限制")
    return result


def _hotword_prompt(hotwords: list[str]) -> str | None:
    if not settings.enable_hotword or not hotwords:
        return None
    return "请准确识别音频中可能出现的专有名词：" + "、".join(hotwords) + "。"


async def _post_transcription_chunk(
    app: web.Application,
    chunk: AudioChunk,
    fields: dict[str, str],
    language: str = "",
    timing: dict[str, float] | None = None,
) -> ASRChunkResult:
    form = aiohttp.FormData()
    form.add_field("file", chunk.audio, filename=chunk.filename,
                   content_type=chunk.content_type or "application/octet-stream")
    for name, value in fields.items():
        form.add_field(name, value)
    http_started = perf_counter()
    try:
        async with app["session"].post(
            f"{settings.backend_url}/v1/audio/transcriptions", data=form
        ) as response:
            status = response.status
            if timing is not None:
                timing["upstream_status"] = status
            body = await response.text()
            if timing is not None:
                timing["upstream_response_bytes"] = len(body.encode("utf-8"))
    except aiohttp.ClientConnectionError as error:
        raise APIError(
            ErrorCode.MODEL_LOAD_FAILED,
            "ASR 模型服务不可用",
            native_status=502,
        ) from error
    except asyncio.TimeoutError as error:
        raise APIError(
            ErrorCode.ASR_INFER_FAILED,
            "ASR 推理超时",
            native_status=504,
        ) from error
    finally:
        if timing is not None:
            timing["http_ms"] = (perf_counter() - http_started) * 1000
    if status in (429, 503):
        raise APIError(ErrorCode.SERVICE_BUSY, "服务繁忙，请稍后重试")
    if status != 200:
        raise APIError(
            ErrorCode.ASR_INFER_FAILED,
            "ASR 推理失败",
            native_status=502,
        )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise APIError(
            ErrorCode.ASR_INFER_FAILED,
            "ASR 后端响应异常",
            native_status=502,
        ) from error
    text = payload.get("text")
    if not isinstance(text, str):
        raise APIError(
            ErrorCode.ASR_INFER_FAILED,
            "ASR 后端响应缺少文本",
            native_status=502,
        )
    return ASRChunkResult(text, language)


async def _post_chat_chunk(
    app: web.Application,
    chunk: AudioChunk,
    prompt: str | None,
    timing: dict[str, float] | None = None,
) -> ASRChunkResult:
    audio_url = (
        f"data:{chunk.content_type or 'application/octet-stream'};base64,"
        + base64.b64encode(chunk.audio).decode("ascii")
    )
    messages: list[dict[str, Any]] = []
    if prompt:
        messages.append({"role": "system", "content": prompt})
    messages.append({
        "role": "user",
        "content": [{"type": "audio_url", "audio_url": {"url": audio_url}}],
    })
    request_body = {
        "model": settings.served_model_name,
        "messages": messages,
        "temperature": 0,
    }
    http_started = perf_counter()
    try:
        async with app["session"].post(
            f"{settings.backend_url}/v1/chat/completions", json=request_body
        ) as response:
            status = response.status
            if timing is not None:
                timing["upstream_status"] = status
            body = await response.text()
            if timing is not None:
                timing["upstream_response_bytes"] = len(body.encode("utf-8"))
    except aiohttp.ClientConnectionError as error:
        raise APIError(
            ErrorCode.MODEL_LOAD_FAILED,
            "ASR 模型服务不可用",
            native_status=502,
        ) from error
    except asyncio.TimeoutError as error:
        raise APIError(
            ErrorCode.ASR_INFER_FAILED,
            "ASR 推理超时",
            native_status=504,
        ) from error
    finally:
        if timing is not None:
            timing["http_ms"] = (perf_counter() - http_started) * 1000
    if status in (429, 503):
        raise APIError(ErrorCode.SERVICE_BUSY, "服务繁忙，请稍后重试")
    if status != 200:
        raise APIError(ErrorCode.ASR_INFER_FAILED, "ASR 推理失败", native_status=502)
    try:
        payload = json.loads(body)
        content = payload["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise APIError(
            ErrorCode.ASR_INFER_FAILED,
            "ASR 后端响应缺少原始文本",
            native_status=502,
        ) from error
    if not isinstance(content, str):
        raise APIError(
            ErrorCode.ASR_INFER_FAILED,
            "ASR 后端原始文本类型异常",
            native_status=502,
        )
    return _parse_raw_asr_output(content)


async def _recognize_chunks(
    app: web.Application,
    chunks: list[AudioChunk],
    fields: dict[str, str],
    requested_language: tuple[str, str] | None = None,
    prompt: str | None = None,
    detect_language: bool = True,
    on_chunk_ready: Callable[[int, ASRChunkResult], Awaitable[None]] | None = None,
) -> tuple[list[ASRChunkResult], dict[str, float]]:
    """并发识别分片，返回有序结果及队列、纯 HTTP 和 gather 墙钟计时。

    传入 ``on_chunk_ready`` 时，每个分片识别成功后立即回调，用于让下游阶段与 ASR 重叠。
    回调在分片并发槽位释放之后执行，因此不会延长该分片对本地与全局在途额度的占用。
    """
    if detect_language and requested_language is None:
        backend_api = "chat"
        post = lambda chunk, timing: _post_chat_chunk(app, chunk, prompt, timing)
    elif requested_language is not None:
        backend_api = "transcriptions"
        language_name, language_code = requested_language
        transcription_fields = dict(fields)
        transcription_fields["to_language"] = language_code
        if prompt:
            transcription_fields["prompt"] = prompt
        post = lambda chunk, timing: _post_transcription_chunk(
            app, chunk, transcription_fields, language_name, timing
        )
    else:
        backend_api = "transcriptions"
        post = lambda chunk, timing: _post_transcription_chunk(
            app, chunk, fields, timing=timing
        )

    local_slots = asyncio.Semaphore(settings.chunk_concurrency)

    async def run_chunk(
        index: int, chunk: AudioChunk
    ) -> tuple[ASRChunkResult, dict[str, float]]:
        """识别单个分片；提供 on_chunk_ready 时在本分片完成后立即回调。"""
        started = perf_counter()
        timing = {
            "local_wait_ms": 0.0,
            "global_wait_ms": 0.0,
            "http_ms": 0.0,
            "upstream_status": 0,
            "upstream_response_bytes": 0,
        }
        local_acquired = False
        global_acquired = False
        try:
            if len(chunks) > 1:
                queue_started = perf_counter()
                await local_slots.acquire()
                local_acquired = True
                timing["local_wait_ms"] = (perf_counter() - queue_started) * 1000

                queue_started = perf_counter()
                await app["long_chunk_slots"].acquire()
                global_acquired = True
                timing["global_wait_ms"] = (perf_counter() - queue_started) * 1000
            result = await post(chunk, timing)
        except (Exception, asyncio.CancelledError) as error:
            total_ms = (perf_counter() - started) * 1000
            root_error: BaseException = error
            cause_depth = 0
            seen_errors: set[int] = set()
            while (
                root_error.__cause__ is not None
                and id(root_error) not in seen_errors
                and cause_depth < 8
            ):
                seen_errors.add(id(root_error))
                root_error = root_error.__cause__
                cause_depth += 1
            root_errno = getattr(root_error, "errno", None)
            if not isinstance(root_errno, int):
                os_error = getattr(root_error, "os_error", None)
                root_errno = getattr(os_error, "errno", None)
            exception_fields: dict[str, Any] = {
                "exception_type": type(error).__name__,
                "root_exception_type": type(root_error).__name__,
                "exception_cause_depth": cause_depth,
            }
            if isinstance(root_errno, int):
                exception_fields["root_errno"] = root_errno
            upstream_status = timing["upstream_status"]
            if isinstance(upstream_status, int) and upstream_status > 0:
                exception_fields["upstream_status"] = upstream_status
                exception_fields["upstream_response_bytes"] = int(
                    timing["upstream_response_bytes"]
                )
            log_event(
                logging.WARNING,
                "backend_chunk_failed",
                "ASR 后端分片失败",
                backend_api=backend_api,
                chunk_index=index,
                chunk_start_ms=round(chunk.start * 1000),
                chunk_end_ms=round(chunk.end * 1000),
                chunk_audio_bytes=len(chunk.audio),
                backend_latency_ms=round(timing["http_ms"], 2),
                backend_total_latency_ms=round(total_ms, 2),
                local_queue_wait_ms=round(timing["local_wait_ms"], 2),
                global_queue_wait_ms=round(timing["global_wait_ms"], 2),
                backend_queue_wait_ms=round(
                    timing["local_wait_ms"] + timing["global_wait_ms"], 2
                ),
                **exception_fields,
            )
            raise
        finally:
            if global_acquired:
                app["long_chunk_slots"].release()
            if local_acquired:
                local_slots.release()
        total_ms = (perf_counter() - started) * 1000
        log_event(
            logging.DEBUG,
            "backend_chunk_completed",
            "ASR 后端分片完成",
            backend_api=backend_api,
            chunk_index=index,
            chunk_start_ms=round(chunk.start * 1000),
            chunk_end_ms=round(chunk.end * 1000),
            chunk_audio_bytes=len(chunk.audio),
            backend_latency_ms=round(timing["http_ms"], 2),
            backend_total_latency_ms=round(total_ms, 2),
            local_queue_wait_ms=round(timing["local_wait_ms"], 2),
            global_queue_wait_ms=round(timing["global_wait_ms"], 2),
            backend_queue_wait_ms=round(
                timing["local_wait_ms"] + timing["global_wait_ms"], 2
            ),
            result_chars=len(result.text),
            language=_log_language_label(result.language),
        )
        if on_chunk_ready is not None:
            await on_chunk_ready(index, result)
        return result, timing

    gather_started = perf_counter()
    gathered = await asyncio.gather(*(
        run_chunk(index, chunk) for index, chunk in enumerate(chunks)
    ))
    asr_ms = (perf_counter() - gather_started) * 1000
    results = [result for result, _ in gathered]
    chunk_timings = [timing for _, timing in gathered]
    local_waits = [timing["local_wait_ms"] for timing in chunk_timings]
    global_waits = [timing["global_wait_ms"] for timing in chunk_timings]
    timings = {
        "chunk-local-queue": round(sum(local_waits), 2),
        "chunk-local-queue-max": round(max(local_waits, default=0.0), 2),
        "chunk-global-queue": round(sum(global_waits), 2),
        "chunk-global-queue-max": round(max(global_waits, default=0.0), 2),
        "asr-backend": round(sum(
            timing["http_ms"] for timing in chunk_timings
        ), 2),
        "asr": round(asr_ms, 2),
    }
    return results, timings


async def transcribe(request: web.Request) -> web.Response:
    parse_started = perf_counter()
    try:
        if not request.content_type.startswith("multipart/"):
            raise web.HTTPUnsupportedMediaType(text="请求必须使用 multipart/form-data")
        reader = await request.multipart()
        audio = b""
        filename = "audio.wav"
        content_type = "application/octet-stream"
        fields: dict[str, str] = {}
        while field := await reader.next():
            if field.name == "file":
                filename = field.filename or filename
                content_type = field.headers.get("Content-Type", content_type)
                audio = await field.read(decode=False)
            elif field.name:
                fields[field.name] = await field.text()
        if not audio:
            raise web.HTTPBadRequest(text="缺少非空 file 字段")
        if len(audio) > settings.max_upload_mb * 1024**2:
            raise web.HTTPRequestEntityTooLarge(
                max_size=settings.max_upload_mb * 1024**2, actual_size=len(audio)
            )
        fields.setdefault("model", settings.served_model_name)
        fields["response_format"] = "json"
    finally:
        _finish_stage(request, "request-parse", parse_started)

    split_started = perf_counter()
    try:
        chunks = _split_audio(audio, filename)
        if len(chunks) == 1:
            chunks[0].content_type = content_type
    finally:
        _finish_stage(request, "audio-split", split_started)
    request["log_fields"].update(
        input_fields=[
            name for name in ("file", "model", "response_format", "language", "prompt")
            if name == "file" or name in fields
        ],
        audio_bytes=len(audio),
        audio_duration_ms=round(chunks[-1].end * 1000),
        chunk_count=len(chunks),
        audio_content_type=_logged_media_type(content_type),
        language_mode="explicit" if fields.get("language") else "auto",
        requested_language_present=bool(fields.get("language")),
        prompt_present=bool(fields.get("prompt")),
    )
    results, asr_timings = await _recognize_chunks(
        request.app, chunks, fields, detect_language=False
    )
    request["stage_timings_ms"].update(asr_timings)

    postprocess_started = perf_counter()
    texts = [result.text for result in results]
    merged_text = _merge_text(texts)
    output = {"text": merged_text}
    request["log_fields"].update(
        output_fields=list(output),
        result_chars=len(merged_text),
        output_chunk_count=len(results),
    )
    _finish_stage(request, "postprocess", postprocess_started)
    return _json_response(
        request,
        output,
        headers={"X-Audio-Chunks": str(len(chunks))},
    )


async def chinese_asr(request: web.Request) -> web.Response:
    parse_started = perf_counter()
    try:
        if request.content_type != "application/json":
            raise APIError(ErrorCode.INPUT_PARAM_FAILED, "Content-Type 必须是 application/json")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise APIError(ErrorCode.INPUT_PARAM_FAILED, "请求 JSON 格式错误") from error
        request["log_fields"]["input_json"] = _logged_chinese_asr_input(payload)
        if not isinstance(payload, dict):
            raise APIError(ErrorCode.INPUT_PARAM_FAILED, "请求体必须是 JSON 对象")
        encoded = payload.get("base64")
        if not isinstance(encoded, str) or not encoded.strip():
            raise APIError(ErrorCode.INPUT_PARAM_FAILED, "base64 是必填非空字符串")
        article_url = payload.get("article_url")
        if article_url is not None and not isinstance(article_url, str):
            raise APIError(ErrorCode.INPUT_PARAM_FAILED, "article_url 必须是字符串或 null")
        hotwords = _normalize_hotwords(payload.get("hotwords"))
        language_value = payload.get("language")
        if language_value is not None and not isinstance(language_value, str):
            raise APIError(ErrorCode.INPUT_PARAM_FAILED, "language 必须是字符串或 null")
        requested_language = _normalize_requested_language(language_value)
    finally:
        _finish_stage(request, "request-parse", parse_started)

    decode_started = perf_counter()
    try:
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise APIError(ErrorCode.DECODE_FAILED, "Base64 解码失败") from error
    finally:
        _finish_stage(request, "base64-decode", decode_started)
    if not audio:
        raise APIError(ErrorCode.DECODE_FAILED, "音频内容为空")
    if len(audio) > settings.max_upload_mb * 1024**2:
        raise APIError(ErrorCode.AUDIO_TOO_LONG, "音频文件超过大小限制")

    split_started = perf_counter()
    try:
        chunks = _split_audio(audio, "audio.wav")
    finally:
        _finish_stage(request, "audio-split", split_started)
    request["log_fields"].update(
        input_fields=[
            name for name in ("base64", "article_url", "hotwords", "language")
            if name in payload
        ],
        base64_encoded_chars=len(encoded),
        audio_bytes=len(audio),
        audio_duration_ms=round(chunks[-1].end * 1000),
        chunk_count=len(chunks),
        language_mode="explicit" if requested_language else "auto",
        requested_language=requested_language[0] if requested_language else "",
        hotword_count=len(hotwords),
        hotword_chars=sum(map(len, hotwords)),
        article_url_present=article_url is not None,
    )
    fields = {
        "model": settings.served_model_name,
        "response_format": "json",
    }
    prompt = _hotword_prompt(hotwords)
    # 两条分支都必须产出 results、texts、languages 与 alignment，供后续合并与响应组装使用。
    pipeline_mode = settings.aligner_pipeline_enabled and forced_aligner.enabled
    async with forced_aligner.prepare_chunks(chunks) as alignment_preparation:
        if pipeline_mode:
            # 分片级流水：每个分片 ASR 完成即提交对齐，两个阶段重叠。因此
            # aligner_latency_ms 覆盖“首个分片提交到全部对齐完成”，与 ASR 时间部分重叠，
            # 不能与批量模式的同名字段直接相减比较。
            # 该路径已实测拒绝（PERF-ALI-009）：提前提交会打散动态微批，对齐阶段反而变慢。
            # 仅保留供 Aligner 改由 vLLM 调度器管理合批后重新实测，生产默认走 else 分支。
            pipeline = forced_aligner.start_pipeline(
                chunks, preparation=alignment_preparation
            )
            aligner_started = perf_counter()
            try:
                results, asr_timings = await _recognize_chunks(
                    request.app,
                    chunks,
                    fields,
                    requested_language=requested_language,
                    prompt=prompt,
                    on_chunk_ready=lambda index, result: pipeline.submit(
                        index, result.text, result.language
                    ),
                )
                request["stage_timings_ms"].update(asr_timings)
                alignment = await pipeline.finish()
                texts = [result.text for result in results]
                languages = [result.language for result in results]
            except (asyncio.TimeoutError, RuntimeError, ValueError) as error:
                pipeline.close()
                log_event(
                    logging.ERROR,
                    "aligner_failed",
                    "ForcedAligner 对齐失败",
                    chunk_count=len(chunks),
                    aligner_pipeline=True,
                    aligner_latency_ms=round(
                        (perf_counter() - aligner_started) * 1000, 2
                    ),
                    exception_type=type(error).__name__,
                )
                raise APIError(
                    ErrorCode.ALIGNER_INFER_FAILED, "时间戳对齐失败"
                ) from error
            except BaseException:
                pipeline.close()
                raise
        else:
            results, asr_timings = await _recognize_chunks(
                request.app,
                chunks,
                fields,
                requested_language=requested_language,
                prompt=prompt,
            )
            request["stage_timings_ms"].update(asr_timings)
            texts = [result.text for result in results]
            languages = [result.language for result in results]

            aligner_started = perf_counter()
            try:
                alignment = await forced_aligner.align_chunks(
                    chunks,
                    texts,
                    languages,
                    preparation=alignment_preparation,
                )
            except (asyncio.TimeoutError, RuntimeError, ValueError) as error:
                log_event(
                    logging.ERROR,
                    "aligner_failed",
                    "ForcedAligner 对齐失败",
                    chunk_count=len(chunks),
                    aligner_pipeline=False,
                    aligner_latency_ms=round(
                        (perf_counter() - aligner_started) * 1000, 2
                    ),
                    exception_type=type(error).__name__,
                )
                raise APIError(
                    ErrorCode.ALIGNER_INFER_FAILED, "时间戳对齐失败"
                ) from error
    request["stage_timings_ms"]["aligner-queue"] = alignment.queue_wait_ms
    request["stage_timings_ms"]["aligner"] = alignment.inference_ms
    postprocess_started = perf_counter()
    aligned = alignment.units
    skipped_indices = {item.chunk_index for item in alignment.skipped}
    skipped_languages = sorted({
        _log_language_label(item.language) or "Unknown"
        for item in alignment.skipped
    })
    if settings.timestamp_enabled:
        log_event(
            logging.INFO,
            "aligner_completed",
            "ForcedAligner 对齐完成",
            chunk_count=len(chunks),
            aligner_pipeline=pipeline_mode,
            aligner_latency_ms=round((perf_counter() - aligner_started) * 1000, 2),
            aligner_queue_wait_ms=alignment.queue_wait_ms,
            aligner_inference_ms=alignment.inference_ms,
            aligner_batch_count=alignment.batch_count,
            aligner_batch_size_max=alignment.batch_size_max,
            aligner_batch_size_mean=alignment.batch_size_mean,
            aligner_queue_depth_max=alignment.queue_depth_max,
            aligner_batch_audio_decode_ms=alignment.batch_audio_decode_ms,
            aligner_batch_model_call_ms=alignment.batch_model_call_ms,
            aligner_batch_result_build_ms=alignment.batch_result_build_ms,
            aligner_predecode_wait_ms=alignment.predecode_wait_ms,
            aligner_predecode_audio_decode_ms=alignment.predecode_audio_decode_ms,
            aligner_predecode_count=alignment.predecode_count,
            aligned_unit_count=sum(map(len, aligned)),
            skipped_count=len(alignment.skipped),
            skipped_indices=sorted(skipped_indices),
            skipped_languages=skipped_languages,
        )

    warning = ""
    if alignment.skipped:
        details = ", ".join(
            f"分片 {item.chunk_index}={item.language or '未知'}"
            for item in alignment.skipped
        )
        warning = f"部分分片语种不受 ForcedAligner 支持，已跳过真实对齐：{details}"
        log_event(
            logging.WARNING,
            "aligner_chunks_skipped",
            "部分分片跳过真实对齐",
            skipped_count=len(alignment.skipped),
            skipped_indices=sorted(skipped_indices),
            skipped_languages=skipped_languages,
        )
    segments = [
        {
            "idx": index,
            "slid": result.language,
            "text": result.text,
            "speaker": "",
            "timestamp": [round(chunk.start, 3), round(chunk.end, 3)],
            "words": [
                {
                    "text": unit.text,
                    "timestamp": [round(unit.start, 3), round(unit.end, 3)],
                }
                for unit in aligned[index]
            ] if settings.enable_word_timestamp else [],
        }
        for index, (chunk, result) in enumerate(zip(chunks, results))
    ]
    merged_text = _merge_text(texts)
    if settings.enable_sentence_timestamp and not skipped_indices:
        try:
            segments = build_sentences(
                merged_text,
                [unit for chunk_units in aligned for unit in chunk_units],
            )
        except ValueError as error:
            raise APIError(ErrorCode.ALIGNER_INFER_FAILED, "句级时间戳生成失败") from error
    elif settings.enable_sentence_timestamp and skipped_indices:
        suffix = "；句级聚合已禁用，响应保留物理分片边界"
        warning = f"{warning}{suffix}"
    output = _legacy_payload(
        article_url,
        istar_asr=merged_text,
        asr=segments,
        message=warning,
    )
    request["log_fields"].update(
        business_code=0,
        output_fields=list(output),
        result_json=output,
        result_chars=len(merged_text),
        segment_count=len(segments),
        word_unit_count=sum(len(segment["words"]) for segment in segments),
        result_languages=sorted({
            label for language in languages
            if (label := _log_language_label(language))
        }),
        alignment_skipped_count=len(alignment.skipped),
        aligner_batch_count=alignment.batch_count,
        aligner_batch_size_max=alignment.batch_size_max,
        aligner_batch_size_mean=alignment.batch_size_mean,
        aligner_queue_depth_max=alignment.queue_depth_max,
        aligner_batch_audio_decode_ms=alignment.batch_audio_decode_ms,
        aligner_batch_model_call_ms=alignment.batch_model_call_ms,
        aligner_batch_result_build_ms=alignment.batch_result_build_ms,
        aligner_predecode_wait_ms=alignment.predecode_wait_ms,
        aligner_predecode_audio_decode_ms=alignment.predecode_audio_decode_ms,
        aligner_predecode_count=alignment.predecode_count,
        warning_present=bool(warning),
    )
    _finish_stage(request, "postprocess", postprocess_started)
    return _json_response(
        request,
        output,
        headers={"X-Audio-Chunks": str(len(chunks))},
    )


async def health(request: web.Request) -> web.Response:
    try:
        async with request.app["session"].get(
            f"{settings.backend_url}/health"
        ) as response:
            if response.status != 200:
                raise web.HTTPServiceUnavailable(text="vLLM 后端未就绪")
    except aiohttp.ClientError as error:
        raise web.HTTPServiceUnavailable(text=f"无法连接 vLLM 后端: {error}") from error
    return _json_response(
        request,
        {
            "status": "ok",
            "version": settings.service_version,
            "model": settings.served_model_name,
            "timestamps": settings.timestamp_enabled,
            "aligner_device": settings.aligner_device if settings.timestamp_enabled else None,
        },
    )


async def proxy_get(request: web.Request) -> web.Response:
    try:
        async with request.app["session"].get(
            f"{settings.backend_url}{request.path}"
        ) as response:
            body = await response.read()
            request["log_fields"].update(
                upstream_status=response.status,
                upstream_response_bytes=len(body),
            )
            if response.status >= 400:
                request["log_fields"]["error_code"] = f"UPSTREAM_HTTP_{response.status}"
            headers = {}
            if content_type := response.headers.get("Content-Type"):
                headers["Content-Type"] = content_type
            return web.Response(body=body, status=response.status, headers=headers)
    except asyncio.TimeoutError as error:
        request["log_fields"].update(error_code="UPSTREAM_TIMEOUT")
        raise web.HTTPGatewayTimeout(text="vLLM 后端请求超时") from error
    except aiohttp.ClientError as error:
        request["log_fields"].update(error_code="UPSTREAM_UNAVAILABLE")
        raise web.HTTPBadGateway(text="vLLM 后端不可用") from error


async def _session_context(app: web.Application):
    setup_logger()
    log_event(
        logging.INFO,
        "gateway_starting",
        "网关开始初始化",
        service_version=settings.service_version,
        served_model_name=settings.served_model_name,
        gateway_host=settings.gateway_host,
        gateway_port=settings.gateway_port,
        timestamp_enabled=settings.timestamp_enabled,
        log_level=settings.log_level,
        log_file_enabled=settings.log_file_enabled,
        log_max_file_mb=settings.log_max_file_mb,
        log_backup_count=settings.log_backup_count,
        log_retention_days=settings.log_retention_days,
        log_queue_size=settings.log_queue_size,
        backend_connection_limit=settings.backend_connection_limit,
        backend_keepalive_timeout=settings.backend_keepalive_timeout,
        aligner_concurrency=settings.aligner_concurrency,
        aligner_device=settings.aligner_device,
        aligner_batch_size=settings.aligner_batch_size,
        aligner_decode_workers=settings.aligner_decode_workers,
        aligner_predecode_enabled=settings.aligner_predecode_enabled,
        aligner_predecode_max_mb=settings.aligner_predecode_max_mb,
        aligner_batch_wait_ms=settings.aligner_batch_wait_ms,
        aligner_queue_size=settings.aligner_queue_size,
        aligner_attention_backend_configured=settings.aligner_attention_backend,
    )
    session: aiohttp.ClientSession | None = None
    started = False
    try:
        forced_aligner.load()
        await forced_aligner.start()
        timeout = aiohttp.ClientTimeout(total=settings.backend_timeout)
        connector = aiohttp.TCPConnector(
            limit=settings.backend_connection_limit,
            limit_per_host=settings.backend_connection_limit,
            keepalive_timeout=settings.backend_keepalive_timeout,
        )
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        app["session"] = session
        app["long_chunk_slots"] = asyncio.Semaphore(settings.long_chunks_in_flight)
        started = True
        log_event(
            logging.INFO,
            "gateway_started",
            "网关初始化完成",
            service_version=settings.service_version,
            served_model_name=settings.served_model_name,
            timestamp_enabled=settings.timestamp_enabled,
            aligner_device=(
                settings.aligner_device if settings.timestamp_enabled else None
            ),
            aligner_decode_workers=settings.aligner_decode_workers,
            aligner_predecode_enabled=settings.aligner_predecode_enabled,
            aligner_predecode_max_mb=settings.aligner_predecode_max_mb,
            aligner_attention_backend_configured=settings.aligner_attention_backend,
            aligner_attention_backend_actual=forced_aligner.attention_backend_actual,
        )
        yield
    except Exception:  # noqa: BLE001 - 启动失败必须记录堆栈后继续抛出。
        logger.error(
            "网关初始化或清理失败",
            extra={
                "event": "gateway_lifecycle_failed",
                "extra_fields": {"started": started},
            },
            exc_info=True,
        )
        raise
    finally:
        try:
            await forced_aligner.close()
        finally:
            try:
                if session is not None and not session.closed:
                    await session.close()
            finally:
                log_event(
                    logging.INFO,
                    "gateway_stopped",
                    "网关已停止",
                    service_version=settings.service_version,
                    started=started,
                )
                shutdown_logging()


def create_app() -> web.Application:
    app = web.Application(
        client_max_size=settings.max_json_body_mb * 1024**2,
        middlewares=[request_log_middleware, error_middleware],
    )
    app.cleanup_ctx.append(_session_context)
    app.router.add_post("/chinese_asr", chinese_asr)
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", proxy_get)
    app.router.add_get("/v1/models", proxy_get)
    return app


if __name__ == "__main__":
    web.run_app(
        create_app(),
        host=settings.gateway_host,
        port=settings.gateway_port,
        access_log=None,
    )
