"""Qwen3-ASR CPU 网关：兼容 /chinese_asr，并代理原生转写接口。"""

import asyncio
import base64
import binascii
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import soundfile as sf
from aiohttp import web

try:
    from .config import settings
except ImportError:  # 兼容 python src/gateway.py 直接执行。
    from config import settings


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


class APIError(Exception):
    def __init__(
        self,
        definition: tuple[int, str, int],
        message: str,
        native_status: int | None = None,
    ):
        self.code, self.error, self.status = definition
        self.native_status = native_status or self.status
        self.message = message
        super().__init__(message)


@dataclass
class AudioChunk:
    audio: bytes
    filename: str
    content_type: str
    start: float
    end: float


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
        return web.json_response(_error_payload(error), status=error.status, dumps=_json_dumps)
    return web.json_response(
        _native_error_payload(error.message, error.error),
        status=error.native_status,
        dumps=_json_dumps,
    )


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
        return web.json_response(
            _native_error_payload(exception.text or exception.reason, exception.reason),
            status=exception.status,
            dumps=_json_dumps,
        )
    except asyncio.TimeoutError:
        error = APIError(ErrorCode.ASR_INFER_FAILED, "ASR 推理超时", native_status=504)
        return _error_response(request, error)
    except Exception as exception:  # noqa: BLE001
        print(f"网关未处理异常: {type(exception).__name__}: {exception}", file=sys.stderr)
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
            return [AudioChunk(audio, filename, "application/octet-stream", 0.0, duration)]
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


async def _post_chunk(app: web.Application, chunk: AudioChunk, fields: dict[str, str]) -> str:
    form = aiohttp.FormData()
    form.add_field("file", chunk.audio, filename=chunk.filename,
                   content_type=chunk.content_type or "application/octet-stream")
    for name, value in fields.items():
        form.add_field(name, value)
    try:
        async with app["session"].post(
            f"{settings.backend_url}/v1/audio/transcriptions", data=form
        ) as response:
            body = await response.text()
            if response.status in (429, 503):
                raise APIError(ErrorCode.SERVICE_BUSY, "服务繁忙，请稍后重试")
            if response.status != 200:
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
            return text
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


async def _recognize_chunks(
    app: web.Application, chunks: list[AudioChunk], fields: dict[str, str]
) -> list[str]:
    if len(chunks) == 1:
        return [await _post_chunk(app, chunks[0], fields)]
    local_slots = asyncio.Semaphore(settings.chunk_concurrency)

    async def limited(chunk: AudioChunk) -> str:
        async with local_slots, app["long_chunk_slots"]:
            return await _post_chunk(app, chunk, fields)

    return await asyncio.gather(*(limited(chunk) for chunk in chunks))


async def transcribe(request: web.Request) -> web.Response:
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
    chunks = _split_audio(audio, filename)
    if len(chunks) == 1:
        chunks[0].content_type = content_type
    texts = await _recognize_chunks(request.app, chunks, fields)
    return web.json_response({"text": _merge_text(texts)},
                             headers={"X-Audio-Chunks": str(len(chunks))},
                             dumps=_json_dumps)


async def chinese_asr(request: web.Request) -> web.Response:
    if request.content_type != "application/json":
        raise APIError(ErrorCode.INPUT_PARAM_FAILED, "Content-Type 必须是 application/json")
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise APIError(ErrorCode.INPUT_PARAM_FAILED, "请求 JSON 格式错误") from error
    if not isinstance(payload, dict):
        raise APIError(ErrorCode.INPUT_PARAM_FAILED, "请求体必须是 JSON 对象")
    encoded = payload.get("base64")
    if not isinstance(encoded, str) or not encoded.strip():
        raise APIError(ErrorCode.INPUT_PARAM_FAILED, "base64 是必填非空字符串")
    article_url = payload.get("article_url")
    if article_url is not None and not isinstance(article_url, str):
        raise APIError(ErrorCode.INPUT_PARAM_FAILED, "article_url 必须是字符串或 null")
    hotwords = _normalize_hotwords(payload.get("hotwords"))
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise APIError(ErrorCode.DECODE_FAILED, "Base64 解码失败") from error
    if not audio:
        raise APIError(ErrorCode.DECODE_FAILED, "音频内容为空")
    if len(audio) > settings.max_upload_mb * 1024**2:
        raise APIError(ErrorCode.AUDIO_TOO_LONG, "音频文件超过大小限制")
    chunks = _split_audio(audio, "audio.wav")
    fields = {
        "model": settings.served_model_name,
        "response_format": "json",
        "temperature": "0",
    }
    if prompt := _hotword_prompt(hotwords):
        fields["prompt"] = prompt
    texts = await _recognize_chunks(request.app, chunks, fields)
    segments = [
        {
            "idx": index,
            "slid": "",
            "text": text,
            "speaker": "",
            "timestamp": [round(chunk.start, 3), round(chunk.end, 3)],
            "words": [],
        }
        for index, (chunk, text) in enumerate(zip(chunks, texts))
    ]
    return web.json_response(
        _legacy_payload(article_url, istar_asr=_merge_text(texts), asr=segments),
        headers={"X-Audio-Chunks": str(len(chunks))}, dumps=_json_dumps,
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
    return web.json_response(
        {
            "status": "ok",
            "version": settings.service_version,
            "model": settings.served_model_name,
        },
        dumps=_json_dumps,
    )


async def proxy_get(request: web.Request) -> web.Response:
    async with request.app["session"].get(
        f"{settings.backend_url}{request.path}"
    ) as response:
        body = await response.read()
        headers = {}
        if content_type := response.headers.get("Content-Type"):
            headers["Content-Type"] = content_type
        return web.Response(body=body, status=response.status, headers=headers)


async def _session_context(app: web.Application):
    timeout = aiohttp.ClientTimeout(total=settings.backend_timeout)
    app["session"] = aiohttp.ClientSession(timeout=timeout)
    app["long_chunk_slots"] = asyncio.Semaphore(settings.long_chunks_in_flight)
    yield
    await app["session"].close()


def create_app() -> web.Application:
    app = web.Application(
        client_max_size=settings.max_json_body_mb * 1024**2,
        middlewares=[error_middleware],
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
