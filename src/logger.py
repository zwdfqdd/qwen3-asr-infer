"""网关结构化日志：有界异步队列、受限业务 JSON、统计摘要与文件轮转。"""

import atexit
import copy
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import unicodedata
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any

try:
    from .config import settings
except ImportError:  # 兼容 python src/gateway.py 直接执行。
    from config import settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,64}")
_RESERVED_FIELDS = {
    "timestamp", "level", "logger", "event", "message", "request_id",
}
_MAX_STRING_CHARS = 512
_MAX_EXCEPTION_CHARS = 8192
_MAX_COLLECTION_ITEMS = 64
# result_json 的 words[].timestamp 需要 6 层才能完整进入安全序列化。
_MAX_NESTING = 6


def _utc_timestamp(created: float) -> str:
    return datetime.fromtimestamp(created, timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _safe_string(value: str, limit: int = _MAX_STRING_CHARS) -> str:
    """转义控制/格式字符并限制长度，避免日志欺骗和单字段放大。"""
    escaped = "".join(
        f"\\u{ord(char):04x}" if unicodedata.category(char) in {"Cc", "Cf"} else char
        for char in value
    )
    if len(escaped) <= limit:
        return escaped
    omitted = len(escaped) - limit
    return f"{escaped[:limit]}…<截断 {omitted} 字符>"

def _sanitize(value: Any, depth: int = 0) -> Any:
    """只允许有界 JSON 值；未知对象不调用其 __str__，二进制仅记录长度。"""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "<非有限数值>"
    if isinstance(value, str):
        return _safe_string(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"redacted_binary_bytes": len(value)}
    if depth >= _MAX_NESTING:
        return "<达到最大嵌套深度>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                result["truncated_items"] = len(value) - _MAX_COLLECTION_ITEMS
                break
            safe_key = _safe_string(key if isinstance(key, str) else type(key).__name__, 128)
            result[safe_key] = _sanitize(item, depth + 1)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = [_sanitize(item, depth + 1) for item in items[:_MAX_COLLECTION_ITEMS]]
        if len(items) > _MAX_COLLECTION_ITEMS:
            result.append({"truncated_items": len(items) - _MAX_COLLECTION_ITEMS})
        return result
    return f"<{type(value).__name__}>"


class RequestContextFilter(logging.Filter):
    """在请求线程入队前固化 ContextVar，避免监听线程丢失请求 ID。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class BoundedQueueHandler(QueueHandler):
    """队列满时不阻塞事件循环，并把丢弃数附加到下一条成功入队日志。"""

    def __init__(self, log_queue: queue.Queue[logging.LogRecord]) -> None:
        super().__init__(log_queue)
        self._dropped = 0
        self._drop_lock = threading.Lock()

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        # 监听线程与生产线程同进程，浅拷贝可保留 exc_info 供 JSONFormatter 单独输出。
        return copy.copy(record)

    def enqueue(self, record: logging.LogRecord) -> None:
        with self._drop_lock:
            dropped = self._dropped
            self._dropped = 0
        if dropped:
            record.dropped_logs_before = dropped
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            with self._drop_lock:
                self._dropped += dropped + 1

    def pending_drops(self) -> int:
        with self._drop_lock:
            return self._dropped


class DrainingQueueListener(QueueListener):
    """关闭时允许阻塞等待一个队列空位，保证哨兵入队并排空已有日志。"""

    def enqueue_sentinel(self) -> None:
        self.queue.put(self._sentinel)


def _write_internal_error(event: str, message: str) -> None:
    """handler 故障时绕过 logging 输出固定 JSON，绝不包含原日志字段。"""
    payload = json.dumps({
        "timestamp": _utc_timestamp(time.time()),
        "level": "ERROR",
        "logger": "qwen3_asr.gateway",
        "event": event,
        "message": message,
        "request_id": "",
    }, ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        os.write(2, payload.encode("utf-8", errors="replace"))
    except OSError:
        pass


class ResilientStreamHandler(logging.StreamHandler):
    """stdout 失败后停用自身，避免 logging 内部 traceback 污染日志流。"""

    def __init__(self) -> None:
        super().__init__(sys.stdout)
        self._disabled = False

    def emit(self, record: logging.LogRecord) -> None:
        if not self._disabled:
            super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        self._disabled = True
        _write_internal_error("stdout_logger_disabled", "stdout 日志写入失败，已停用该输出")


class ResilientRotatingFileHandler(RotatingFileHandler):
    """文件写入或轮转失败后停用自身，stdout 队列输出仍可继续。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._disabled = False

    def emit(self, record: logging.LogRecord) -> None:
        if not self._disabled:
            super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        self._disabled = True
        _write_internal_error("file_logger_disabled", "文件日志写入失败，已停用该输出")


class JSONFormatter(logging.Formatter):
    """输出严格单行 JSON；格式化失败时返回不含业务值的固定降级事件。"""

    def format(self, record: logging.LogRecord) -> str:
        try:
            data: dict[str, Any] = {
                "timestamp": _utc_timestamp(record.created),
                "level": record.levelname,
                "logger": _safe_string(record.name, 128),
                "event": _safe_string(str(getattr(record, "event", "log")), 128),
                "message": _safe_string(record.getMessage()),
                "request_id": _safe_string(str(getattr(record, "request_id", "")), 64),
            }
            fields = getattr(record, "extra_fields", {})
            if isinstance(fields, dict):
                data.update(
                    (_safe_string(key, 128), _sanitize(value))
                    for key, value in fields.items()
                    if isinstance(key, str) and key not in _RESERVED_FIELDS
                )
            dropped = getattr(record, "dropped_logs_before", 0)
            if isinstance(dropped, int) and dropped > 0:
                data["dropped_logs_before"] = dropped
            if record.exc_info:
                data["exception"] = _safe_string(
                    self.formatException(record.exc_info), _MAX_EXCEPTION_CHARS
                )
            return json.dumps(
                data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        except Exception:  # noqa: BLE001 - 日志格式化绝不能反向打断业务请求。
            fallback = {
                "timestamp": _utc_timestamp(getattr(record, "created", time.time())),
                "level": "ERROR",
                "logger": "qwen3_asr.gateway",
                "event": "log_format_failed",
                "message": "日志格式化失败，已丢弃原字段",
                "request_id": "",
            }
            return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))


_setup_lock = threading.Lock()
_queue_handler: BoundedQueueHandler | None = None
_listener: DrainingQueueListener | None = None
_output_handlers: list[logging.Handler] = []


def _cleanup_expired_logs(log_dir: Path) -> None:
    """清理稳定文件和旧 PID 方案产生的过期备份，跨重启执行保留策略。"""
    cutoff = time.time() - settings.log_retention_days * 86400
    candidates = set(log_dir.glob("gateway.log.*"))
    candidates.update(log_dir.glob("gateway_*.log*"))
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
        except OSError:
            # 清理失败不应阻断服务；后续 file_logger_unavailable 仅用于初始化失败。
            continue


def setup_logger() -> logging.Logger:
    """幂等初始化异步日志；文件不可写时自动降级为 stdout。"""
    global _queue_handler, _listener, _output_handlers
    result = logging.getLogger("qwen3_asr.gateway")
    result.setLevel(getattr(logging, settings.log_level))
    result.propagate = False
    with _setup_lock:
        if _queue_handler is not None:
            return result

        formatter = JSONFormatter()
        stdout_handler = ResilientStreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler._qwen_owned = True  # type: ignore[attr-defined]
        outputs: list[logging.Handler] = [stdout_handler]
        file_error = ""
        if settings.log_file_enabled:
            try:
                log_dir = Path(settings.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                _cleanup_expired_logs(log_dir)
                file_handler = ResilientRotatingFileHandler(
                    filename=str(log_dir / "gateway.log"),
                    maxBytes=settings.log_max_file_mb * 1024**2,
                    backupCount=settings.log_backup_count,
                    encoding="utf-8",
                    delay=True,
                )
                file_handler.setFormatter(formatter)
                file_handler._qwen_owned = True  # type: ignore[attr-defined]
                outputs.append(file_handler)
            except (OSError, ValueError) as error:
                file_error = type(error).__name__

        log_queue: queue.Queue[logging.LogRecord] = queue.Queue(
            maxsize=settings.log_queue_size
        )
        queue_handler = BoundedQueueHandler(log_queue)
        queue_handler.addFilter(RequestContextFilter())
        queue_handler._qwen_owned = True  # type: ignore[attr-defined]
        result.addHandler(queue_handler)
        listener = DrainingQueueListener(
            log_queue, *outputs, respect_handler_level=True
        )
        listener.start()
        _queue_handler = queue_handler
        _listener = listener
        _output_handlers = outputs

    if file_error:
        result.warning(
            "文件日志初始化失败，仅使用 stdout",
            extra={
                "event": "file_logger_unavailable",
                "extra_fields": {"exception_type": file_error},
            },
        )
    return result


def shutdown_logging() -> None:
    """停止接收新日志，排空队列并关闭输出 handler；可重复调用。"""
    global _queue_handler, _listener, _output_handlers
    with _setup_lock:
        queue_handler = _queue_handler
        listener = _listener
        outputs = _output_handlers
        _queue_handler = None
        _listener = None
        _output_handlers = []
        if queue_handler is not None:
            logging.getLogger("qwen3_asr.gateway").removeHandler(queue_handler)
    if listener is not None:
        listener.stop()
    dropped = queue_handler.pending_drops() if queue_handler is not None else 0
    for handler in outputs:
        try:
            handler.flush()
            handler.close()
        except OSError:
            pass
    if dropped:
        fallback = json.dumps({
            "timestamp": _utc_timestamp(time.time()),
            "level": "WARNING",
            "logger": "qwen3_asr.gateway",
            "event": "log_records_dropped_on_shutdown",
            "message": "日志队列关闭时仍有丢弃记录",
            "request_id": "",
            "dropped_logs": dropped,
        }, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            os.write(2, fallback.encode("utf-8", errors="replace"))
        except OSError:
            pass


def generate_request_id(candidate: str | None = None) -> str:
    """接受受限客户端请求 ID；无效或缺失时生成完整 UUID hex。"""
    value = (candidate or "").strip()
    if _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid.uuid4().hex


def log_event(
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    """非阻塞记录结构化摘要；任意值都会在监听线程中受限并安全序列化。"""
    logger.log(
        level,
        message,
        extra={"event": event, "extra_fields": fields},
    )


atexit.register(shutdown_logging)
logger = setup_logger()
