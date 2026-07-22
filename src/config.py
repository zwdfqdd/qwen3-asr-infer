"""网关系统配置：集中读取环境变量并在启动前校验。"""

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"配置 {name} 必须是整数") from error


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"配置 {name} 必须是数字") from error


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"配置 {name} 必须是 true 或 false")


@dataclass(frozen=True)
class Settings:
    service_version: str
    backend_url: str
    served_model_name: str
    gateway_host: str
    gateway_port: int
    chunk_seconds: float
    max_audio_seconds: float
    max_upload_mb: int
    max_json_body_mb: int
    chunk_concurrency: int
    long_chunks_in_flight: int
    backend_timeout: float
    enable_hotword: bool
    enable_vad: bool
    enable_word_timestamp: bool
    enable_sentence_timestamp: bool
    max_hotwords: int
    max_hotword_length: int
    max_hotword_chars: int

    @classmethod
    def from_env(cls) -> "Settings":
        result = cls(
            service_version=os.getenv("SERVICE_VERSION", "1.0.0").strip(),
            backend_url=os.getenv("BACKEND_URL", "http://127.0.0.1:8081").rstrip("/"),
            served_model_name=os.getenv("SERVED_MODEL_NAME", "qwen3-asr").strip(),
            gateway_host=os.getenv("GATEWAY_HOST", "0.0.0.0").strip(),
            gateway_port=_env_int("GATEWAY_PORT", 8080),
            chunk_seconds=_env_float("AUDIO_CHUNK_SECONDS", 30),
            max_audio_seconds=_env_float("MAX_AUDIO_SECONDS", 300),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 64),
            max_json_body_mb=_env_int("MAX_JSON_BODY_MB", 96),
            chunk_concurrency=_env_int("CHUNK_CONCURRENCY", 3),
            long_chunks_in_flight=_env_int("LONG_CHUNKS_IN_FLIGHT", 64),
            backend_timeout=_env_float("BACKEND_TIMEOUT", 300),
            enable_hotword=_env_bool("ENABLE_HOTWORD", True),
            enable_vad=_env_bool("ENABLE_VAD", False),
            enable_word_timestamp=_env_bool("ENABLE_WORD_TIMESTAMP", False),
            enable_sentence_timestamp=_env_bool("ENABLE_SENTENCE_TIMESTAMP", False),
            max_hotwords=_env_int("MAX_HOTWORDS", 100),
            max_hotword_length=_env_int("MAX_HOTWORD_LENGTH", 64),
            max_hotword_chars=_env_int("MAX_HOTWORD_CHARS", 1000),
        )
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            "GATEWAY_PORT": self.gateway_port,
            "AUDIO_CHUNK_SECONDS": self.chunk_seconds,
            "MAX_AUDIO_SECONDS": self.max_audio_seconds,
            "MAX_UPLOAD_MB": self.max_upload_mb,
            "MAX_JSON_BODY_MB": self.max_json_body_mb,
            "CHUNK_CONCURRENCY": self.chunk_concurrency,
            "LONG_CHUNKS_IN_FLIGHT": self.long_chunks_in_flight,
            "BACKEND_TIMEOUT": self.backend_timeout,
            "MAX_HOTWORDS": self.max_hotwords,
            "MAX_HOTWORD_LENGTH": self.max_hotword_length,
            "MAX_HOTWORD_CHARS": self.max_hotword_chars,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise RuntimeError("以下配置必须大于 0: " + ", ".join(invalid))
        if self.gateway_port > 65535:
            raise RuntimeError("GATEWAY_PORT 必须在 1～65535 之间")
        if (
            not self.service_version
            or not self.backend_url
            or not self.served_model_name
            or not self.gateway_host
        ):
            raise RuntimeError(
                "SERVICE_VERSION、BACKEND_URL、SERVED_MODEL_NAME 和 GATEWAY_HOST 不能为空"
            )
        unsupported = []
        if self.enable_vad:
            unsupported.append("ENABLE_VAD")
        if self.enable_word_timestamp:
            unsupported.append("ENABLE_WORD_TIMESTAMP")
        if self.enable_sentence_timestamp:
            unsupported.append("ENABLE_SENTENCE_TIMESTAMP")
        if unsupported:
            raise RuntimeError(
                "当前版本未实现以下可选引擎，不能开启或伪造结果: "
                + ", ".join(unsupported)
            )


settings = Settings.from_env()