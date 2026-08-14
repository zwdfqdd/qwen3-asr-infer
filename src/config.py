"""网关系统配置：集中读取环境变量并在启动前校验。"""

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    """读取名为 ``name`` 的整数环境变量；未设置时使用 ``default``。"""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"配置 {name} 必须是整数") from error


def _env_float(name: str, default: float) -> float:
    """读取名为 ``name`` 的浮点环境变量；未设置时使用 ``default``。"""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"配置 {name} 必须是数字") from error


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量；支持 true/false、1/0、yes/no、on/off。"""
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"配置 {name} 必须是 true 或 false")


@dataclass(frozen=True)
class Settings:
    """网关运行参数；模块导入时从环境变量读取一次，运行中不热更新。

    大小参数均使用 MiB（1 MiB = 1024² 字节），时间参数均使用秒。
    模型下载和 vLLM 引擎参数由 ``run.sh`` 管理，不属于本配置对象。
    字段后的注释依次说明环境变量名、用途和关键约束；默认值集中在
    :meth:`from_env`，部署入口可在启动前通过同名环境变量覆盖。
    """

    service_version: str  # SERVICE_VERSION：健康检查返回的服务版本。
    backend_url: str  # BACKEND_URL：网关转发到 vLLM 的 HTTP 基址。
    served_model_name: str  # SERVED_MODEL_NAME：网关缺省提交的 vLLM 模型名。
    gateway_host: str  # GATEWAY_HOST：网关监听地址。
    gateway_port: int  # GATEWAY_PORT：网关唯一对外端口，范围 1～65535。
    log_level: str  # LOG_LEVEL：网关日志级别，如 INFO、WARNING 或 ERROR。
    log_file_enabled: bool  # LOG_FILE_ENABLED：是否额外写入本地轮转文件。
    log_dir: str  # LOG_DIR：轮转文件日志目录；stdout 始终保留。
    log_max_file_mb: int  # LOG_MAX_FILE_MB：单个日志文件的硬大小上限（MiB）。
    log_backup_count: int  # LOG_BACKUP_COUNT：最多保留的轮转备份数。
    log_retention_days: int  # LOG_RETENTION_DAYS：启动时清理超过该天数的旧日志。
    log_queue_size: int  # LOG_QUEUE_SIZE：异步日志队列容量；满时丢弃并记录计数。

    chunk_seconds: float  # AUDIO_CHUNK_SECONDS：长音频单片最长秒数。
    max_audio_seconds: float  # MAX_AUDIO_SECONDS：单条原始音频最大时长。
    max_upload_mb: int  # MAX_UPLOAD_MB：Base64 解码后或 multipart file 的音频字节上限。
    max_json_body_mb: int  # MAX_JSON_BODY_MB：整个 Base64 JSON HTTP 请求体上限。
    # 字节上限按目标输入格式 16 kHz 单声道 PCM16 WAV 与 MAX_AUDIO_SECONDS 推导：
    # 7500 s × 16000 Hz × 1 ch × 2 B = 240,000,000 B，加 44 B WAV 头约 228.88 MiB → 229 MiB；
    # Base64 膨胀 4/3 约 305.18 MiB，再留 JSON 字段与热词开销 → 308 MiB。
    # 更高采样率、多声道或 24/32 位输入会在同样时长下超出该字节预算，属于预期拒绝。
    chunk_concurrency: int  # CHUNK_CONCURRENCY：单个长请求并行提交的最大分片数。
    long_chunks_in_flight: int  # LONG_CHUNKS_IN_FLIGHT：全局长音频分片在途上限。
    backend_timeout: float  # BACKEND_TIMEOUT：单个 vLLM 分片 HTTP 请求超时。
    backend_connection_limit: int  # BACKEND_CONNECTION_LIMIT：到 vLLM 的连接池上限。
    backend_keepalive_timeout: float  # BACKEND_KEEPALIVE_TIMEOUT：客户端空闲连接保留秒数。

    enable_hotword: bool  # ENABLE_HOTWORD：是否把动态热词写入 ASR Prompt。
    enable_vad: bool  # ENABLE_VAD：预留开关；当前开启会拒绝启动。
    enable_word_timestamp: bool  # ENABLE_WORD_TIMESTAMP：返回 Aligner 字/词级时间戳。
    enable_sentence_timestamp: bool  # ENABLE_SENTENCE_TIMESTAMP：按 ASR 标点聚合句级结果。

    aligner_model_dir: str  # ALIGNER_MODEL_DIR：ForcedAligner 本地权重目录。
    aligner_device: str  # ALIGNER_DEVICE：Aligner 设备，如 cuda:0、cuda:1 或 cpu。
    aligner_dtype: str  # ALIGNER_DTYPE：float16、bfloat16 或 float32。
    aligner_attention_backend: str  # ALIGNER_ATTENTION_BACKEND：auto/eager/sdpa/flash_attention_2。
    aligner_concurrency: int  # ALIGNER_CONCURRENCY：并发模型 worker 数；共享模型需验证线程安全。
    aligner_batch_size: int  # ALIGNER_BATCH_SIZE：单次模型调用跨请求合并的最大物理分片数。
    aligner_decode_workers: int  # ALIGNER_DECODE_WORKERS：每批 WAV 并行解码线程上限；1 为串行回滚。
    aligner_predecode_enabled: bool  # ALIGNER_PREDECODE_ENABLED：是否在 ASR 阶段有界预解码。
    aligner_predecode_max_mb: int  # ALIGNER_PREDECODE_MAX_MB：全局已解码 PCM 字节预算（MiB）。
    aligner_batch_wait_ms: float  # ALIGNER_BATCH_WAIT_MS：首条入队后的最大动态合批等待毫秒数。
    aligner_queue_size: int  # ALIGNER_QUEUE_SIZE：有界对齐分片队列容量。
    aligner_pipeline_enabled: bool  # ALIGNER_PIPELINE_ENABLED：分片 ASR 完成即提交对齐。

    max_hotwords: int  # MAX_HOTWORDS：去空、去重后的最大热词数量。
    max_hotword_length: int  # MAX_HOTWORD_LENGTH：单个热词最大 Unicode 字符数。
    max_hotword_chars: int  # MAX_HOTWORD_CHARS：所有热词合计最大 Unicode 字符数。

    @classmethod
    def from_env(cls) -> "Settings":
        result = cls(
            service_version=os.getenv("SERVICE_VERSION", "2.1.0").strip(),
            backend_url=os.getenv("BACKEND_URL", "http://127.0.0.1:8081").rstrip("/"),
            served_model_name=os.getenv("SERVED_MODEL_NAME", "qwen3-asr").strip(),
            gateway_host=os.getenv("GATEWAY_HOST", "0.0.0.0").strip(),
            gateway_port=_env_int("GATEWAY_PORT", 8080),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            log_file_enabled=_env_bool("LOG_FILE_ENABLED", True),
            log_dir=os.getenv("LOG_DIR", "logs").strip(),
            log_max_file_mb=_env_int("LOG_MAX_FILE_MB", 200),
            log_backup_count=_env_int("LOG_BACKUP_COUNT", 7),
            log_retention_days=_env_int("LOG_RETENTION_DAYS", 7),
            log_queue_size=_env_int("LOG_QUEUE_SIZE", 10000),
            chunk_seconds=_env_float("AUDIO_CHUNK_SECONDS", 32),
            # 以下三个默认值互相绑定，按 7500 秒 16 kHz 单声道 PCM16 WAV 推导，改一个必须重算其余两个。
            max_audio_seconds=_env_float("MAX_AUDIO_SECONDS", 7500),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 229),
            max_json_body_mb=_env_int("MAX_JSON_BODY_MB", 308),
            chunk_concurrency=_env_int("CHUNK_CONCURRENCY", 3),
            long_chunks_in_flight=_env_int("LONG_CHUNKS_IN_FLIGHT", 96),
            backend_timeout=_env_float("BACKEND_TIMEOUT", 300),
            backend_connection_limit=_env_int("BACKEND_CONNECTION_LIMIT", 100),
            backend_keepalive_timeout=_env_float("BACKEND_KEEPALIVE_TIMEOUT", 4),
            enable_hotword=_env_bool("ENABLE_HOTWORD", True),
            enable_vad=_env_bool("ENABLE_VAD", False),
            enable_word_timestamp=_env_bool("ENABLE_WORD_TIMESTAMP", True),
            enable_sentence_timestamp=_env_bool("ENABLE_SENTENCE_TIMESTAMP", True),
            aligner_model_dir=os.getenv(
                "ALIGNER_MODEL_DIR", "models/qwen3-forced-aligner-0.6b/pt"
            ).strip(),
            aligner_device=os.getenv("ALIGNER_DEVICE", "cuda:0").strip(),
            aligner_dtype=os.getenv("ALIGNER_DTYPE", "bfloat16").strip().lower(),
            aligner_attention_backend=os.getenv(
                "ALIGNER_ATTENTION_BACKEND", "auto"
            ).strip().lower(),
            aligner_concurrency=_env_int("ALIGNER_CONCURRENCY", 1),
            # batch=32 为目标 A10 实测最佳平衡：端到端 107.33→228.57 audio_s/s，显存峰值约
            # 20315 MiB；batch=48 收益不足 5% 已拒绝，batch=1 保留为逐片回滚值。
            aligner_batch_size=_env_int("ALIGNER_BATCH_SIZE", 32),
            aligner_decode_workers=_env_int("ALIGNER_DECODE_WORKERS", 1),
            aligner_predecode_enabled=_env_bool("ALIGNER_PREDECODE_ENABLED", False),
            aligner_predecode_max_mb=_env_int("ALIGNER_PREDECODE_MAX_MB", 128),
            aligner_batch_wait_ms=_env_float("ALIGNER_BATCH_WAIT_MS", 5),
            aligner_queue_size=_env_int("ALIGNER_QUEUE_SIZE", 256),
            # 默认关闭：单分片请求与批量模式等价，多分片长音频才体现重叠收益，
            # 需在目标机用真实多分片流量验证后再考虑改默认值。
            aligner_pipeline_enabled=_env_bool("ALIGNER_PIPELINE_ENABLED", False),
            max_hotwords=_env_int("MAX_HOTWORDS", 100),
            max_hotword_length=_env_int("MAX_HOTWORD_LENGTH", 64),
            max_hotword_chars=_env_int("MAX_HOTWORD_CHARS", 1000),
        )
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            "GATEWAY_PORT": self.gateway_port,
            "LOG_MAX_FILE_MB": self.log_max_file_mb,
            "LOG_BACKUP_COUNT": self.log_backup_count,
            "LOG_RETENTION_DAYS": self.log_retention_days,
            "LOG_QUEUE_SIZE": self.log_queue_size,
            "AUDIO_CHUNK_SECONDS": self.chunk_seconds,
            "MAX_AUDIO_SECONDS": self.max_audio_seconds,
            "MAX_UPLOAD_MB": self.max_upload_mb,
            "MAX_JSON_BODY_MB": self.max_json_body_mb,
            "CHUNK_CONCURRENCY": self.chunk_concurrency,
            "LONG_CHUNKS_IN_FLIGHT": self.long_chunks_in_flight,
            "BACKEND_TIMEOUT": self.backend_timeout,
            "BACKEND_CONNECTION_LIMIT": self.backend_connection_limit,
            "BACKEND_KEEPALIVE_TIMEOUT": self.backend_keepalive_timeout,
            "ALIGNER_CONCURRENCY": self.aligner_concurrency,
            "ALIGNER_BATCH_SIZE": self.aligner_batch_size,
            "ALIGNER_DECODE_WORKERS": self.aligner_decode_workers,
            "ALIGNER_PREDECODE_MAX_MB": self.aligner_predecode_max_mb,
            "ALIGNER_QUEUE_SIZE": self.aligner_queue_size,
            "MAX_HOTWORDS": self.max_hotwords,
            "MAX_HOTWORD_LENGTH": self.max_hotword_length,
            "MAX_HOTWORD_CHARS": self.max_hotword_chars,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise RuntimeError("以下配置必须大于 0: " + ", ".join(invalid))
        if not 0 <= self.aligner_batch_wait_ms <= 1000:
            raise RuntimeError("ALIGNER_BATCH_WAIT_MS 必须在 0～1000 毫秒之间")
        if self.aligner_decode_workers > 32:
            raise RuntimeError("ALIGNER_DECODE_WORKERS 不能大于 32")
        if self.aligner_queue_size < self.aligner_batch_size:
            raise RuntimeError("ALIGNER_QUEUE_SIZE 不能小于 ALIGNER_BATCH_SIZE")
        if self.gateway_port > 65535:
            raise RuntimeError("GATEWAY_PORT 必须在 1～65535 之间")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise RuntimeError("LOG_LEVEL 仅支持 DEBUG、INFO、WARNING、ERROR 或 CRITICAL")
        if self.log_file_enabled and not self.log_dir:
            raise RuntimeError("LOG_FILE_ENABLED=true 时 LOG_DIR 不能为空")
        if (
            not self.service_version
            or not self.backend_url
            or not self.served_model_name
            or not self.gateway_host
        ):
            raise RuntimeError(
                "SERVICE_VERSION、BACKEND_URL、SERVED_MODEL_NAME 和 GATEWAY_HOST 不能为空"
            )
        if self.enable_vad:
            raise RuntimeError("当前版本未实现 ENABLE_VAD，不能开启或伪造结果")
        if self.enable_word_timestamp or self.enable_sentence_timestamp:
            if not self.aligner_model_dir or not self.aligner_device:
                raise RuntimeError("启用时间戳时 ALIGNER_MODEL_DIR 和 ALIGNER_DEVICE 不能为空")
            if self.aligner_dtype not in {"float16", "bfloat16", "float32"}:
                raise RuntimeError(
                    "ALIGNER_DTYPE 仅支持 float16、bfloat16 或 float32"
                )
            if self.aligner_attention_backend not in {
                "auto", "eager", "sdpa", "flash_attention_2",
            }:
                raise RuntimeError(
                    "ALIGNER_ATTENTION_BACKEND 仅支持 auto、eager、sdpa 或 "
                    "flash_attention_2"
                )
            if self.aligner_device == "cpu" and self.aligner_dtype != "float32":
                raise RuntimeError("ALIGNER_DEVICE=cpu 时 ALIGNER_DTYPE 必须为 float32")

    @property
    def timestamp_enabled(self) -> bool:
        return self.enable_word_timestamp or self.enable_sentence_timestamp


settings = Settings.from_env()