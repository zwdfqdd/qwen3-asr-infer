"""Qwen3-ASR 网关并发压测：延迟、吞吐、阶段耗时、指标及资源统计。"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import aiohttp
import soundfile as sf

try:
    import psutil
except ImportError:  # 资源采样是可选能力，不应阻断接口压测。
    psutil = None


_STAGE_NAMES = (
    "request-parse", "base64-decode", "audio-split",
    "chunk-local-queue", "chunk-local-queue-max",
    "chunk-global-queue", "chunk-global-queue-max",
    "asr-backend", "asr", "aligner-queue", "aligner",
    "postprocess", "serialize", "total",
)
_TIMING_ITEM_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_-]*);dur=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
_PROM_SAMPLE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$"
)
_PROM_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
_ENV_ALLOWLIST = (
    "SERVICE_VERSION", "AUDIO_CHUNK_SECONDS", "CHUNK_CONCURRENCY",
    "LONG_CHUNKS_IN_FLIGHT", "BACKEND_TIMEOUT", "BACKEND_CONNECTION_LIMIT",
    "BACKEND_KEEPALIVE_TIMEOUT", "DTYPE", "VLLM_LOAD_FORMAT",
    "VLLM_MAX_MODEL_LEN", "VLLM_MAX_NUM_SEQS",
    "VLLM_MAX_NUM_BATCHED_TOKENS", "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_ATTENTION_BACKEND_NAME", "ALIGNER_DEVICE", "ALIGNER_DTYPE",
    "ALIGNER_ATTENTION_BACKEND", "ALIGNER_CONCURRENCY", "ALIGNER_BATCH_SIZE",
    "ALIGNER_DECODE_WORKERS", "ALIGNER_PREDECODE_ENABLED",
    "ALIGNER_PREDECODE_MAX_MB", "ALIGNER_BATCH_WAIT_MS",
    "ALIGNER_QUEUE_SIZE", "ENABLE_WORD_TIMESTAMP", "ENABLE_SENTENCE_TIMESTAMP",
)

@dataclass
class Result:
    """单次结果：latency 单位秒，timings 中各阶段单位毫秒。"""

    latency: float
    status: int | None
    chunks: int | None = None
    text_chars: int = 0
    error: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    timing_error: str = ""
    timing_required_failure: bool = False
    request_id: str = ""


@dataclass
class ResourceStats:
    """周期采样值：利用率单位百分比，GPU 显存单位 MiB。"""

    gpu_utilization: list[float] = field(default_factory=list)
    gpu_memory_mib: list[float] = field(default_factory=list)
    cpu_utilization: list[float] = field(default_factory=list)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | int]:
    """生成稳定的统计字段；调用者负责传入相同单位的数据。"""
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "min": min(values), "max": max(values),
        "p50": _percentile(values, 0.50), "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95), "p99": _percentile(values, 0.99),
    }

def _parse_server_timing(header: str | None) -> tuple[dict[str, float], str]:
    """解析网关固定阶段；任一项缺失、重复或非法时整头视为无效。"""
    if header is None or not header.strip():
        return {}, "缺少Server-Timing"
    timings: dict[str, float] = {}
    for raw_item in header.split(","):
        item = raw_item.strip()
        match = _TIMING_ITEM_RE.fullmatch(item)
        if not match:
            return {}, f"无效Server-Timing项: {item[:100]}"
        name, raw_value = match.groups()
        if name not in _STAGE_NAMES:
            return {}, f"未知Server-Timing阶段: {name}"
        if name in timings:
            return {}, f"重复Server-Timing阶段: {name}"
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            return {}, f"非法Server-Timing耗时: {name}={raw_value}"
        timings[name] = value
    missing = [name for name in _STAGE_NAMES if name not in timings]
    if missing:
        return {}, "Server-Timing缺少阶段: " + ",".join(missing)
    return timings, ""


def _unescape_prometheus_label(value: str) -> str:
    return value.replace(r"\n", "\n").replace(r'\"', '"').replace(r"\\", "\\")


def _parse_prometheus_labels(raw: str | None) -> dict[str, str] | None:
    if raw is None or raw == "":
        return {}
    labels: dict[str, str] = {}
    position = 0
    while position < len(raw):
        match = _PROM_LABEL_RE.match(raw, position)
        if match is None:
            return None
        name, value = match.groups()
        if name in labels:
            return None
        labels[name] = _unescape_prometheus_label(value)
        position = match.end()
        if position == len(raw):
            break
        if raw[position] != ",":
            return None
        position += 1
    return labels

def _parse_prometheus(text: str) -> list[dict[str, object]]:
    """仅保留可安全写入 JSON 的 vLLM 非直方图桶样本。"""
    samples: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_SAMPLE_RE.fullmatch(line)
        if match is None:
            continue
        name, raw_labels, raw_value = match.groups()
        if not (name.startswith("vllm:") or name.startswith("vllm_")):
            continue
        if name.endswith("_bucket"):
            continue
        labels = _parse_prometheus_labels(raw_labels)
        if labels is None:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        samples.append({"sample": {"name": name, "labels": labels}, "value": value})
    return samples


def _audio_info(path: Path, chunk_seconds: float) -> tuple[float, int]:
    with sf.SoundFile(path) as audio:
        if audio.samplerate <= 0 or audio.frames <= 0:
            raise ValueError("音频为空或采样率无效")
        frames_per_chunk = int(audio.samplerate * chunk_seconds)
        if frames_per_chunk <= 0:
            raise ValueError("--chunk-seconds必须大于0")
        duration = audio.frames / audio.samplerate
        chunks = (audio.frames + frames_per_chunk - 1) // frames_per_chunk
        return duration, chunks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _git_info(repo: Path) -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments], capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()

    try:
        tracked_status = run("status", "--porcelain", "--untracked-files=no")
        all_status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(tracked_status),
            "untracked_file_count": sum(
                line.startswith("??") for line in all_status.splitlines()
            ),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"commit": None, "branch": None, "dirty": None,
                "error": f"{type(error).__name__}: {error}"}


def _runtime_info() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in (
        "vllm", "transformers", "qwen-asr", "aiohttp", "soundfile", "psutil"
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "packages": packages,
    }


def _resource_summary(resources: ResourceStats, gpu_index: int) -> dict[str, object]:
    return {
        "gpu": {
            "index": gpu_index,
            "utilization_percent": _summary(resources.gpu_utilization),
            "memory_mib": _summary(resources.gpu_memory_mib),
        },
        "cpu": {"utilization_percent": _summary(resources.cpu_utilization)},
    }

async def _sample_resources(
    stats: ResourceStats, interval: float, gpu_index: int, stop: asyncio.Event
):
    nvidia_smi = shutil.which("nvidia-smi")
    if psutil is not None:
        psutil.cpu_percent(interval=None)
    while not stop.is_set():
        if psutil is not None:
            cpu = float(psutil.cpu_percent(interval=None))
            if math.isfinite(cpu):
                stats.cpu_utilization.append(cpu)
        if nvidia_smi:
            process = await asyncio.create_subprocess_exec(
                nvidia_smi, f"--id={gpu_index}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                for line in stdout.decode(errors="replace").splitlines():
                    try:
                        utilization, memory = (
                            float(value.strip()) for value in line.split(",")[:2]
                        )
                    except (ValueError, IndexError):
                        continue
                    if math.isfinite(utilization) and math.isfinite(memory):
                        stats.gpu_utilization.append(utilization)
                        stats.gpu_memory_mib.append(memory)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def _request(session, args, audio: bytes, filename: str, content_type: str) -> Result:
    if args.api_mode == "chinese-asr":
        import base64
        request_data = {
            "base64": base64.b64encode(audio).decode("ascii"),
            "article_url": args.article_url, "hotwords": args.hotword or None,
        }
        if args.language:
            request_data["language"] = args.language
        request_kwargs = {"json": request_data}
    else:
        form = aiohttp.FormData()
        form.add_field("file", audio, filename=filename, content_type=content_type)
        form.add_field("model", args.model)
        form.add_field("response_format", "json")
        if args.language:
            form.add_field("language", args.language)
        request_kwargs = {"data": form}

    started = perf_counter()
    try:
        async with session.post(args.url, **request_kwargs) as response:
            body = await response.text()
            latency = perf_counter() - started
            raw_request_id = response.headers.get("X-Request-ID", "").strip()
            request_id = (
                raw_request_id
                if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", raw_request_id)
                else ""
            )
            timings, timing_error = _parse_server_timing(
                response.headers.get("Server-Timing")
            )
            chunk_header = response.headers.get("X-Audio-Chunks")
            try:
                chunks = int(chunk_header) if chunk_header is not None else None
            except ValueError:
                return Result(latency, response.status, error=f"无效X-Audio-Chunks: {chunk_header}",
                              timings=timings, timing_error=timing_error,
                              request_id=request_id)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return Result(latency, response.status, chunks=chunks,
                              error=f"无效JSON: {body[:300]}", timings=timings,
                              timing_error=timing_error, request_id=request_id)
            if response.status != 200:
                detail = payload.get("error") or payload.get("message") or body[:300]
                return Result(
                    latency, response.status, chunks=chunks, error=str(detail),
                    timings=timings, timing_error=timing_error, request_id=request_id,
                )
            text_field = "istar_asr" if args.api_mode == "chinese-asr" else "text"
            text = payload.get(text_field)
            if not isinstance(text, str):
                return Result(latency, response.status, chunks=chunks,
                              error=f"响应缺少{text_field}字段", timings=timings,
                              timing_error=timing_error, request_id=request_id)
            if args.api_mode == "chinese-asr" and payload.get("code") != 0:
                return Result(latency, response.status, chunks=chunks,
                              error=f"业务码异常: {payload.get('code')}", timings=timings,
                              timing_error=timing_error, request_id=request_id)
            if timing_error and args.require_server_timing:
                return Result(
                    latency, response.status, chunks, len(text),
                    error=f"Server-Timing要求未满足: {timing_error}", timings=timings,
                    timing_error=timing_error, timing_required_failure=True,
                    request_id=request_id,
                )
            return Result(latency, response.status, chunks, len(text),
                          timings=timings, timing_error=timing_error)
    except Exception as error:  # noqa: BLE001
        return Result(perf_counter() - started, None,
                      error=f"{type(error).__name__}: {error}")

async def _check_health(session: aiohttp.ClientSession, health_url: str) -> dict[str, object]:
    async with session.get(health_url) as response:
        body = await response.text()
        record: dict[str, object] = {"url": health_url, "status": response.status}
        try:
            record["json"] = json.loads(
                body,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"不允许的JSON常量: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError):
            record["raw"] = body[:500]
        if response.status != 200:
            raise RuntimeError(f"服务健康检查失败（HTTP {response.status}）: {body[:300]}")
        return record


async def _fetch_metrics(
    session: aiohttp.ClientSession, metrics_url: str, skip: bool
) -> dict[str, object]:
    record: dict[str, object] = {"url": metrics_url, "samples": []}
    if skip:
        record["skipped"] = True
        return record
    try:
        async with session.get(metrics_url) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {body[:200]}")
            record["samples"] = _parse_prometheus(body)
    except Exception as error:  # noqa: BLE001 - 指标旁路失败不得阻断压测。
        record["error"] = f"{type(error).__name__}: {error}"
    return record


async def _execute_requests(args, audio: bytes, filename: str, content_type: str):
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def run_one() -> Result:
            async with semaphore:
                return await _request(session, args, audio, filename, content_type)

        return await asyncio.gather(*(run_one() for _ in range(args.total)))

async def _warmup(args, audio: bytes, filename: str, content_type: str):
    if args.warmup <= 0:
        return
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(args.warmup):
            result = await _request(session, args, audio, filename, content_type)
            if result.status != 200 or result.error:
                raise RuntimeError(f"预热失败: HTTP={result.status}, {result.error}")


def _successful(results: list[Result]) -> list[Result]:
    return [result for result in results if result.status == 200 and not result.error]


def _stage_summaries(results: list[Result]) -> dict[str, dict[str, float | int]]:
    return {
        name: _summary([result.timings[name] for result in results if name in result.timings])
        for name in _STAGE_NAMES
    }


def _print_report(args, duration, expected_chunks, elapsed, results, resources):
    successes = _successful(results)
    failures = [result for result in results if result not in successes]
    latencies = [result.latency * 1000 for result in successes]
    print("推理服务性能测试报告")
    print("=" * 60)
    print(
        f"音频时长: {duration:.3f}s  预期分片: {expected_chunks}  "
        f"总请求: {len(results)}  成功: {len(successes)}  "
        f"失败: {len(failures)}  并发: {args.concurrency}"
    )
    qps = len(successes) / elapsed if elapsed else 0.0
    throughput = len(successes) * duration / elapsed if elapsed else 0.0
    backend_requests = sum(result.chunks or 1 for result in successes)
    backend_rps = backend_requests / elapsed if elapsed else 0.0
    print(
        f"总耗时: {elapsed:.2f}s  原始请求QPS: {qps:.2f}  "
        f"后端分片RPS: {backend_rps:.2f}  吞吐: {throughput:.2f} audio_s/s"
    )
    if latencies:
        summary = _summary(latencies)
        print(
            f"成功请求延迟: 平均 {summary['mean']:.1f}ms  "
            f"P50 {summary['p50']:.1f}ms  P90 {summary['p90']:.1f}ms  "
            f"P95 {summary['p95']:.1f}ms  P99 {summary['p99']:.1f}ms"
        )

    stage_summaries = _stage_summaries(successes)
    if any(summary["count"] for summary in stage_summaries.values()):
        print("Server-Timing阶段（毫秒）:")
        for name, summary in stage_summaries.items():
            if summary["count"]:
                print(
                    f"  {name}: count={summary['count']}  平均={summary['mean']:.2f}  "
                    f"P95={summary['p95']:.2f}  P99={summary['p99']:.2f}"
                )
        print("  注: chunk-local/global-queue与asr-backend为每请求各分片累计值；asr/total为墙钟耗时。")
    timing_candidates = [
        result for result in results
        if result.status == 200 and (not result.error or result.timing_required_failure)
    ]
    valid_timings = sum(not result.timing_error for result in timing_candidates)
    coverage = valid_timings / len(timing_candidates) if timing_candidates else 0.0
    print(
        f"Server-Timing覆盖率: {valid_timings}/{len(timing_candidates)} "
        f"({coverage:.1%})"
    )
    timing_issues = Counter(
        result.timing_error for result in timing_candidates if result.timing_error
    )
    for issue, count in timing_issues.most_common(5):
        print(f"警告: {count}个成功响应{issue}")
    chunk_counts = Counter(result.chunks for result in successes if result.chunks is not None)
    if chunk_counts:
        print("网关分片数: " + ", ".join(
            f"{count}片={amount}请求" for count, amount in sorted(chunk_counts.items())
        ))
    missing_headers = sum(result.chunks is None for result in successes)
    if missing_headers:
        print(f"警告: {missing_headers}个成功响应缺少X-Audio-Chunks，可能绕过了网关")
    mismatches = sum(
        result.chunks is not None and result.chunks != expected_chunks for result in successes
    )
    if mismatches:
        print(f"警告: {mismatches}个响应的分片数与预期{expected_chunks}不一致")
    if successes:
        print(f"返回文本长度: 平均 {statistics.fmean(r.text_chars for r in successes):.1f}字符")

    if failures:
        reasons = Counter(
            f"HTTP {result.status}: {result.error}" if result.status else result.error
            for result in failures
        )
        print("失败分类:")
        for reason, count in reasons.most_common(10):
            print(f"  {count} × {reason}")
        failure_request_ids = [result.request_id for result in failures if result.request_id]
        if failure_request_ids:
            print("失败请求ID（最多20个）: " + ", ".join(failure_request_ids[:20]))
    if resources.gpu_utilization:
        print(
            f"GPU {args.gpu_index}: 平均 {statistics.fmean(resources.gpu_utilization):.1f}%  "
            f"峰值 {max(resources.gpu_utilization):.0f}%  "
            f"显存峰值 {max(resources.gpu_memory_mib):.0f} MiB"
        )
    else:
        print(f"GPU {args.gpu_index}: 未采集到nvidia-smi数据")
    if resources.cpu_utilization:
        print(
            f"CPU: 平均 {statistics.fmean(resources.cpu_utilization):.1f}%  "
            f"峰值 {max(resources.cpu_utilization):.1f}%"
        )
    else:
        print("CPU: 未采集（未安装psutil或无可用数据）")


def _result_report(results: list[Result], elapsed: float, audio_duration: float) -> dict[str, object]:
    successes = _successful(results)
    failures = [result for result in results if result not in successes]
    timing_candidates = [
        result for result in results
        if result.status == 200 and (not result.error or result.timing_required_failure)
    ]
    valid_timings = sum(not result.timing_error for result in timing_candidates)
    status_counts = Counter("network_error" if r.status is None else str(r.status) for r in results)
    error_counts = Counter(r.error for r in failures)
    chunks = sum(result.chunks or 1 for result in successes)
    return {
        "total": len(results), "success": len(successes), "failure": len(failures),
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "failure_samples": [
            {
                "status": result.status,
                "error": result.error,
                "request_id": result.request_id or None,
                "latency_ms": result.latency * 1000,
                "chunks": result.chunks,
            }
            for result in failures[:20]
        ],
        "elapsed_seconds": elapsed,
        "qps": len(successes) / elapsed if elapsed else 0.0,
        "chunk_rps": chunks / elapsed if elapsed else 0.0,
        "audio_s_per_s": len(successes) * audio_duration / elapsed if elapsed else 0.0,
        "end_to_end_ms": _summary([result.latency * 1000 for result in successes]),

        "stage_summaries_ms": _stage_summaries(successes),
        "stage_semantics": {
            "chunk-local-queue": "每请求各分片累计值",
            "chunk-local-queue-max": "每请求内单分片最大等待值",
            "chunk-global-queue": "每请求各分片累计值",
            "chunk-global-queue-max": "每请求内单分片最大等待值",
            "asr-backend": "每请求各分片aiohttp后端请求累计值",
            "asr": "墙钟耗时", "aligner-queue": "墙钟耗时",
            "aligner": "墙钟耗时", "total": "墙钟耗时",
        },
        "server_timing": {
            "eligible_success_responses": len(timing_candidates),
            "valid_responses": valid_timings,
            "coverage": valid_timings / len(timing_candidates) if timing_candidates else 0.0,
            "errors": dict(Counter(
                result.timing_error for result in timing_candidates if result.timing_error
            )),
        },
    }


def _atomic_write_json(path: Path, report: dict[str, object]) -> None:
    """同目录落临时文件并替换，避免留下半份实验报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _build_report(args, audio_path, audio_sha256, duration, expected_chunks,
                  elapsed, health, metrics_before, metrics_after, results, resources):
    parameters = {
        "api_mode": args.api_mode, "url": args.url, "health_url": args.health_url,
        "metrics_url": args.metrics_url, "skip_metrics": args.skip_metrics,
        "model": args.model, "language": args.language,
        "hotword_count": len(args.hotword), "article_url_set": args.article_url is not None,
        "concurrency": args.concurrency, "total": args.total, "warmup": args.warmup,
        "timeout_seconds": args.timeout, "sample_interval_seconds": args.sample_interval,
        "gpu_index": args.gpu_index, "chunk_seconds": args.chunk_seconds,
        "require_server_timing": args.require_server_timing,
    }

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git": _git_info(Path(__file__).resolve().parents[1]),
        "runtime": _runtime_info(),
        "environment": {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ},
        "health": health,
        "input": {
            "filename": audio_path.name, "bytes": audio_path.stat().st_size,
            "sha256": audio_sha256, "duration_seconds": duration,
            "expected_chunks": expected_chunks,
        },
        "parameters": parameters,
        "results": _result_report(results, elapsed, duration),
        "metrics": {"before": metrics_before, "after": metrics_after},
        "resources": _resource_summary(resources, args.gpu_index),
    }


async def _run(args) -> int:
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频不存在: {audio_path}")
    duration, expected_chunks = _audio_info(audio_path, args.chunk_seconds)
    audio_sha256 = _sha256(audio_path)
    audio = audio_path.read_bytes()
    content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        health = await _check_health(session, args.health_url)
    await _warmup(args, audio, audio_path.name, content_type)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        metrics_before = await _fetch_metrics(session, args.metrics_url, args.skip_metrics)
    resources = ResourceStats()
    stop = asyncio.Event()
    sampler = asyncio.create_task(
        _sample_resources(resources, args.sample_interval, args.gpu_index, stop)
    )
    started = perf_counter()
    try:
        results = await _execute_requests(args, audio, audio_path.name, content_type)
    finally:
        elapsed = perf_counter() - started
        stop.set()
        await sampler

    async with aiohttp.ClientSession(timeout=timeout) as session:
        metrics_after = await _fetch_metrics(session, args.metrics_url, args.skip_metrics)
    _print_report(args, duration, expected_chunks, elapsed, results, resources)
    if args.output_json:
        report = _build_report(
            args, audio_path, audio_sha256, duration, expected_chunks, elapsed,
            health, metrics_before, metrics_after, results, resources,
        )
        _atomic_write_json(Path(args.output_json), report)
        print(f"JSON报告已写入: {args.output_json}")
    return 0 if not any(result.status != 200 or result.error for result in results) else 1


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR网关并发性能测试")
    parser.add_argument("--audio", required=True, help="每个请求重复使用的本地音频文件路径")
    parser.add_argument(
        "--api-mode", choices=("native", "chinese-asr"), default="native",
        help="接口模式：native=multipart，chinese-asr=Base64 JSON",
    )
    parser.add_argument("--hotword", action="append", default=[], help="兼容接口动态热词，可重复")
    parser.add_argument("--article-url", default=None, help="兼容接口原样返回的来源标识")
    parser.add_argument("--url", default=None, help="压测地址；缺省时根据 --api-mode 自动选择")
    parser.add_argument(
        "--health-url", default="http://127.0.0.1:8080/health",
        help="压测前检查的健康端点地址",
    )
    parser.add_argument(
        "--metrics-url", default="http://127.0.0.1:8080/metrics",
        help="预热后、正式压测前后抓取的Prometheus指标地址",
    )
    parser.add_argument("--skip-metrics", action="store_true", help="跳过压测前后的指标抓取")
    parser.add_argument(
        "--require-server-timing", action="store_true",
        help="成功响应缺少或包含非法Server-Timing时判定测试失败",
    )
    parser.add_argument(
        "--output-json", metavar="PATH",
        help="原子写入机器可读JSON报告；路径必须以.json结尾",
    )

    parser.add_argument(
        "--model", default=os.getenv("SERVED_MODEL_NAME", "qwen3-asr"),
        help="multipart 请求模型名；默认读取 SERVED_MODEL_NAME",
    )
    parser.add_argument("--language", default="", help="可选强制语种；留空时由模型自动检测")
    parser.add_argument("--concurrency", type=int, default=96, help="客户端最大并发请求数")
    parser.add_argument("--total", type=int, default=2000, help="正式压测请求总数")
    parser.add_argument("--warmup", type=int, default=3, help="正式计时前串行预热请求数")
    parser.add_argument("--timeout", type=float, default=300, help="每个 HTTP 请求总超时，单位秒")
    parser.add_argument("--sample-interval", type=float, default=0.5, help="资源采样间隔，单位秒")
    parser.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi 采样的宿主 GPU 索引")
    parser.add_argument(
        "--chunk-seconds", type=float, default=32,
        help="网关单片秒数，仅用于计算预期分片数；需与 AUDIO_CHUNK_SECONDS 一致",
    )
    args = parser.parse_args()
    if args.url is None:
        endpoint = "/chinese_asr" if args.api_mode == "chinese-asr" else "/v1/audio/transcriptions"
        args.url = f"http://127.0.0.1:8080{endpoint}"
    if args.concurrency < 1 or args.total < 1:
        parser.error("--concurrency和--total必须大于0")
    if args.warmup < 0:
        parser.error("--warmup不能小于0")
    positive_floats = (args.timeout, args.sample_interval, args.chunk_seconds)
    if any(not math.isfinite(value) or value <= 0 for value in positive_floats):
        parser.error("--timeout、--sample-interval和--chunk-seconds必须为有限正数")
    if args.gpu_index < 0:
        parser.error("--gpu-index不能小于0")
    if args.output_json and not args.output_json.endswith(".json"):
        parser.error("--output-json路径必须以.json结尾")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
