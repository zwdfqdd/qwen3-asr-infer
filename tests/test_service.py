"""Qwen3-ASR 网关并发压测：延迟、吞吐、分片及资源统计。"""

import argparse
import asyncio
import json
import mimetypes
import os
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import aiohttp
import soundfile as sf

try:
    import psutil
except ImportError:  # 资源采样是可选能力，不应阻断接口压测。
    psutil = None


@dataclass
class Result:
    latency: float
    status: int | None
    chunks: int | None = None
    text_chars: int = 0
    error: str = ""


@dataclass
class ResourceStats:
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


async def _sample_resources(
    stats: ResourceStats, interval: float, gpu_index: int, stop: asyncio.Event
):
    nvidia_smi = shutil.which("nvidia-smi")
    if psutil is not None:
        psutil.cpu_percent(interval=None)
    while not stop.is_set():
        if psutil is not None:
            stats.cpu_utilization.append(psutil.cpu_percent(interval=None))
        if nvidia_smi:
            process = await asyncio.create_subprocess_exec(
                nvidia_smi,
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
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
            "article_url": args.article_url,
            "hotwords": args.hotword or None,
        }
        request_kwargs = {"json": request_data}
    else:
        form = aiohttp.FormData()
        form.add_field("file", audio, filename=filename, content_type=content_type)
        form.add_field("model", args.model)
        form.add_field("response_format", "json")
        form.add_field("temperature", "0")
        if args.language:
            form.add_field("language", args.language)
        request_kwargs = {"data": form}
    started = perf_counter()
    try:
        async with session.post(args.url, **request_kwargs) as response:
            body = await response.text()
            latency = perf_counter() - started
            header = response.headers.get("X-Audio-Chunks")
            try:
                chunks = int(header) if header is not None else None
            except ValueError:
                return Result(latency, response.status, error=f"无效X-Audio-Chunks: {header}")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return Result(latency, response.status, chunks=chunks,
                              error=f"无效JSON: {body[:300]}")
            if response.status != 200:
                detail = payload.get("error") or payload.get("message") or body[:300]
                return Result(latency, response.status, chunks=chunks, error=str(detail))
            text_field = "istar_asr" if args.api_mode == "chinese-asr" else "text"
            text = payload.get(text_field)
            if not isinstance(text, str):
                return Result(latency, response.status, chunks=chunks,
                              error=f"响应缺少{text_field}字段")
            if args.api_mode == "chinese-asr" and payload.get("code") != 0:
                return Result(latency, response.status, chunks=chunks,
                              error=f"业务码异常: {payload.get('code')}")
            return Result(latency, response.status, chunks, len(text))
    except Exception as error:  # noqa: BLE001
        return Result(perf_counter() - started, None,
                      error=f"{type(error).__name__}: {error}")


async def _check_health(session: aiohttp.ClientSession, health_url: str):
    async with session.get(health_url) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"服务健康检查失败（HTTP {response.status}）: {body[:300]}")


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


def _print_report(args, duration, expected_chunks, elapsed, results, resources):
    successes = [result for result in results if result.status == 200 and not result.error]
    failures = [result for result in results if result.status != 200 or result.error]
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
        print(
            f"成功请求延迟: 平均 {statistics.fmean(latencies):.1f}ms  "
            f"P50 {_percentile(latencies, 0.50):.1f}ms  "
            f"P90 {_percentile(latencies, 0.90):.1f}ms  "
            f"P95 {_percentile(latencies, 0.95):.1f}ms  "
            f"P99 {_percentile(latencies, 0.99):.1f}ms"
        )
    chunk_counts = Counter(result.chunks for result in successes if result.chunks is not None)
    if chunk_counts:
        print("网关分片数: " + ", ".join(
            f"{count}片={amount}请求" for count, amount in sorted(chunk_counts.items())
        ))
    missing_headers = sum(result.chunks is None for result in successes)
    if missing_headers:
        print(f"警告: {missing_headers}个成功响应缺少X-Audio-Chunks，可能绕过了网关")
    mismatches = sum(
        result.chunks is not None and result.chunks != expected_chunks
        for result in successes
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


async def _run(args) -> int:
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频不存在: {audio_path}")
    duration, expected_chunks = _audio_info(audio_path, args.chunk_seconds)
    audio = audio_path.read_bytes()
    content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _check_health(session, args.health_url)
    await _warmup(args, audio, audio_path.name, content_type)
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
    _print_report(args, duration, expected_chunks, elapsed, results, resources)
    return 0 if not any(result.status != 200 or result.error for result in results) else 1


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR网关并发性能测试")
    parser.add_argument("--audio", required=True, help="本地音频文件")
    parser.add_argument("--api-mode", choices=("native", "chinese-asr"), default="native")
    parser.add_argument("--hotword", action="append", default=[], help="兼容接口动态热词，可重复")
    parser.add_argument("--article-url", default=None, help="兼容接口原样返回的来源标识")
    parser.add_argument("--url", default=None)
    parser.add_argument("--health-url", default="http://127.0.0.1:8080/health")
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "qwen3-asr"))
    parser.add_argument("--language", default="")
    parser.add_argument("--concurrency", type=int, default=96)
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--chunk-seconds", type=float, default=30,
        help="仅用于计算网关预期分片数，需与AUDIO_CHUNK_SECONDS一致",
    )
    args = parser.parse_args()
    if args.url is None:
        endpoint = "/chinese_asr" if args.api_mode == "chinese-asr" else "/v1/audio/transcriptions"
        args.url = f"http://127.0.0.1:8080{endpoint}"
    if args.concurrency < 1 or args.total < 1:
        parser.error("--concurrency和--total必须大于0")
    if args.warmup < 0 or args.timeout <= 0 or args.sample_interval <= 0:
        parser.error("--warmup不能小于0，--timeout和--sample-interval必须大于0")
    if args.gpu_index < 0 or args.chunk_seconds <= 0:
        parser.error("--gpu-index不能小于0，--chunk-seconds必须大于0")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
