"""单条调用 /chinese_asr，校验兼容响应及段级时间戳。

运行命令：
    python tests/test_chinese_asr_single.py \
      --audio test_data/audio_16000_10s.wav \
      --article-url source-001 \
      --hotword 通义千问 \
      --hotword 水滴筹

仅测试基础识别：
    python tests/test_chinese_asr_single.py \
      --audio test_data/audio_16000_10s.wav

指定服务地址：
    python tests/test_chinese_asr_single.py \
      --audio test_data/audio_16000_10s.wav \
      --url http://127.0.0.1:8080/chinese_asr
"""

import argparse
import asyncio
import base64
import json
from pathlib import Path

import aiohttp


def _validate_response(body: dict, article_url: str | None) -> None:
    if body.get("code") != 0:
        raise RuntimeError(
            f"业务失败: code={body.get('code')}, error={body.get('error')}, "
            f"message={body.get('message')}"
        )
    if body.get("article_url") != article_url:
        raise RuntimeError("响应 article_url 与请求不一致")
    if not isinstance(body.get("istar_asr"), str):
        raise RuntimeError("响应缺少字符串 istar_asr")
    segments = body.get("asr")
    if not isinstance(segments, list):
        raise RuntimeError("响应 asr 不是数组")
    previous_end = 0.0
    for index, segment in enumerate(segments):
        if segment.get("idx") != index:
            raise RuntimeError(f"asr[{index}].idx 不连续")
        timestamp = segment.get("timestamp")
        if not isinstance(timestamp, list) or len(timestamp) != 2:
            raise RuntimeError(f"asr[{index}].timestamp 格式错误")
        start, end = timestamp
        if start < previous_end or end < start:
            raise RuntimeError(f"asr[{index}].timestamp 非单调")
        if not isinstance(segment.get("text"), str):
            raise RuntimeError(f"asr[{index}].text 不是字符串")
        if segment.get("words") != []:
            raise RuntimeError(f"asr[{index}].words 当前应为空数组")
        previous_end = end


async def _run(args) -> int:
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频不存在: {audio_path}")
    request_body = {
        "base64": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
        "article_url": args.article_url,
        "hotwords": args.hotword or None,
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    started = asyncio.get_running_loop().time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.url, json=request_body) as response:
            raw = await response.text()
            elapsed = asyncio.get_running_loop().time() - started
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"服务返回非JSON: HTTP {response.status}: {raw[:500]}") from error
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: code={body.get('code')}, "
                    f"error={body.get('error')}, message={body.get('message')}"
                )
            _validate_response(body, args.article_url)
            raw_chunks = response.headers.get("X-Audio-Chunks")
            try:
                chunks = int(raw_chunks or "")
            except ValueError as error:
                raise RuntimeError("响应缺少有效 X-Audio-Chunks") from error
            if chunks < 1 or chunks != len(body["asr"]):
                raise RuntimeError("分片响应头必须为正整数且与 asr[] 数量一致")
    print(f"HTTP 200  耗时: {elapsed:.3f}s  分片数: {chunks}")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="单条测试 /chinese_asr Base64 JSON接口")
    parser.add_argument("--audio", required=True, help="本地音频文件")
    parser.add_argument("--url", default="http://127.0.0.1:8080/chinese_asr")
    parser.add_argument("--article-url", default=None, help="仅作为来源标识原样返回")
    parser.add_argument("--hotword", action="append", default=[], help="动态热词，可重复传入")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout必须大于0")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
