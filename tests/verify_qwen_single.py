"""通过 vLLM 原生转写 API 执行推理和可选 CER 验证。"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import unicodedata
from pathlib import Path

import aiohttp

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac")


def _normalize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "")
    return [char for char in normalized if not char.isspace()]


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    previous = list(range(len(hyp) + 1))
    for row, ref_char in enumerate(ref, 1):
        current = [row]
        for column, hyp_char in enumerate(hyp, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (ref_char != hyp_char),
            ))
        previous = current
    return previous[-1]


def _cer(ref: str, hyp: str) -> tuple[int, int]:
    normalized_ref = _normalize(ref)
    return _edit_distance(normalized_ref, _normalize(hyp)), len(normalized_ref)


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _is_audio(value: str) -> bool:
    return value.lower().endswith(_AUDIO_EXTS)


def _collect_audio(input_path: str) -> list[str]:
    if _is_url(input_path) or (os.path.isfile(input_path) and _is_audio(input_path)):
        return [input_path]
    if os.path.isdir(input_path):
        audios = sorted(str(path) for path in Path(input_path).iterdir()
                        if path.is_file() and _is_audio(path.name))
        if audios:
            return audios
        raise RuntimeError(f"{input_path} 下未找到音频文件")
    raise FileNotFoundError(f"无效 --input: {input_path}")


def _load_refs(
    ref: str | None,
    ref_text: str | None,
    audios: list[str],
) -> dict[str, str]:
    if ref and ref_text is not None:
        raise ValueError("--ref 与 --ref-text 不能同时使用")
    if ref_text is not None:
        if len(audios) != 1:
            raise ValueError("--ref-text 只适用于单个输入音频")
        return {audios[0]: ref_text}
    if not ref:
        return {}
    ref_path = Path(ref)
    if not ref_path.exists():
        raise FileNotFoundError(f"参考路径不存在: {ref}")
    if ref_path.is_dir():
        result = {}
        for audio in audios:
            if _is_url(audio):
                continue
            text_path = ref_path / f"{Path(audio).stem}.txt"
            if text_path.is_file():
                result[audio] = text_path.read_text(encoding="utf-8").strip()
        return result
    if ref_path.suffix.lower() == ".txt":
        if len(audios) != 1:
            raise ValueError("单个 .txt 参考文件只适用于单个输入音频")
        return {audios[0]: ref_path.read_text(encoding="utf-8").strip()}
    if ref_path.suffix.lower() != ".jsonl":
        raise ValueError("--ref 仅支持目录、.txt 或 .jsonl；直接文本请用 --ref-text")
    result = {}
    base = ref_path.parent
    with ref_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("wav"), str) or not isinstance(item.get("text"), str):
                raise ValueError(f"JSONL 第 {line_number} 行必须包含字符串 wav/text")
            wav = item["wav"]
            if _is_url(wav):
                for audio in audios:
                    if audio == wav:
                        result[audio] = item["text"]
                continue
            expected = Path(wav)
            expected = expected if expected.is_absolute() else base / expected
            for audio in audios:
                if not _is_url(audio) and Path(audio).resolve() == expected.resolve():
                    result[audio] = item["text"]
    return result


def _validate_refs(audios: list[str], refs: dict[str, str]) -> None:
    missing = [audio for audio in audios if audio not in refs]
    empty = [audio for audio in audios if audio in refs and not _normalize(refs[audio])]
    if missing or empty:
        details = []
        if missing:
            details.append("缺少参考: " + ", ".join(missing[:10]))
        if empty:
            details.append("空参考: " + ", ".join(empty[:10]))
        raise ValueError("；".join(details))


async def _load_audio(session, source: str) -> tuple[bytes, str, str]:
    if _is_url(source):
        async with session.get(source) as response:
            response.raise_for_status()
            return (
                await response.read(),
                Path(response.url.path).name or "audio.wav",
                response.headers.get("Content-Type", "application/octet-stream"),
            )
    path = Path(source)
    return (path.read_bytes(), path.name,
            mimetypes.guess_type(path.name)[0] or "application/octet-stream")


async def _transcribe(session, args, source):
    audio, filename, content_type = await _load_audio(session, source)
    if args.api_mode == "chinese-asr":
        request_kwargs = {"json": {
            "base64": base64.b64encode(audio).decode("ascii"),
            "article_url": source,
            "hotwords": args.hotword or None,
        }}
    else:
        form = aiohttp.FormData()
        form.add_field("file", audio, filename=filename, content_type=content_type)
        form.add_field("model", args.model)
        form.add_field("response_format", "json")
        form.add_field("temperature", "0")
        if args.language:
            form.add_field("language", args.language)
        request_kwargs = {"data": form}
    async with session.post(args.url, **request_kwargs) as response:
        body = await response.json(content_type=None)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body}")
        if args.api_mode == "chinese-asr" and body.get("code") != 0:
            raise RuntimeError(f"业务失败: {body}")
        field = "istar_asr" if args.api_mode == "chinese-asr" else "text"
        text = body.get(field)
        if not isinstance(text, str):
            raise RuntimeError(f"响应缺少字符串字段 {field}: {body}")
        return text


async def _run(args):
    audios = _collect_audio(args.input)
    if args.limit:
        audios = audios[:args.limit]
    refs = _load_refs(args.ref, args.ref_text, audios)
    if args.ref or args.ref_text is not None:
        _validate_refs(audios, refs)
    if args.baseline_cer is not None and not refs:
        raise ValueError("使用 --baseline-cer 时必须提供 --ref 或 --ref-text")
    total_distance = 0
    total_length = 0
    scored = []
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for source in audios:
            text = await _transcribe(session, args, source)
            name = source if _is_url(source) else Path(source).name
            print(f"[{name}] {text}")
            if source in refs:
                distance, length = _cer(refs[source], text)
                total_distance += distance
                total_length += length
                if length <= 0:
                    raise ValueError(f"参考文本归一化后为空: {source}")
                sample_cer = distance / length
                scored.append((source, sample_cer, refs[source], text))
                print(f"  CER={sample_cer:.4f}")
    if not scored:
        if args.baseline_cer is not None:
            raise ValueError("没有可用于 CER 门禁的有效样本")
        return 0
    if total_length <= 0:
        raise ValueError("参考文本总字符数为 0，无法计算 CER")
    average = total_distance / total_length
    print(
        f"\n输入样本: {len(audios)}  计分样本: {len(scored)}  "
        f"总参考字符: {total_length}  平均CER: {average:.4f}"
    )
    for source, sample_cer, ref, hyp in sorted(
        scored, key=lambda item: item[1], reverse=True
    )[:5]:
        print(f"[{Path(source).name}] CER={sample_cer:.4f}\n  ref: {ref}\n  hyp: {hyp}")
    if args.baseline_cer is not None:
        passed = average <= args.baseline_cer
        print(f"go/no-go: {average:.4f} <= {args.baseline_cer:.4f} → "
              f"{'GO' if passed else 'NO-GO'}")
        return 0 if passed else 2
    return 0


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR API推理与CER验证")
    parser.add_argument("--api-mode", choices=("native", "chinese-asr"), default="native")
    parser.add_argument("--url", default=None)
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "qwen3-asr"))
    parser.add_argument("--hotword", action="append", default=[], help="兼容接口动态热词，可重复")
    parser.add_argument("--input", required=True)
    parser.add_argument("--ref", default=None, help="参考目录、单条.txt或JSONL文件")
    parser.add_argument("--ref-text", default=None, help="单音频直接参考文本")
    parser.add_argument("--language", default="")
    parser.add_argument("--baseline-cer", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.baseline_cer is not None and not 0 <= args.baseline_cer <= 1:
        parser.error("--baseline-cer 必须在 0～1 之间")
    if args.limit < 0 or args.timeout <= 0:
        parser.error("--limit 不能小于 0，--timeout 必须大于 0")
    if args.url is None:
        endpoint = "/chinese_asr" if args.api_mode == "chinese-asr" else "/v1/audio/transcriptions"
        args.url = f"http://127.0.0.1:8080{endpoint}"
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
