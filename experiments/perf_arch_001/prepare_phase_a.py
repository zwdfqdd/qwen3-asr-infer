#!/usr/bin/env python3
"""PERF-ARCH-001 Phase A 样本准备：把任意来源的 11 语种音频整理成合规样本与 manifest。

这不是门禁工具，只做三件人工重复度高的事：

1. 用 ffmpeg 把源音频统一成 16 kHz 单声道 PCM WAV 并截到 32 秒以内；
2. 缺参考文本时调用本机 `/chinese_asr` 显式指定语种转写，产出同名 `.txt`；
3. 生成 `phase_a.py` 需要的 11 语种 manifest。

音频语种必须真实，脚本不做语种判定也不猜测；转写文本来自主模型真实识别结果，仅作为两个
Aligner 后端的共同输入，不冒充权威逐字稿。产出仍必须由 `phase_a.py check` 复核。
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import soundfile as sf

_SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
)
_MAX_SECONDS = 32.0
_TARGET_RATE = 16000


def _find_source(source_dir: Path, language: str) -> Path | None:
    """按 `<语种>.*` 匹配源文件，语种大小写不敏感，扩展名不限。"""
    stem = language.lower()
    candidates = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.stem.lower() == stem
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeError(
            f"{language} 匹配到多个源文件，请只保留一个：{[p.name for p in candidates]}"
        )
    return candidates[0]


def _audio_info(path: Path) -> tuple[int, int, float]:
    with sf.SoundFile(path) as source:
        return source.samplerate, source.channels, source.frames / source.samplerate


def _is_compliant(path: Path) -> bool:
    try:
        rate, channels, duration = _audio_info(path)
    except Exception:  # noqa: BLE001 - 无法解码一律视为不合规，交由 ffmpeg 处理。
        return False
    return rate == _TARGET_RATE and channels == 1 and 0 < duration <= _MAX_SECONDS


def _convert(source: Path, target: Path, ffmpeg: str) -> None:
    """统一为 16 kHz 单声道 PCM WAV，并截断到 32 秒；参数显式给出，不依赖 ffmpeg 默认值。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-nostdin", "-y", "-loglevel", "error",
        "-i", str(source),
        "-map", "0:a:0", "-ac", "1", "-ar", str(_TARGET_RATE),
        "-t", str(_MAX_SECONDS), "-c:a", "pcm_s16le",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 转换失败（退出码 {completed.returncode}）："
            f"{completed.stderr.strip()[:400]}"
        )


def _transcribe(audio_path: Path, language: str, endpoint: str, timeout: float) -> str:
    """调用本机 /chinese_asr 显式指定语种转写；字段名为 base64，响应取 istar_asr。"""
    payload = json.dumps({
        "base64": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
        "language": language,
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace").strip()[:400]
        raise RuntimeError(f"转写请求返回 HTTP {error.code}：{detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"无法访问 {endpoint}：{error}") from error
    if body.get("code") != 0:
        raise RuntimeError(f"转写业务失败：code={body.get('code')} {body.get('message')}")
    text = str(body.get("istar_asr", "")).strip()
    if not text:
        raise RuntimeError("转写结果为空，请确认音频有人声且语种正确")
    return text


def _prepare_one(language: str, args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    audio_path = output_dir / f"{language.lower()}.wav"
    text_path = output_dir / f"{language.lower()}.txt"
    row: dict[str, Any] = {"language": language, "problems": []}

    source = _find_source(source_dir, language)
    if source is None and not audio_path.is_file():
        row["problems"].append(
            f"缺少源音频：{source_dir / (language.lower() + '.<ext>')}"
        )
        return row

    if source is not None and source == audio_path:
        # 源目录与输出目录相同时不能就地转换，否则 ffmpeg 会读写同一文件。
        row["audio_action"] = "沿用已有（源与输出同路径）"
    elif source is not None and (args.overwrite_audio or not audio_path.is_file()):
        if _is_compliant(source) and source.suffix.lower() == ".wav":
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(source.read_bytes())
            row["audio_action"] = "复制（已合规）"
        elif args.ffmpeg_convert:
            _convert(source, audio_path, args.ffmpeg)
            row["audio_action"] = "ffmpeg 转换"
        else:
            rate, channels, duration = (None, None, None)
            try:
                rate, channels, duration = _audio_info(source)
            except Exception:  # noqa: BLE001 - 只用于给出可执行的提示。
                pass
            row["problems"].append(
                f"{source.name} 不是 16 kHz 单声道 ≤32 秒 WAV"
                f"（{rate} Hz / {channels} 声道 / {duration}s）；"
                f"加 --ffmpeg-convert 自动转换，或手工执行（输入输出必须是不同路径）："
                f"ffmpeg -i {source} -ac 1 -ar 16000 -t 32 -c:a pcm_s16le {audio_path}"
            )
            return row
    else:
        row["audio_action"] = "沿用已有"

    if not _is_compliant(audio_path):
        rate, channels, duration = _audio_info(audio_path)
        row["problems"].append(
            f"{audio_path.name} 仍不合规：{rate} Hz / {channels} 声道 / {duration:.3f}s"
        )
        return row
    row["duration_seconds"] = round(_audio_info(audio_path)[2], 3)

    if text_path.is_file() and not args.overwrite_text:
        existing = text_path.read_text(encoding="utf-8").strip()
        if not existing:
            row["problems"].append(f"{text_path.name} 为空；删除后重跑或手工补全")
            return row
        row["text_action"] = "沿用已有"
        row["text_chars"] = len(existing)
    elif not args.transcribe:
        row["problems"].append(
            f"缺少 {text_path.name} 且已关闭转写；请手工提供逐字稿或去掉 --no-transcribe"
        )
        return row
    else:
        try:
            text = _transcribe(audio_path, language, args.endpoint, args.timeout)
        except RuntimeError as error:
            row["problems"].append(str(error))
            return row
        text_path.write_text(text + "\n", encoding="utf-8")
        row["text_action"] = "主模型转写"
        row["text_chars"] = len(text)

    row["audio_path"] = audio_path
    row["text_path"] = text_path
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PERF-ARCH-001 Phase A 样本准备：转换音频、生成参考文本与 manifest"
    )
    parser.add_argument(
        "--source-dir", required=True,
        help="源音频目录，文件按 <语种>.<扩展名> 命名，如 french.mp3；语种大小写不敏感",
    )
    parser.add_argument("--output-dir", default="test_data/phase_a", help="合规样本输出目录")
    parser.add_argument("--manifest", required=True, help="生成的 11 语种 manifest 路径")
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8080/chinese_asr", help="本机业务接口地址"
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="单条转写超时秒数")
    parser.add_argument(
        "--transcribe", action=argparse.BooleanOptionalAction, default=True,
        help="缺参考文本时用主模型转写；已有真实逐字稿时可传 --no-transcribe",
    )
    parser.add_argument(
        "--ffmpeg-convert", action=argparse.BooleanOptionalAction, default=True,
        help="源音频不合规时调用 ffmpeg 转换；关闭则只报错并给出手工命令",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 可执行文件")
    parser.add_argument("--overwrite-audio", action="store_true", help="覆盖已有输出音频")
    parser.add_argument("--overwrite-text", action="store_true", help="覆盖已有参考文本")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.suffix.lower() != ".json":
        parser.error("--manifest 必须以 .json 结尾")
    if not Path(args.source_dir).is_dir():
        parser.error(f"源目录不存在：{args.source_dir}")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")

    rows: list[dict[str, Any]] = []
    for language in _SUPPORTED_LANGUAGES:
        try:
            row = _prepare_one(language, args)
        except Exception as error:  # noqa: BLE001 - 必须跑完 11 种再汇总，不首错退出。
            row = {
                "language": language,
                "problems": [f"{type(error).__name__}: {error}"],
            }
        rows.append(row)
        if row["problems"]:
            print(f"{language}：不通过")
            for problem in row["problems"]:
                print(f"    - {problem}")
        else:
            print(
                f"{language}：就绪（{row['duration_seconds']}s，"
                f"音频{row['audio_action']}，文本{row['text_action']}，"
                f"{row['text_chars']} 字符）"
            )

    ready = [row for row in rows if not row["problems"]]
    if len(ready) != len(_SUPPORTED_LANGUAGES):
        print(
            f"\n{len(ready)}/{len(_SUPPORTED_LANGUAGES)} 就绪，未生成 manifest。"
            "补齐缺失语种后重跑；已就绪的语种会自动沿用，不会重复转写。"
        )
        return 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({
            "schema_version": 1,
            "samples": [
                {
                    "id": f"{row['language'].lower()}-01",
                    "language": row["language"],
                    "audio": str(row["audio_path"]),
                    "text": str(row["text_path"]),
                }
                for row in rows
            ],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n11/11 就绪，manifest 已写入 {manifest_path}")
    print("下一步：phase_a.py check 复核，并人工抽查各语种文本与音频内容是否对应。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
