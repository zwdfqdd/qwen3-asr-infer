"""从 ModelScope 下载并校验 Qwen3-ASR 原始权重，供原生 vLLM 使用。"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# 以下四项在脚本导入时读取；命令行参数可覆盖仓库 ID、本地目录和固定 revision。
MODEL_ID = os.getenv("VLLM_MODEL_ID", "Qwen/Qwen3-ASR-0.6B")  # ModelScope 仓库 ID。
TARGET_DIR = os.getenv("VLLM_MODEL_DIR", "models/qwen3-asr-0.6b/vllm")  # 本地持久化目录。
REVISION = os.getenv(
    "MODELSCOPE_REVISION", "4ce9cc728b473a5aedbe7b6e1ea45646316824dc"
)  # 不可变提交 revision，禁止使用 latest。
ENDPOINT = os.getenv("MODELSCOPE_ENDPOINT", "https://modelscope.cn").rstrip("/")  # API 基址。
MANIFEST_NAME = ".modelscope-manifest.json"
_ALLOW_SUFFIXES = (".json", ".txt", ".safetensors", ".model", ".jinja")
_SKIP_FILES = {".gitattributes", "README.md", "generation_config.json"}
_REMOVE_LOCAL_FILES = {"generation_config.json"}
_REQUIRED = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _has_weights(target: Path) -> bool:
    """兼容单文件和分片 safetensors 权重。"""
    single = target / "model.safetensors"
    index = target / "model.safetensors.index.json"
    if single.is_file() and single.stat().st_size > 0:
        return True
    if not index.is_file() or index.stat().st_size == 0:
        return False
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = set(weight_map.values())
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return bool(shards) and all(
        (target / name).is_file() and (target / name).stat().st_size > 0
        for name in shards
    )


def missing_model_files(target_dir: str) -> list[str]:
    target = Path(target_dir)
    missing = [
        name for name in _REQUIRED
        if not (target / name).is_file() or (target / name).stat().st_size == 0
    ]
    if not _has_weights(target):
        missing.append("model.safetensors（或完整分片权重）")
    return missing


@contextlib.contextmanager
def _download_lock(target: Path):
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".download.lock").open("a+b") as lock_file:
        try:
            import fcntl
        except ImportError:
            yield
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _list_repo_files(repo_id: str, revision: str) -> list[dict[str, Any]]:
    repo_path = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    query = urllib.parse.urlencode({"Revision": revision, "Recursive": "true"})
    url = f"{ENDPOINT}/api/v1/models/{repo_path}/repo/files?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "qwen3asr-downloader"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    files = []
    for item in data.get("Data", {}).get("Files", []):
        path = item.get("Path")
        if item.get("Type") != "blob" or not isinstance(path, str):
            continue
        if not path.lower().endswith(_ALLOW_SUFFIXES) or Path(path).name in _SKIP_FILES:
            continue
        files.append({
            "path": path,
            "size": int(item.get("Size") or 0),
            "sha256": str(item.get("Sha256") or "").lower(),
        })
    if not files:
        raise RuntimeError(f"仓库未返回推理文件: {repo_id}")
    return files


def _file_matches(path: Path, metadata: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    expected_size = metadata["size"]
    if expected_size and path.stat().st_size != expected_size:
        return False
    expected_hash = metadata["sha256"]
    return not expected_hash or _hash_file(path) == expected_hash


def _download_file(
    repo_id: str,
    metadata: dict[str, Any],
    target: Path,
    revision: str,
) -> None:
    path = metadata["path"]
    destination = target / path
    if _file_matches(destination, metadata):
        print(f"  跳过（校验通过）：{path}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    repo_path = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    file_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    revision_path = urllib.parse.quote(revision, safe="")
    url = f"{ENDPOINT}/models/{repo_path}/resolve/{revision_path}/{file_path}"
    request = urllib.request.Request(url, headers={"User-Agent": "qwen3asr-downloader"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while block := response.read(1 << 20):
                output.write(block)
                done += len(block)
                if total:
                    print(f"\r  {path}: {done * 100 // total:3d}%", end="")
        print()
        if not _file_matches(part, metadata):
            raise RuntimeError(f"下载文件大小或 SHA256 异常: {path}")
        os.replace(part, destination)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def _manifest_payload(
    repo_id: str,
    revision: str,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "endpoint": ENDPOINT,
        "repo_id": repo_id,
        "revision": revision,
        "files": files,
    }


def _read_manifest(target: Path) -> dict[str, Any] | None:
    path = target / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"模型 manifest 损坏: {path}") from error


def _write_manifest(target: Path, payload: dict[str, Any]) -> None:
    destination = target / MANIFEST_NAME
    part = destination.with_name(destination.name + ".part")
    part.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(part, destination)


def _validate_manifest_identity(
    manifest: dict[str, Any], repo_id: str, revision: str
) -> None:
    if manifest.get("repo_id") != repo_id or manifest.get("revision") != revision:
        raise RuntimeError(
            "模型目录已绑定其他仓库或 revision；请使用新的 VLLM_MODEL_DIR，"
            "不要在同一目录混用权重"
        )


def _validate_files(target: Path, files: list[dict[str, Any]]) -> list[str]:
    return [item["path"] for item in files if not _file_matches(target / item["path"], item)]


def ensure_model(
    target_dir: str = TARGET_DIR,
    repo_id: str = MODEL_ID,
    revision: str = REVISION,
) -> str:
    """确保本地目录完整对应指定 ModelScope 仓库和 revision，并返回目录字符串。

    已有 manifest 会绑定仓库身份；文件按大小/SHA256 校验，缺失项才下载，不允许在同一
    目录静默混用模型或 revision。
    """
    if not repo_id.strip() or not revision.strip():
        raise ValueError("model-id 和 revision 不能为空")
    target = Path(target_dir)
    with _download_lock(target):
        manifest = _read_manifest(target)
        if manifest:
            _validate_manifest_identity(manifest, repo_id, revision)
            files = manifest.get("files")
            if not isinstance(files, list) or not files:
                raise RuntimeError("模型 manifest 缺少文件清单")
        else:
            files = _list_repo_files(repo_id, revision)

        # 兼容旧 manifest：generation_config 同时含 do_sample=false/temperature，
        # 对 greedy ASR 无效且会触发 Transformers 启动告警；vLLM 使用内置生成配置。
        files = [
            item for item in files
            if Path(str(item.get("path", ""))).name not in _SKIP_FILES
        ]
        for filename in _REMOVE_LOCAL_FILES:
            (target / filename).unlink(missing_ok=True)

        invalid = _validate_files(target, files)
        if invalid:
            print(f"下载 {repo_id} → {target}（revision={revision}）")
            for item in files:
                if item["path"] in invalid:
                    _download_file(repo_id, item, target, revision)

        invalid = _validate_files(target, files)
        missing = missing_model_files(str(target))
        if invalid or missing:
            raise RuntimeError(f"模型校验失败: 文件={invalid}，必需项={missing}")
        _write_manifest(target, _manifest_payload(repo_id, revision, files))
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 ModelScope 下载并校验 Qwen3-ASR 权重")
    parser.add_argument(
        "--model-id", default=MODEL_ID,
        help="ModelScope 仓库 ID；默认读取 VLLM_MODEL_ID",
    )
    parser.add_argument(
        "--dir", default=TARGET_DIR,
        help="模型本地目标目录；默认读取 VLLM_MODEL_DIR",
    )
    parser.add_argument(
        "--revision", default=REVISION,
        help="固定提交 revision；默认读取 MODELSCOPE_REVISION，禁止留空",
    )
    args = parser.parse_args()
    try:
        ensure_model(args.dir, args.model_id, args.revision)
    except Exception as error:  # noqa: BLE001
        print(f"下载失败: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()