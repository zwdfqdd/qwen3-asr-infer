"""管理 Qwen3-ASR 模型：同步 ModelScope 原始权重，或只读验真本地微调模型。"""



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

    "MODELSCOPE_REVISION", "master"

)  # 不可变提交 revision，禁止使用 latest。

ENDPOINT = os.getenv("MODELSCOPE_ENDPOINT", "https://modelscope.cn").rstrip("/")  # API 基址。

MANIFEST_NAME = ".modelscope-manifest.json"

_ALLOW_SUFFIXES = (".json", ".txt", ".safetensors", ".model", ".jinja")

_SKIP_FILES = {".gitattributes", "README.md", "generation_config.json"}

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





def verify_local_model(target_dir: str) -> str:

    """只检查本地模型启动必需文件；不联网、不监测版本、不写入任何文件。"""

    target = Path(target_dir)

    if not target.is_dir():

        raise RuntimeError(f"本地模型目录不存在或不是目录: {target}")

    missing = missing_model_files(str(target))

    if missing:

        raise RuntimeError(f"本地模型缺少启动必需文件: {missing}")

    print(f"本地模型结构检查通过: {target}（不校验版本、大小或 SHA256）")

    return str(target)





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
    """只判断文件是否存在且非空，不校验版本、大小或 SHA256。"""
    return path.is_file() and path.stat().st_size > 0


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

    """确保本地目录含完整模型并返回目录字符串。

    只判断必需文件存在且非空，不校验版本、大小或 SHA256；缺失或为空时才从 ModelScope

    按官方清单补齐，绝不覆盖已存在的非空文件。

    """

    if not repo_id.strip() or not revision.strip():

        raise ValueError("model-id 和 revision 不能为空")

    target = Path(target_dir)

    with _download_lock(target):

        missing = missing_model_files(str(target))

        if not missing:

            print(f"本地模型完整: {target}（不校验版本、大小或 SHA256）")

            return str(target)

        files = _list_repo_files(repo_id, revision)

        # generation_config 不属于官方同步清单；run.sh 已显式使用
        # \`--generation-config vllm\`，因此保留目录中的该文件且不参与 ModelScope 校验。
        files = [

            item for item in files

            if Path(str(item.get("path", ""))).name not in _SKIP_FILES

        ]



        # 不校验目录绑定的仓库版本、文件大小或 SHA256；只补齐缺失/空文件。
        invalid = _validate_files(target, files)
        if invalid:
            print(f"同步缺失或空文件: {repo_id} -> {target}（revision={revision}）")
            for item in files:
                if item["path"] in invalid:
                    _download_file(repo_id, item, target, revision)

        missing = missing_model_files(str(target))
        if missing:
            raise RuntimeError(f"模型必需文件缺失: {missing}")
    return str(target)





def main() -> None:

    parser = argparse.ArgumentParser(

        description="同步 ModelScope 原始模型，或只读验真本地微调模型"

    )

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

        help="ModelScope 分支或 revision；默认 master，不用于本地版本校验",

    )

    parser.add_argument(

        "--verify-local",

        action="store_true",

        help="只检查本地模型启动必需文件；不联网、不监测版本或哈希",

    )

    args = parser.parse_args()

    try:

        if args.verify_local:

            verify_local_model(args.dir)

        else:

            ensure_model(args.dir, args.model_id, args.revision)

    except Exception as error:  # noqa: BLE001

        print(f"模型操作失败: {type(error).__name__}: {error}", file=sys.stderr)

        raise SystemExit(1) from error





if __name__ == "__main__":

    main()