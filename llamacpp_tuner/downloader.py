"""Discover and download GGUF files from Hugging Face."""

import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from huggingface_hub import hf_hub_download, list_repo_files

from llamacpp_tuner.cache import get_models_dir

_SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _repository_dir(repo_id: str) -> Path:
    parts = repo_id.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid Hugging Face repository ID: {repo_id}")
    return get_models_dir() / quote(repo_id, safe="")


def _repo_files(repo_id: str) -> list[str]:
    return list_repo_files(repo_id)


def _local_repo_files(repo_id: str) -> list[str]:
    repo_dir = _repository_dir(repo_id)
    if not repo_dir.exists():
        return []
    return [path.relative_to(repo_dir).as_posix() for path in repo_dir.rglob("*.gguf")]


def _shard_info(filename: str) -> tuple[str, str, int, int] | None:
    path = PurePosixPath(filename)
    match = _SHARD_RE.match(path.name)
    if not match:
        return None
    prefix, index, total = match.groups()
    return path.parent.as_posix(), prefix, int(index), int(total)


def _without_shard(filename: str) -> str:
    name = PurePosixPath(filename).name
    match = _SHARD_RE.match(name)
    return match.group(1) if match else name.removesuffix(".gguf")


def _matches_quant(filename: str, quant: str) -> bool:
    stem = _without_shard(filename).lower()
    quant = quant.lower()
    return stem == quant or stem.endswith((f"-{quant}", f"_{quant}", f".{quant}"))


def _artifact_key(filename: str) -> tuple[str, ...]:
    if shard := _shard_info(filename):
        parent, prefix, _, total = shard
        return "sharded", parent, prefix, str(total)
    return "single", filename


def _complete_artifact(files: list[str], selected: str) -> list[str]:
    shard = _shard_info(selected)
    if not shard:
        return [selected]

    parent, prefix, _, total = shard
    shards = sorted(
        (info[2], filename)
        for filename in files
        if (info := _shard_info(filename))
        and info[0] == parent
        and info[1] == prefix
        and info[3] == total
    )
    if [index for index, _ in shards] != list(range(1, total + 1)):
        raise ValueError(f"Incomplete GGUF shard set for {selected}")
    return [filename for _, filename in shards]


def _select_model(
    files: list[str], quant: str | None = None, filename: str | None = None
) -> list[str]:
    models = [
        item
        for item in files
        if item.lower().endswith(".gguf")
        and "mmproj" not in PurePosixPath(item).name.lower()
    ]

    if filename:
        if filename not in models:
            return []
        return _complete_artifact(models, filename)
    if not quant:
        return []

    matches = [item for item in models if _matches_quant(item, quant)]
    artifacts = {_artifact_key(item): item for item in matches}
    if len(artifacts) > 1:
        choices = "\n  - ".join(sorted(artifacts.values()))
        raise ValueError(
            f"Multiple {quant} artifacts found. Choose one with --file:\n  - {choices}"
        )
    if not artifacts:
        return []
    return _complete_artifact(models, next(iter(artifacts.values())))


def _select_projector(files: list[str], filename: str | None = None) -> str | None:
    projectors = [
        item
        for item in files
        if item.lower().endswith(".gguf")
        and "mmproj" in PurePosixPath(item).name.lower()
    ]
    if filename:
        return filename if filename in projectors else None
    if len(projectors) <= 1:
        return projectors[0] if projectors else None

    choices = "\n  - ".join(sorted(projectors))
    raise ValueError(
        f"Multiple projectors found. Choose one with --mmproj-file:\n  - {choices}"
    )


def find_gguf_files(
    repo_id: str, quant: str | None = None, filename: str | None = None
) -> list[str]:
    """Return one exact artifact or its complete shard set."""
    return _select_model(_repo_files(repo_id), quant=quant, filename=filename)


def find_mmproj_file(repo_id: str, filename: str | None = None) -> str | None:
    return _select_projector(_repo_files(repo_id), filename=filename)


def _download(repo_id: str, filename: str, force: bool) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=_repository_dir(repo_id),
            force_download=force,
        )
    )


def download_model(
    repo_id: str,
    quant: str | None = None,
    filename: str | None = None,
    force: bool = False,
) -> Path | None:
    files = find_gguf_files(repo_id, quant=quant, filename=filename)
    paths = [_download(repo_id, item, force) for item in files]
    return paths[0] if paths else None


def download_mmproj(
    repo_id: str, filename: str | None = None, force: bool = False
) -> Path | None:
    selected = find_mmproj_file(repo_id, filename=filename)
    return _download(repo_id, selected, force) if selected else None


def get_model_path(
    repo_id: str, quant: str | None = None, filename: str | None = None
) -> Path | None:
    repo_dir = _repository_dir(repo_id)
    files = _select_model(_local_repo_files(repo_id), quant=quant, filename=filename)
    paths = [repo_dir / item for item in files]
    return paths[0] if paths and all(path.is_file() for path in paths) else None


def get_mmproj_path(repo_id: str, filename: str | None = None) -> Path | None:
    selected = _select_projector(_local_repo_files(repo_id), filename=filename)
    if not selected:
        return None
    path = _repository_dir(repo_id) / selected
    return path if path.is_file() else None


def resolve_model(
    model: str, quant: str | None = None, filename: str | None = None
) -> Path:
    """Resolve a local path, downloaded filename, or repository ID offline."""
    supplied = Path(model).expanduser()
    if supplied.is_file():
        return supplied

    if quant or filename:
        if path := get_model_path(model, quant=quant, filename=filename):
            return path
    elif model.lower().endswith(".gguf") and supplied.parent == Path("."):
        matches = list(get_models_dir().rglob(supplied.name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = "\n  - ".join(str(path) for path in matches)
            raise ValueError(f"Multiple local files match {model}:\n  - {choices}")
    elif not model.lower().endswith(".gguf"):
        raise ValueError("Repository models require --quant or --file.")

    selector = filename or quant or model
    raise FileNotFoundError(
        f"Model not found: {model} ({selector}). Download it with 'lct pull'."
    )


def list_downloaded_models() -> list[Path]:
    return sorted(get_models_dir().rglob("*.gguf"))
