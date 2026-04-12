"""Download models from HuggingFace Hub."""

import json
import os
import sys
from pathlib import Path

import requests
from huggingface_hub import HfApi, list_repo_files
from huggingface_hub.errors import RepositoryNotFoundError
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from llamacpp_tuner.cache import CACHE_DIR, get_models_dir
from llamacpp_tuner.constants import GGUF_MIN_SIZE_BYTES
from llamacpp_tuner.types import Quant

CHUNK_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 3
RETRY_DELAY = 5


def _get_expected_sizes_path() -> Path:
    return CACHE_DIR / "expected_sizes.json"


def _save_expected_size(filename: str, size: int) -> None:
    sizes_path = _get_expected_sizes_path()
    sizes: dict[str, int] = {}
    if sizes_path.exists():
        with open(sizes_path) as f:
            sizes = json.load(f)
    sizes[filename] = size
    with open(sizes_path, "w") as f:
        json.dump(sizes, f)


def _get_expected_size(filename: str) -> int | None:
    sizes_path = _get_expected_sizes_path()
    if not sizes_path.exists():
        return None
    with open(sizes_path) as f:
        sizes: dict[str, int] = json.load(f)
    return sizes.get(filename)


def _quant_to_pattern(quant: Quant) -> str:
    mapping = {
        "Q4_K_M": "q4_k_m",
        "Q4_K_S": "q4_k_s",
        "Q5_K_M": "q5_k_m",
        "Q5_K_S": "q5_k_s",
        "Q8_0": "q8_0",
        "IQ4_XS": "iq4_xs",
    }
    return mapping[quant]


def find_mmproj_file(repo_id: str) -> str | None:
    """Find mmproj file in a HuggingFace repo."""
    try:
        files = list_repo_files(repo_id)
    except RepositoryNotFoundError:
        return None

    mmproj_files = [f for f in files if f.startswith("mmproj") and f.endswith(".gguf")]
    if not mmproj_files:
        return None

    priority = ["f16", "bf16", "q8", "q6", "q5", "q4", "f32"]
    for p in priority:
        for f in mmproj_files:
            if p in f.lower():
                return f

    return mmproj_files[0] if mmproj_files else None


def download_mmproj(repo_id: str, force: bool = False) -> Path | None:
    """Download mmproj file from HuggingFace repo."""
    filename = find_mmproj_file(repo_id)
    if not filename:
        return None

    models_dir = get_models_dir()
    local_path = models_dir / filename

    if local_path.exists() and not force:
        expected_size = _get_expected_size(filename)
        if expected_size and local_path.stat().st_size >= expected_size:
            return local_path

    file_size = _get_file_size(repo_id, filename)
    if file_size:
        _save_expected_size(filename, file_size)

    size_str = _format_size(file_size) if file_size else "unknown size"
    print(f"Downloading mmproj: {filename} ({size_str})...")

    url = _get_download_url(repo_id, filename)

    downloaded_size = 0
    for attempt in range(MAX_RETRIES):
        try:
            downloaded_size = _download_with_progress(
                url, local_path, file_size, filename
            )

            if file_size and downloaded_size < file_size:
                print(
                    f"\nDownload incomplete: {downloaded_size}/{file_size} bytes",
                    file=sys.stderr,
                )
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 2}/{MAX_RETRIES})"
                    )
                    import time

                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print("Max retries reached. Download incomplete.", file=sys.stderr)
                    return None
            break
        except requests.exceptions.RequestException as e:
            if local_path.exists():
                local_path.unlink()
            if attempt < MAX_RETRIES - 1:
                print(f"Download error: {e}", file=sys.stderr)
                print(
                    f"Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 2}/{MAX_RETRIES})"
                )
                import time

                time.sleep(RETRY_DELAY)
            else:
                print(
                    f"Download failed after {MAX_RETRIES} attempts: {e}",
                    file=sys.stderr,
                )
                return None

    return local_path


def get_mmproj_path(repo_id: str) -> Path | None:
    """Get path to downloaded mmproj file."""
    filename = find_mmproj_file(repo_id)
    if not filename:
        return None
    models_dir = get_models_dir()
    local_path = models_dir / filename
    if local_path.exists():
        return local_path
    return None


def find_gguf_file(repo_id: str, quant: Quant) -> str | None:
    pattern = _quant_to_pattern(quant)

    try:
        files = list_repo_files(repo_id)
    except RepositoryNotFoundError:
        return None

    gguf_files = [f for f in files if f.endswith(".gguf")]

    for f in gguf_files:
        if pattern in f.lower():
            return f

    if "q4" in pattern:
        for f in gguf_files:
            if "q4" in f.lower():
                return f

    return gguf_files[0] if gguf_files else None


def _get_file_size(repo_id: str, filename: str) -> int | None:
    try:
        api = HfApi()
        repo_info = api.model_info(repo_id, files_metadata=True)
        for sibling in repo_info.siblings or []:
            if sibling.rfilename == filename and sibling.size is not None:
                return sibling.size
        return None
    except Exception as e:
        print(f"Warning: Could not get file size: {e}", file=sys.stderr)
        return None


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _get_download_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _download_with_progress(
    url: str, dest: Path, total_size: int | None, filename: str
) -> int:
    headers = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    existing_size = 0
    if dest.exists():
        existing_size = dest.stat().st_size
        if total_size and existing_size >= total_size:
            return existing_size
        headers["Range"] = f"bytes={existing_size}-"

    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    if total_size is None:
        total_size = int(response.headers.get("Content-Length", 0))
        if existing_size > 0 and "Content-Range" in response.headers:
            content_range = response.headers["Content-Range"]
            if "/" in content_range:
                total_size = int(content_range.split("/")[-1])

    mode = "ab" if existing_size > 0 else "wb"

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[filename]}", justify="left"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )

    downloaded = existing_size
    with progress:
        task = progress.add_task(
            "download",
            total=total_size if total_size > 0 else None,
            completed=existing_size,
            filename=filename[:40],
        )

        with open(dest, mode) as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress.update(task, advance=len(chunk))

    return downloaded


def download_model(repo_id: str, quant: Quant, force: bool = False) -> Path | None:
    filename = find_gguf_file(repo_id, quant)
    if not filename:
        return None

    models_dir = get_models_dir()
    local_path = models_dir / filename

    if local_path.exists() and _is_valid_model(local_path) and not force:
        expected_size = _get_expected_size(filename)
        if expected_size and local_path.stat().st_size >= expected_size:
            return local_path

    file_size = _get_file_size(repo_id, filename)
    if file_size:
        _save_expected_size(filename, file_size)

    size_str = _format_size(file_size) if file_size else "unknown size"
    print(f"Downloading {filename} ({size_str})...")

    url = _get_download_url(repo_id, filename)

    downloaded_size = 0
    for attempt in range(MAX_RETRIES):
        try:
            downloaded_size = _download_with_progress(
                url, local_path, file_size, filename
            )

            if file_size and downloaded_size < file_size:
                print(
                    f"\nDownload incomplete: {downloaded_size}/{file_size} bytes",
                    file=sys.stderr,
                )
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 2}/{MAX_RETRIES})"
                    )
                    import time

                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print("Max retries reached. Download incomplete.", file=sys.stderr)
                    return None

            break
        except requests.exceptions.RequestException as e:
            if local_path.exists():
                local_path.unlink()
            if attempt < MAX_RETRIES - 1:
                print(f"Download error: {e}", file=sys.stderr)
                print(
                    f"Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 2}/{MAX_RETRIES})"
                )
                import time

                time.sleep(RETRY_DELAY)
            else:
                print(
                    f"Download failed after {MAX_RETRIES} attempts: {e}",
                    file=sys.stderr,
                )
                return None

    return local_path


def list_downloaded_models() -> list[Path]:
    models_dir = get_models_dir()
    return sorted(models_dir.glob("*.gguf"))


def list_valid_models() -> list[Path]:
    models_dir = get_models_dir()
    return sorted([p for p in models_dir.glob("*.gguf") if _is_valid_model(p)])


def list_invalid_models() -> list[Path]:
    models_dir = get_models_dir()
    return sorted([p for p in models_dir.glob("*.gguf") if not _is_valid_model(p)])


def remove_model(path: Path) -> bool:
    if path.exists() and path.suffix == ".gguf":
        path.unlink()
        return True
    return False


def _is_valid_model(path: Path) -> bool:
    if not path.exists():
        return False
    actual_size = path.stat().st_size
    if actual_size < GGUF_MIN_SIZE_BYTES:
        return False

    expected_size = _get_expected_size(path.name)
    if expected_size is not None:
        return actual_size >= expected_size
    return True


def get_model_path(repo_id: str, quant: Quant) -> Path | None:
    filename = find_gguf_file(repo_id, quant)
    if not filename:
        return None
    models_dir = get_models_dir()
    local_path = models_dir / filename
    if _is_valid_model(local_path):
        return local_path
    return None
