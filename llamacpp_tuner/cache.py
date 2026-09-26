"""Writable paths managed by lct."""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def get_cache_dir() -> Path:
    if home := os.environ.get("LCT_HOME"):
        return Path(home).expanduser()
    if (_PROJECT_ROOT / "pyproject.toml").is_file():
        return _PROJECT_ROOT / "tmp"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "lct"


def get_aliases_path() -> Path:
    return get_cache_dir() / "aliases.toml"


def get_models_dir() -> Path:
    path = get_cache_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_llama_dir() -> Path:
    path = get_cache_dir() / "llama.cpp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_server_log_path() -> Path:
    return get_cache_dir() / "server.log"


def get_server_state_path() -> Path:
    return get_cache_dir() / "server.json"
