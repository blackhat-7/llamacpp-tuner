"""Writable paths managed by lct."""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def get_checkout_dir() -> Path | None:
    """The lct repository this code runs from; None for an installed lct."""
    return _PROJECT_ROOT if (_PROJECT_ROOT / "pyproject.toml").is_file() else None


def get_cache_dir() -> Path:
    if home := os.environ.get("LCT_HOME"):
        return Path(home).expanduser()
    if checkout := get_checkout_dir():
        return checkout / "tmp"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "lct"


def get_profiles_path() -> Path:
    """A checkout's profiles.toml, kept in git; with LCT_HOME set, or for an
    installed lct, aliases.toml in the cache dir."""
    checkout = get_checkout_dir()
    if checkout and not os.environ.get("LCT_HOME"):
        return checkout / "profiles.toml"
    return get_cache_dir() / "aliases.toml"


def get_models_dir() -> Path:
    path = get_cache_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_llama_dir() -> Path:
    path = get_cache_dir() / "llama.cpp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_servers_dir() -> Path:
    """Per-profile pid and log files of servers started with `lct up` or the TUI."""
    path = get_cache_dir() / "servers"
    path.mkdir(parents=True, exist_ok=True)
    return path
