"""Cache utilities for hardware profile and model metadata."""

import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent.parent / "tmp"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _convert_tuples(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_convert_tuples(item) for item in obj]
    if isinstance(obj, list):
        return [_convert_tuples(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _convert_tuples(v) for k, v in obj.items()}
    return obj


def load_cache(key: str) -> Any | None:
    cache_file = CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    with open(cache_file) as f:
        return json.load(f)


def save_cache(key: str, data: Any) -> None:
    cache_file = CACHE_DIR / f"{key}.json"
    data = _convert_tuples(data)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def get_models_dir() -> Path:
    models_dir = CACHE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_llama_dir() -> Path:
    llama_dir = CACHE_DIR / "llama.cpp"
    llama_dir.mkdir(exist_ok=True)
    return llama_dir
