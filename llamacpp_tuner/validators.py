"""Validation helpers for CLI commands."""

from pathlib import Path

from llamacpp_tuner.cache import get_models_dir, load_cache
from llamacpp_tuner.downloader import _is_valid_model, get_model_path, list_valid_models
from llamacpp_tuner.exceptions import (
    HardwareNotDetectedError,
    LlamaNotInstalledError,
    ModelNotFoundError,
)
from llamacpp_tuner.hardware import HardwareProfile
from llamacpp_tuner.llama import get_llama_binary, is_llama_installed
from llamacpp_tuner.types import Quant


def validate_hardware_cache() -> HardwareProfile:
    """Load and validate hardware cache.

    Raises:
        HardwareNotDetectedError: If hardware cache doesn't exist
    """
    cached = load_cache("hardware")
    if not cached:
        raise HardwareNotDetectedError(
            "Hardware not detected. Run 'llamacpp-tuner setup' first."
        )
    return HardwareProfile.from_dict(cached)


def validate_llama_installed() -> Path:
    """Validate llama.cpp is installed.

    Raises:
        LlamaNotInstalledError: If llama.cpp is not installed
    """
    if not is_llama_installed():
        raise LlamaNotInstalledError(
            "llama.cpp not installed. Run 'llamacpp-tuner setup' first."
        )
    binary = get_llama_binary()
    if not binary:
        raise LlamaNotInstalledError(
            "llama.cpp not installed. Run 'llamacpp-tuner setup' first."
        )
    return binary


def validate_model_path(repo_id: str, quant: Quant) -> Path:
    """Validate model exists locally.

    Args:
        repo_id: HuggingFace repository ID or local filename
        quant: Quantization level

    Returns:
        Path to model file

    Raises:
        ModelNotFoundError: If model not found
    """
    if repo_id.endswith(".gguf"):
        models_dir = get_models_dir()
        local_path = models_dir / repo_id
        if _is_valid_model(local_path):
            return local_path
        available = [m.name for m in list_valid_models()]
        raise ModelNotFoundError(repo_id, quant, available)

    model_path = get_model_path(repo_id, quant)
    if not model_path:
        available = [m.name for m in list_valid_models()]
        raise ModelNotFoundError(repo_id, quant, available)
    return model_path
