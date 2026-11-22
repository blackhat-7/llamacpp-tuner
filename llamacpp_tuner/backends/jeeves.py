"""Install and run Jeeves (PostHog/jeeves), a classifier llama.cpp cannot run.

Its answers come from a pointer head and a diffusion drafter on top of the base
model, so it runs on its own PyTorch server in a separate virtual environment.
"""

import json
import shlex
import shutil
import subprocess
from pathlib import Path

from llamacpp_tuner.backends.base import (
    ServeOptions,
    UnsupportedOptionError,
    run_process,
)
from llamacpp_tuner.cache import get_cache_dir
from llamacpp_tuner.downloader import resolve_repository

JEEVES_REPO_URL = "https://github.com/PostHog/jeeves.git"
# Jeeves pins torch 2.14.0; ROCm wheels start at 2.14.1, which is compatible.
TORCH = "torch==2.14.*"


def _jeeves_dir() -> Path:
    return get_cache_dir() / "jeeves"


def _python() -> Path:
    return _jeeves_dir() / "venv" / "bin" / "python"


def install_jeeves(torch_index: str | None = None, force: bool = False) -> Path:
    """Clone Jeeves and install it into its own venv; returns that venv's python.

    torch_index is a PyTorch wheel index, such as a ROCm one; PyPI's torch is CUDA.
    """
    if _python().is_file() and not force:
        return _python()
    uv = shutil.which("uv")
    if not uv:
        raise FileNotFoundError("uv not found; it is needed to install Jeeves.")
    source = _jeeves_dir() / "src"
    if force:
        shutil.rmtree(_jeeves_dir(), ignore_errors=True)
    if not (source / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", JEEVES_REPO_URL, str(source)], check=True
        )
    venv = _jeeves_dir() / "venv"
    subprocess.run([uv, "venv", "--clear", "-p", "3.12", str(venv)], check=True)
    # torch and triton come from TORCH instead, so a ROCm index can supply them.
    requirements = _jeeves_dir() / "requirements.txt"
    requirements.write_text(
        "".join(
            line
            for line in (source / "requirements.txt").read_text().splitlines(True)
            if not line.startswith(("torch", "triton"))
        )
    )
    command = [uv, "pip", "install", "-p", str(venv), TORCH, "-r", str(requirements)]
    if torch_index:
        command += ["--index-strategy", "unsafe-best-match"]
        command += ["--extra-index-url", torch_index]
    subprocess.run(command, check=True)
    return _python()


def run_jeeves(args: list[str], env: dict[str, str] | None = None) -> None:
    """Run Jeeves's /v1/systemone server in the foreground."""
    if not _python().is_file():
        raise FileNotFoundError(
            "Jeeves is not installed. Run 'lct setup --backend jeeves'."
        )
    run_process(
        [str(_python()), "-m", "inference.serve", *args],
        env=env,
        cwd=_jeeves_dir() / "src",
    )


def serve(model: str, options: ServeOptions, env: dict[str, str]) -> None:
    """Run Jeeves on a model directory or a downloaded repository; --ctx is --max-len."""
    if options.quant or options.filename or options.mmproj or options.no_mmproj_offload:
        raise UnsupportedOptionError(
            "Jeeves takes a whole model directory, not a file or projector."
        )
    weights = resolve_repository(model)
    args = ["--model", str(weights)]
    args += ["--drafter", str(weights / "drafter_k4.safetensors")]
    for flag, value in (
        ("--max-len", options.ctx),
        ("--host", options.host),
        ("--port", options.port),
    ):
        if value is not None:
            args.extend([flag, str(value)])
    run_jeeves(args + shlex.split(options.extra_args), env=env)


def setup(force: bool, cmake_args: tuple[str, ...], torch_index: str | None) -> str:
    return f"Jeeves: {install_jeeves(torch_index, force=force)}"


def address(line: str) -> str:
    if line.startswith('{"serving": '):
        return json.loads(line)["serving"]
    return ""
