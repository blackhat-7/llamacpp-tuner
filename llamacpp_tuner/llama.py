"""Build and run llama.cpp."""

import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from llamacpp_tuner.cache import get_llama_dir

LLAMA_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"


def _binary_name() -> str:
    return "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def _managed_binary(source_dir: Path) -> Path | None:
    bin_dir = source_dir / "build" / "bin"
    candidates = [bin_dir / _binary_name(), bin_dir / "Release" / _binary_name()]
    return next((path for path in candidates if path.is_file()), None)


def get_llama_binary() -> Path | None:
    if system_binary := shutil.which("llama-server"):
        return Path(system_binary)
    return _managed_binary(get_llama_dir())


def build_from_source(
    force: bool = False, extra_cmake_args: Sequence[str] = ()
) -> Path:
    """Build llama.cpp with an available CUDA or Vulkan backend."""
    source_dir = get_llama_dir()
    build_dir = source_dir / "build"

    if (binary := _managed_binary(source_dir)) and not force:
        return binary
    if force and build_dir.exists():
        shutil.rmtree(build_dir)

    if not (source_dir / ".git").exists():
        if source_dir.exists():
            shutil.rmtree(source_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_REPO_URL, str(source_dir)],
            check=True,
        )

    configure = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    nvcc = shutil.which("nvcc")
    if nvcc:
        configure.extend(["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc}"])
    elif shutil.which("vulkaninfo"):
        configure.append("-DGGML_VULKAN=ON")
    configure.extend(extra_cmake_args)
    subprocess.run(configure, check=True)

    jobs = min(os.cpu_count() or 4, 4) if nvcc else os.cpu_count() or 4
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs)],
        check=True,
    )
    if not (binary := _managed_binary(source_dir)):
        raise RuntimeError("Build succeeded but llama-server was not created.")
    return binary


def install_llama(force: bool = False, extra_cmake_args: Sequence[str] = ()) -> Path:
    existing = get_llama_binary()
    if existing and not force:
        return existing
    return build_from_source(force=force, extra_cmake_args=extra_cmake_args)


def run_server(args: list[str]) -> None:
    binary = get_llama_binary()
    if not binary:
        raise FileNotFoundError(
            "llama-server not found. Install it or run 'lct setup'."
        )
    subprocess.run([str(binary), *args], check=True)
