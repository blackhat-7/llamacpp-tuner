"""Build and run llama.cpp."""

import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from llamacpp_tuner.backends.base import (
    ServeOptions,
    UnsupportedOptionError,
    run_process,
)
from llamacpp_tuner.cache import get_llama_dir
from llamacpp_tuner.downloader import get_mmproj_path, resolve_model

LLAMA_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
# Every llama.cpp build is published as a prerelease, so "latest" finds none.
RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"


def _managed_binary(source_dir: Path, name: str = "llama-server") -> Path | None:
    if platform.system() == "Windows":
        name += ".exe"
    bin_dir = source_dir / "build" / "bin"
    candidates = [bin_dir / name, bin_dir / "Release" / name]
    return next((path for path in candidates if path.is_file()), None)


def get_llama_binary(name: str = "llama-server") -> Path | None:
    """Find a llama.cpp tool such as llama-server or llama-bench."""
    if system_binary := shutil.which(name):
        return Path(system_binary)
    return _managed_binary(get_llama_dir(), name)


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


def _release_assets() -> list[tuple[str, str]]:
    """(prefix, suffix) of each release asset for this machine's GPU.

    NVIDIA gets CUDA plus its runtime libraries, everything else Vulkan, macOS Metal.
    """
    system = platform.system()
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if system == "Darwin":
        return [("llama-", f"-bin-macos-{arch}.tar.gz")]
    # nvidia-smi ships with NVIDIA's driver, which is all CUDA 12 needs once
    # the runtime libraries come from the release; no toolkit or compiler.
    cuda = {"Linux": "ubuntu-cuda-12.8-x64.tar.gz", "Windows": "win-cuda-12.4-x64.zip"}
    if shutil.which("nvidia-smi") and arch == "x64" and system in cuda:
        build = f"-bin-{cuda[system]}"
        return [("llama-", build), ("cudart-llama-", build)]
    vulkan = {
        "Linux": f"ubuntu-vulkan-{arch}.tar.gz",
        "Windows": f"win-vulkan-{arch}.zip",
    }
    return [("llama-", f"-bin-{vulkan[system]}")] if system in vulkan else []


def _asset_urls(wanted: list[tuple[str, str]]) -> list[str]:
    """Download URLs of the newest release that has every wanted asset."""
    with urllib.request.urlopen(RELEASES_URL, timeout=30) as response:
        releases = json.load(response)
    for release in releases:
        urls = [
            next(
                (
                    asset["browser_download_url"]
                    for asset in release["assets"]
                    if asset["name"].startswith(prefix)
                    and asset["name"].endswith(suffix)
                ),
                None,
            )
            for prefix, suffix in wanted
        ]
        if wanted and all(urls):
            return [url for url in urls if url]
    raise RuntimeError(
        f"No prebuilt llama.cpp for {platform.system()} {platform.machine()}. "
        "Build it: lct setup --cmake-arg=-DGGML_VULKAN=ON"
    )


def download_release() -> Path:
    """Install the newest prebuilt llama.cpp for this machine and check it runs."""
    urls = _asset_urls(_release_assets())
    source_dir = get_llama_dir()
    bin_dir = source_dir / "build" / "bin"
    with tempfile.TemporaryDirectory(dir=source_dir) as scratch:
        staged = Path(scratch) / "bin"
        for i, url in enumerate(urls):
            print(f"Downloading {url}", flush=True)
            archive = Path(scratch) / url.rsplit("/", 1)[-1]
            urllib.request.urlretrieve(url, archive)
            unpacked = Path(scratch) / str(i)
            shutil.unpack_archive(archive, unpacked, filter="data")
            # Tarballs wrap everything in one folder; zips do not. The CUDA
            # runtime lands beside llama.cpp, which loads libraries from its
            # own folder.
            entries = list(unpacked.iterdir())
            root = entries[0] if len(entries) == 1 and entries[0].is_dir() else unpacked
            shutil.copytree(root, staged, symlinks=True, dirs_exist_ok=True)
        # Swap only once everything downloaded, so a failure keeps the old install.
        shutil.rmtree(bin_dir.parent, ignore_errors=True)
        bin_dir.parent.mkdir(parents=True)
        staged.rename(bin_dir)

    binary = _managed_binary(source_dir)
    if not binary:
        raise RuntimeError(f"{urls[0]} has no llama-server.")
    check = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"The prebuilt llama-server does not run here:\n{check.stderr.strip()}\n"
            "Install your GPU driver, or build from source: "
            "lct setup --cmake-arg=-DGGML_VULKAN=ON"
        )
    return binary


def list_devices(binary: Path) -> str:
    """llama.cpp's own list of the GPUs it can use, '(none)' when it finds none."""
    result = subprocess.run(
        [str(binary), "--list-devices"], capture_output=True, text=True, check=False
    )
    return (result.stdout or result.stderr).strip()


def install_llama(force: bool = False, extra_cmake_args: Sequence[str] = ()) -> Path:
    """Find llama-server, else download a prebuilt one; CMake arguments build it."""
    existing = get_llama_binary()
    if existing and not force:
        return existing
    if extra_cmake_args:
        return build_from_source(force=force, extra_cmake_args=extra_cmake_args)
    return download_release()


def run_server(
    args: list[str], env: dict[str, str] | None = None, binary: Path | None = None
) -> None:
    """Run llama-server in the foreground; env adds variables to its environment."""
    binary = binary or get_llama_binary()
    if not binary:
        raise FileNotFoundError(
            "llama-server not found. Install it or run 'lct setup'."
        )
    run_process([str(binary), *args], env=env)


def pick_projector(model: str, mmproj: Path | None, no_mmproj: bool) -> Path | None:
    """Use an explicit projector, else a repository's downloaded one."""
    if mmproj or no_mmproj or model.lower().endswith(".gguf"):
        return mmproj
    return get_mmproj_path(model)


def serve(model: str, options: ServeOptions, env: dict[str, str]) -> None:
    """Run llama-server for a GGUF path or a downloaded repository."""
    model_path = resolve_model(model, quant=options.quant, filename=options.filename)
    mmproj = pick_projector(model, options.mmproj, options.no_mmproj)
    if options.no_mmproj_offload and not mmproj:
        raise UnsupportedOptionError(
            "--no-mmproj-offload requires --mmproj or a downloaded repository projector."
        )
    args = ["-m", str(model_path)]
    for flag, value in (
        ("-c", options.ctx),
        ("--host", options.host),
        ("--port", options.port),
    ):
        if value is not None:
            args.extend([flag, str(value)])
    if mmproj:
        args.extend(["--mmproj", str(mmproj)])
        if options.no_mmproj_offload:
            args.append("--no-mmproj-offload")
    args.extend(shlex.split(options.extra_args))
    run_server(args, env=env, binary=options.server)


def setup(force: bool, cmake_args: tuple[str, ...], torch_index: str | None) -> str:
    """Find or download llama-server; CMake arguments build it instead."""
    binary = install_llama(force=force, extra_cmake_args=cmake_args)
    devices = list_devices(binary)
    text = f"llama-server: {binary}\n{devices}"
    if "(none)" in devices:
        text += (
            "\nNo GPU found: install your GPU driver (NVIDIA, or Vulkan for "
            "AMD/Intel), then run 'lct setup --force'."
        )
    return text


def address(line: str) -> str:
    if "listening on" in line:
        return line.split("listening on")[-1].strip()
    return ""
