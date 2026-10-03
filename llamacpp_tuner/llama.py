"""Build and run llama.cpp."""

import json
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from llamacpp_tuner.cache import get_llama_dir

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


def run_server(args: list[str], env: dict[str, str] | None = None) -> None:
    """Run llama-server in the foreground; env adds variables to its environment."""
    binary = get_llama_binary()
    if not binary:
        raise FileNotFoundError(
            "llama-server not found. Install it or run 'lct setup'."
        )
    run_process([str(binary), *args], env=env)


def run_process(
    command: list[str], env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    """Run a server in the foreground until it exits or lct is stopped."""
    with subprocess.Popen(
        command, env={**os.environ, **env} if env else None, cwd=cwd
    ) as server:
        # Without this, a SIGTERM to lct kills only the wrapper and orphans
        # the server, leaving the model resident in VRAM.
        def shutdown(signum: int, frame: object) -> None:
            server.terminate()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, shutdown)
        code = server.wait()

    # A negative code means the server was stopped by a signal, which is how
    # a requested shutdown ends.
    if code > 0:
        raise subprocess.CalledProcessError(code, command)
