"""llama.cpp binary management and server execution."""

import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

from llamacpp_tuner.cache import get_llama_dir

LLAMA_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download"


def _get_binary_name() -> str:
    return "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def _get_platform_key() -> str:
    """Get platform identifier for pre-built binaries."""
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        return "win-cpu-x64" if "64" in machine else "win-cpu-arm64"
    elif system == "Darwin":
        return "macos-arm64" if "arm" in machine else "macos-x64"
    elif system == "Linux":
        if "arm" in machine or "aarch64" in machine:
            return "ubuntu-arm64"
        return "ubuntu-x64"
    return ""


def _has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is present."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count > 0
    except Exception:
        return False


def _find_nvcc() -> Path | None:
    """Find nvcc compiler and return path, or None if not found."""
    if nvcc := shutil.which("nvcc"):
        return Path(nvcc)

    for path in [
        "/opt/cuda/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ]:
        if Path(path).exists():
            return Path(path)
    return None


def _get_cuda_root(nvcc_path: Path) -> Path:
    """Get CUDA toolkit root directory from nvcc path."""
    return nvcc_path.parent.parent


def get_llama_binary() -> Path | None:
    """Get path to llama-server binary."""
    llama_dir = get_llama_dir()
    binary = llama_dir / "build" / "bin" / _get_binary_name()
    if binary.exists():
        return binary

    # Check for downloaded binary
    binary = llama_dir / _get_binary_name()
    if binary.exists():
        return binary

    return None


def is_llama_installed() -> bool:
    return get_llama_binary() is not None


def download_binary(version: str = "b8664", backend: str = "auto") -> Path:
    """Download pre-built binary from GitHub releases."""
    llama_dir = get_llama_dir()
    llama_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system()

    # Determine backend
    if backend == "auto":
        backend = "cuda-12" if system == "Windows" and _has_nvidia_gpu() else "cpu"

    platform_key = _get_platform_key()
    if not platform_key:
        raise RuntimeError(f"Unsupported platform: {system} {platform.machine()}")

    # Construct download URL
    if system == "Windows":
        if "cuda" in backend:
            platform_key = platform_key.replace("cpu", "cuda-12.4")
        archive_name = f"llama-{version}-bin-{platform_key}.zip"
    else:
        if "cuda" in backend:
            # Linux doesn't have pre-built CUDA binaries in releases
            raise RuntimeError(
                "Linux CUDA binaries not available. "
                "Install CUDA toolkit or use Docker: "
                "https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md"
            )
        archive_name = f"llama-{version}-bin-{platform_key}.tar.gz"

    url = f"{RELEASE_URL}/{version}/{archive_name}"
    archive_path = llama_dir / archive_name

    print(f"Downloading llama.cpp {version} ({backend})...")
    print(f"URL: {url}")

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Extracting {archive_name}...")
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(llama_dir)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(llama_dir)

    archive_path.unlink()

    binary = llama_dir / _get_binary_name()
    if not binary.exists():
        # Try to find binary in extracted structure
        for pattern in ["**/llama-server*", "bin/llama-server*"]:
            matches = list(llama_dir.glob(pattern))
            if matches:
                binary = matches[0]
                break

    if not binary or not binary.exists():
        raise RuntimeError(f"Binary not found after extraction in {llama_dir}")

    binary.chmod(0o755)
    print(f"Installed llama.cpp: {binary}")
    return binary


def build_from_source(force: bool = False) -> Path:
    """Build llama.cpp from source."""
    llama_dir = get_llama_dir()
    build_dir = llama_dir / "build"
    binary = build_dir / "bin" / _get_binary_name()

    if binary.exists() and not force:
        print(f"llama.cpp already built: {binary}")
        return binary

    # Clean build directory if forcing rebuild
    if force and build_dir.exists():
        print("Cleaning previous build...")
        shutil.rmtree(build_dir)

    # Clone repository
    llama_git_dir = llama_dir / ".git"
    if not llama_git_dir.exists():
        if llama_dir.exists():
            shutil.rmtree(llama_dir)
        print("Cloning llama.cpp repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_REPO_URL, str(llama_dir)],
            check=True,
        )

    build_dir.mkdir(exist_ok=True)

    # Configure build
    print("Configuring build...")
    cmake_args = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(llama_dir),
    ]

    # Check for CUDA
    has_gpu = _has_nvidia_gpu()
    nvcc = _find_nvcc()

    if has_gpu and nvcc:
        print("CUDA detected, building with GPU support...")
        cuda_root = _get_cuda_root(nvcc)
        cmake_args.extend(
            [
                "-DGGML_CUDA=ON",
                f"-DCUDAToolkit_ROOT={cuda_root}",
                f"-DCMAKE_CUDA_COMPILER={nvcc}",
            ]
        )
    elif has_gpu and not nvcc:
        print("NVIDIA GPU detected but CUDA toolkit not found!")
        print("Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        print("Or install: sudo pacman -S cuda  (Arch)")
        print("           sudo apt install nvidia-cuda-toolkit  (Ubuntu)")
        print("\nBuilding CPU-only version for now...")

    subprocess.run(cmake_args, check=True)

    # Build
    # CUDA compilation is memory-intensive (2-4GB per nvcc process)
    # Limit parallelism to avoid OOM
    if has_gpu and nvcc:
        print(
            "Building llama.cpp with CUDA (5-10 min, limited parallelism to prevent OOM)..."
        )
        parallel_jobs = 4
    else:
        print("Building llama.cpp (this may take a few minutes)...")
        parallel_jobs = subprocess.cpu_count() or 4

    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "-j",
            str(parallel_jobs),
        ],
        check=True,
    )

    if not binary.exists():
        raise RuntimeError("Build failed: llama-server not found")

    print(f"Built llama.cpp: {binary}")
    return binary


def install_llama(prefer_binary: bool = True) -> Path:
    """Install llama.cpp: try pre-built binary first, then build from source."""
    system = platform.system()
    has_gpu = _has_nvidia_gpu()

    # On Linux with NVIDIA GPU, skip binary download and build from source
    # since pre-built CUDA binaries aren't available
    if system == "Linux" and has_gpu:
        print(
            "NVIDIA GPU detected on Linux - building from source with CUDA support..."
        )
        return build_from_source()

    if prefer_binary:
        try:
            return download_binary()
        except Exception as e:
            print(f"Pre-built binary download failed: {e}")
            print("Falling back to building from source...")

    return build_from_source()


def run_server(args: list[str]) -> None:
    binary = get_llama_binary()
    if not binary:
        print(
            "llama.cpp not installed. Run 'lct setup' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [str(binary)] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def has_gpu_support() -> bool:
    """Check if binary was compiled with CUDA/ROCm/Vulkan support."""
    binary = get_llama_binary()
    if not binary:
        return False

    try:
        # Check if the binary was built with GPU backends
        # CUDA binaries have GGML_CUDA symbols
        result = subprocess.run(
            ["ldd", str(binary)],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        output = result.stdout

        # Check for CUDA libraries
        if "libcudart" in output or "libcublas" in output:
            return True

        # Check for ROCm libraries
        if "libhip" in output or "librocblas" in output:
            return True

        # Check for Vulkan
        return "libvulkan" in output
    except Exception:
        # Fallback: check if we can query devices
        try:
            result = subprocess.run(
                [str(binary), "--list-devices"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            output = result.stdout + result.stderr
            # If it lists any GPU devices, GPU support is working
            return "CUDA" in output or "Vulkan" in output or "ROCm" in output
        except Exception:
            return False
