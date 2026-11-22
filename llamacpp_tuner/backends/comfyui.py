"""Install and run ComfyUI, the node-graph server lct makes video clips with.

ComfyUI lives in its own venv under the cache dir; its models, input and output
sit in a separate base directory so a reinstall keeps them.
"""

import resource
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

COMFYUI_REPO_URL = "https://github.com/Comfy-Org/ComfyUI.git"
# A release rather than master, which changes daily.
COMFYUI_VERSION = "v0.44.2"
TORCH = ("torch", "torchvision", "torchaudio")


def comfyui_dir() -> Path:
    return get_cache_dir() / "comfyui"


def data_dir() -> Path:
    """ComfyUI's --base-directory: models, input and output, kept apart from the
    code so a reinstall keeps them."""
    return comfyui_dir() / "data"


def _python() -> Path:
    return comfyui_dir() / "venv" / "bin" / "python"


def install_comfyui(torch_index: str | None = None, force: bool = False) -> Path:
    """Clone ComfyUI and install it into its own venv; returns that venv's python.

    torch_index is a PyTorch wheel index, such as a ROCm one; PyPI's torch is CUDA.
    """
    if _python().is_file() and not force:
        return _python()
    uv = shutil.which("uv")
    if not uv:
        raise FileNotFoundError("uv not found; it is needed to install ComfyUI.")
    source, venv = comfyui_dir() / "src", comfyui_dir() / "venv"
    if force:
        shutil.rmtree(source, ignore_errors=True)
    if not (source / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", COMFYUI_VERSION]
            + [COMFYUI_REPO_URL, str(source)],
            check=True,
        )
    # uv's own Python: a distribution's Python, such as Nix's, may not find the
    # system libstdc++ that torch links against.
    subprocess.run(
        [uv, "venv", "--clear", "--managed-python", "-p", "3.13", str(venv)], check=True
    )
    index = ["--index-url", torch_index] if torch_index else []
    subprocess.run([uv, "pip", "install", "-p", str(venv), *TORCH, *index], check=True)
    # Pin the torch just installed, so ComfyUI's requirements cannot swap it for
    # PyPI's CUDA build.
    frozen = subprocess.run(
        [uv, "pip", "freeze", "-p", str(venv)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    constraints = comfyui_dir() / "torch-constraints.txt"
    constraints.write_text(
        "".join(
            f"{line}\n" for line in frozen.splitlines() if line.split("==")[0] in TORCH
        )
    )
    subprocess.run(
        [uv, "pip", "install", "-p", str(venv), "-r", str(source / "requirements.txt")]
        + ["-c", str(constraints)],
        check=True,
    )
    return _python()


# A ComfyUI custom node lct installs. ComfyUI turns MIOpen off on AMD, and
# PyTorch's own ROCm kernel for Qwen3-VL's patch-embedding Conv3d (kernel equal
# to stride) segfaults the process, so every H3 job with a photo, a frame or a
# source video crashed. This runs only that convolution through MIOpen.
ROCM_FIX = """\"\"\"Installed by lct: Qwen3-VL's patch embedding through MIOpen on AMD GPUs.\"\"\"
import torch
from comfy.text_encoders import qwen35

if torch.version.hip:
    _forward = qwen35.Qwen35VisionPatchEmbed.forward

    def _forward_with_miopen(self, x):
        with torch.backends.cudnn.flags(enabled=True):
            return _forward(self, x)

    qwen35.Qwen35VisionPatchEmbed.forward = _forward_with_miopen

NODE_CLASS_MAPPINGS = {}
"""


def run_comfyui(
    base_dir: Path, args: list[str], env: dict[str, str] | None = None
) -> None:
    """Run ComfyUI's server in the foreground; base_dir holds models, input and output."""
    if not _python().is_file():
        raise FileNotFoundError(
            "ComfyUI is not installed. Run 'lct setup --backend comfyui'."
        )
    if not base_dir.is_dir():
        raise FileNotFoundError(f"No ComfyUI base directory at {base_dir}.")
    nodes = base_dir / "custom_nodes"  # ComfyUI lists it but does not create it.
    nodes.mkdir(exist_ok=True)
    (nodes / "lct_rocm_fix.py").write_text(ROCM_FIX)
    # No core dumps: a crash would spend minutes dumping ~40 GB of memory while
    # the server still looks alive.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    run_process(
        [str(_python()), "main.py", "--base-directory", str(base_dir), *args],
        env=env,
        cwd=comfyui_dir() / "src",
    )


def serve(model: str, options: ServeOptions, env: dict[str, str]) -> None:
    """Run ComfyUI; MODEL is its base directory (models, input, output)."""
    if (
        options.quant
        or options.filename
        or options.ctx
        or options.mmproj
        or options.no_mmproj
        or options.no_mmproj_offload
    ):
        raise UnsupportedOptionError(
            "ComfyUI takes its base directory; models live in it."
        )
    args = []
    for flag, value in (("--listen", options.host), ("--port", options.port)):
        if value is not None:
            args.extend([flag, str(value)])
    run_comfyui(
        Path(model).expanduser(), args + shlex.split(options.extra_args), env=env
    )


def setup(force: bool, cmake_args: tuple[str, ...], torch_index: str | None) -> str:
    return f"ComfyUI: {install_comfyui(torch_index, force=force)}"


def address(line: str) -> str:
    if "To see the GUI go to: " in line:
        return line.split("go to: ")[-1].strip()
    return ""
