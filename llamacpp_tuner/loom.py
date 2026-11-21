"""Run Loom (ROCm/hrx-system), AMD's inference engine, on a Loom model package.

A package is a directory with a model.json card and its .loom programs; loom-fetch
writes one. lct expects loom-serve in a venv named venv beside the package.
The package's preset sets context, drafting and memory; lct only starts it.
"""

import os
import resource
import subprocess
from pathlib import Path

from llamacpp_tuner.llama import run_process

# Loom pins 16 MiB of I/O buffers through io_uring. Under Linux's 8 MiB default it
# still reports ready, then fails every request.
MIN_MEMLOCK = 64 * 1024**2


def _allow_locking() -> None:
    """Let this process, and the server it starts, lock MIN_MEMLOCK of memory."""
    _, ceiling = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if ceiling == resource.RLIM_INFINITY or ceiling >= MIN_MEMLOCK:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (MIN_MEMLOCK, ceiling))
        return
    # Only root may raise the ceiling. prlimit changes this process alone, until it
    # exits, so no system file has to allow more for every program.
    limit = f"--memlock={MIN_MEMLOCK}:{MIN_MEMLOCK}"
    try:
        subprocess.run(
            ["sudo", "prlimit", "--pid", str(os.getpid()), limit], check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Loom needs to lock 64 MiB of memory, and only root can let this server "
            "lock more than 8 MiB. 'sudo prlimit' was refused or failed."
        ) from error


def run_loom(package: Path, args: list[str], env: dict[str, str] | None = None) -> None:
    """Run loom-serve for a Loom model package in the foreground."""
    if not (package / "model.json").is_file():
        raise FileNotFoundError(
            f"No Loom model package in {package}: model.json is missing. "
            "Fetch one with loom-fetch."
        )
    binary = package.parent / "venv" / "bin" / "loom-serve"
    if not binary.is_file():
        raise FileNotFoundError(
            f"loom-serve is missing: install hrx-loom into the venv {binary.parents[1]}."
        )
    _allow_locking()
    run_process([str(binary), f"--model={package}", *args], env=env, cwd=package.parent)
