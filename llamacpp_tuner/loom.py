"""Run Loom (ROCm/hrx-system), AMD's inference engine, on a Loom model package.

A package is a directory with a model.json card and its .loom programs; loom-fetch
writes one. lct expects loom-serve in a venv named venv beside the package.
The package's preset sets context, drafting and memory; lct only starts it.
"""

import resource
from pathlib import Path

from llamacpp_tuner.llama import run_process

# Loom pins 16 MiB of I/O buffers through io_uring. Under Linux's 8 MiB default it
# still reports ready, then fails every request.
MIN_MEMLOCK = 64 * 1024**2


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
    _, ceiling = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if ceiling != resource.RLIM_INFINITY and ceiling < MIN_MEMLOCK:
        raise RuntimeError(
            f"Loom needs to lock 64 MiB of memory; this session's hard limit is "
            f"{ceiling // 1024**2} MiB. Add '<user> hard memlock 65536' to a file in "
            "/etc/security/limits.d/ and log in again."
        )
    # Only the hard limit is raised system-wide; the soft limit, which is what other
    # programs run under, stays at the default. Raise it for this server alone.
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (ceiling, ceiling))
    run_process([str(binary), f"--model={package}", *args], env=env, cwd=package.parent)
