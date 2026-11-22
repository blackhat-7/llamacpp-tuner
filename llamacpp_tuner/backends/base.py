"""What every backend shares: the serve options, the usage error, the engine
build script runner and the foreground process runner."""

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llamacpp_tuner.cache import get_cache_dir, get_checkout_dir


@dataclass(frozen=True)
class ServeOptions:
    """The 'lct serve' options that backends map onto their own server's flags."""

    quant: str | None = None
    filename: str | None = None
    ctx: int | None = None
    host: str | None = None
    port: int | None = None
    mmproj: Path | None = None
    no_mmproj: bool = False
    no_mmproj_offload: bool = False
    extra_args: str = ""
    server: Path | None = None

    @property
    def picks_gguf(self) -> bool:
        """Whether a GGUF file, context or projector was given; engines with their
        own model format set those themselves."""
        return any(
            (
                self.quant,
                self.filename,
                self.ctx,
                self.mmproj,
                self.no_mmproj,
                self.no_mmproj_offload,
            )
        )


class Backend(Protocol):
    """What every module in BACKENDS provides; pyright checks each one."""

    def serve(self, model: str, options: ServeOptions, env: dict[str, str]) -> None: ...

    def setup(
        self, force: bool, cmake_args: tuple[str, ...], torch_index: str | None
    ) -> str: ...

    def address(self, line: str) -> str: ...


class UnsupportedOptionError(Exception):
    """An option this engine cannot take; the CLI shows it as a usage error."""


def build_engine(name: str, force: bool) -> str:
    """Run the repository's engines/<name>/build.sh, which rebuilds the engine
    with our changes under the cache dir."""
    checkout = get_checkout_dir()
    if checkout is None:
        raise RuntimeError(
            f"{name} builds from the lct repository: clone it and run "
            f"engines/{name}/build.sh there."
        )
    script = checkout / "engines" / name / "build.sh"
    if not script.is_file():
        raise FileNotFoundError(f"No build script at {script}.")
    # The script installs under LCT_HOME; pass lct's own choice so both agree.
    subprocess.run(
        ["bash", str(script), *(["--force"] if force else [])],
        env={**os.environ, "LCT_HOME": str(get_cache_dir())},
        check=True,
    )
    return f"{name}: built by {script}"


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
