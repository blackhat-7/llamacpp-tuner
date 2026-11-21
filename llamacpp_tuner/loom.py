"""Run Loom (ROCm/hrx-system), AMD's inference engine, on a Loom model package.

A package is a directory with a model.json card and its .loom programs; loom-fetch
writes one. lct expects loom-serve in a venv named venv beside the package.
The package's preset sets context, drafting and memory; lct only starts it.
Loom listens on a private port; lct serves the public one through messages.py,
which adds the Anthropic API that Claude Code needs.
"""

import json
import os
import resource
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from llamacpp_tuner.llama import run_process
from llamacpp_tuner.messages import make_server

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


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _serve_when_ready(server: ThreadingHTTPServer, upstream: str, address: str) -> None:
    """Answer clients once Loom lists its model, so 'ready' means ready."""
    while True:
        try:
            with urllib.request.urlopen(f"http://{upstream}/v1/models", timeout=5) as r:
                if json.load(r).get("data"):
                    break
        except OSError:
            pass  # Loom is still starting
        time.sleep(1)
    # lct reads the address after "listening on", as in llama-server's log.
    print(f"lct: Loom at {upstream}; listening on {address}", flush=True)
    server.serve_forever()


def run_loom(
    package: Path,
    args: list[str],
    env: dict[str, str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run loom-serve for a Loom model package in the foreground, served on host:port."""
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
    upstream = f"127.0.0.1:{_free_port()}"
    server = make_server(host, port, upstream)
    threading.Thread(
        target=_serve_when_ready,
        args=(server, upstream, f"http://{host}:{port}"),
        daemon=True,
    ).start()
    command = [str(binary), f"--model={package}", f"--listen={upstream}", *args]
    run_process(command, env=env, cwd=package.parent)
