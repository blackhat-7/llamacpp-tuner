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
import shlex
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from llamacpp_tuner.backends.base import (
    ServeOptions,
    UnsupportedOptionError,
    build_engine,
    run_process,
)
from llamacpp_tuner.messages import make_server
from llamacpp_tuner.parking import Parking

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


def _parking(package: Path, args: list[str]) -> Parking | None:
    """Parking for agents' chats, when the preset lets Loom move them to host RAM.

    Loom pins a chat only into the preset's checkpoint slots, and moves one to host
    RAM only with elastic memory (its default; profiles may pass fixed).
    """
    if "--pool_backing=fixed" in args:
        return None
    preset = next((a.split("=", 1)[1] for a in args if a.startswith("--preset=")), "")
    card = json.loads((package / "model.json").read_text())
    execution = card.get("presets", {}).get(preset, {}).get("execution", {})
    slots = execution.get("checkpoint_capacity", 0)
    # Two slots stay free: a turn pins its new state before releasing the old one.
    return Parking(limit=slots - 2) if slots > 2 else None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _warm(upstream: str, model: str) -> None:
    """Generate one token, so Loom activates the text model before any image request.

    An image as the first request activates the image component and the text model
    together; their two workspaces do not fit beside a 196K context on 24 GB.
    """
    body = {
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    request = urllib.request.Request(
        f"http://{upstream}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as r:
        r.read()


def _serve_when_ready(server: ThreadingHTTPServer, upstream: str, address: str) -> None:
    """Answer clients once Loom lists its model and has served one token, so 'ready' means ready."""
    while True:
        try:
            with urllib.request.urlopen(f"http://{upstream}/v1/models", timeout=5) as r:
                models = json.load(r).get("data")
                if models:
                    break
        except OSError:
            pass  # Loom is still starting
        time.sleep(1)
    try:
        _warm(upstream, models[0]["id"])
    except OSError as error:
        print(f"lct: Loom's warm-up request failed: {error}", flush=True)
    # address() below finds the public address in this line.
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
            "Build ours with 'lct setup --backend loom', or fetch one with loom-fetch."
        )
    binary = package.parent / "venv" / "bin" / "loom-serve"
    if not binary.is_file():
        raise FileNotFoundError(
            f"loom-serve is missing: install hrx-loom into the venv {binary.parents[1]}."
        )
    _allow_locking()
    upstream = f"127.0.0.1:{_free_port()}"
    server = make_server(host, port, upstream, _parking(package, args))
    threading.Thread(
        target=_serve_when_ready,
        args=(server, upstream, f"http://{host}:{port}"),
        daemon=True,
    ).start()
    command = [str(binary), f"--model={package}", f"--listen={upstream}", *args]
    run_process(command, env=env, cwd=package.parent)


def serve(model: str, options: ServeOptions, env: dict[str, str]) -> None:
    """Run Loom for a model package; MODEL is the package directory."""
    if options.picks_gguf:
        raise UnsupportedOptionError(
            "Loom's model, context and drafter are set in its package preset."
        )
    run_loom(
        Path(model).expanduser(),
        shlex.split(options.extra_args),
        env=env,
        host=options.host or "127.0.0.1",
        port=options.port or 8080,
    )


def setup(force: bool, cmake_args: tuple[str, ...], torch_index: str | None) -> str:
    return build_engine("loom", force)


def address(line: str) -> str:
    if line.startswith("lct: Loom at"):
        return line.split("listening on")[-1].strip()
    return ""
