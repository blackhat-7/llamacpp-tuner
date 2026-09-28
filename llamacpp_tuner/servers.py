"""Detached llama-server processes, one per profile, shared by the CLI and the TUI."""

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from llamacpp_tuner.cache import get_servers_dir

LCT = [sys.executable, "-m", "llamacpp_tuner.cli"]


def log_path(name: str) -> Path:
    return get_servers_dir() / f"{name}.log"


def _state_path(name: str) -> Path:
    return get_servers_dir() / f"{name}.json"


def child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}


def alive(pid: int) -> bool:
    # Reap a server this process started, or a finished one lingers as a zombie.
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def interrupt(pid: int) -> None:
    """SIGINT the process group, like Ctrl-C; llama-server ignores a plain SIGTERM."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGINT)


def running() -> dict[str, int]:
    """Profile name to pid for every server still alive; stale state is removed."""
    found = {}
    for path in sorted(get_servers_dir().glob("*.json")):
        pid = json.loads(path.read_text())["pid"]
        if alive(pid):
            found[path.stem] = pid
        else:
            path.unlink(missing_ok=True)
    return found


def start(name: str) -> int:
    """Start `lct serve NAME` in its own session so it outlives the caller."""
    if name in running():
        raise RuntimeError(f"{name} is already running.")
    cmd = [*LCT, "serve", name]
    with log_path(name).open("w") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env(),
            start_new_session=True,
        )
    _state_path(name).write_text(json.dumps({"pid": proc.pid}))
    return proc.pid


def url(name: str) -> str:
    """The address from llama-server's 'listening on' line, empty while loading."""
    with (
        contextlib.suppress(FileNotFoundError),
        log_path(name).open(errors="replace") as log,
    ):
        for line in log:
            if "listening on" in line:
                return line.split("listening on")[-1].strip()
    return ""


def memory(pid: int) -> int:
    """Resident bytes of a server's process tree (lct plus llama-server)."""
    total, pending = 0, [pid]
    while pending:
        current = pending.pop()
        with contextlib.suppress(OSError):
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS"):  # absent for zombies
                    total += int(line.split()[1]) * 1024
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
            pending += [int(child) for child in children.split()]
    return total
