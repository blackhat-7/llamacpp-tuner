"""Detached llama-server processes, one per profile, shared by the CLI and the TUI."""

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

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


def address(line: str) -> str:
    """The address a server announces in a log line, if this is that line."""
    if "listening on" in line:  # llama-server
        return line.split("listening on")[-1].strip()
    if line.startswith('{"serving": '):  # Jeeves
        return json.loads(line)["serving"]
    return ""


def url(name: str) -> str:
    """The server's announced address, empty while loading."""
    with (
        contextlib.suppress(FileNotFoundError),
        log_path(name).open(errors="replace") as log,
    ):
        for line in log:
            if found := address(line):
                return found
    return ""


class Usage(NamedTuple):
    cpu_seconds: float
    gpu_seconds: float
    ram: int
    vram: int


def _tree(pid: int) -> list[int]:
    """A server's process and its descendants (lct, then llama-server)."""
    found, pending = [], [pid]
    while pending:
        current = pending.pop()
        found.append(current)
        with contextlib.suppress(OSError):
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
            pending += [int(child) for child in children.split()]
    return found


def usage(pid: int) -> Usage:
    """CPU and GPU time used so far, resident RAM and VRAM of a server's processes."""
    cpu = gpu = ram = vram = 0.0
    for current in _tree(pid):
        with contextlib.suppress(OSError):
            # Fields after the ')' that ends the command name; utime and stime are 12, 13.
            stat = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
            cpu += (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS"):  # absent for zombies
                    ram += int(line.split()[1]) * 1024
            # amdgpu reports each client's VRAM and busy time per engine (ns) in fdinfo;
            # several fds can share a client.
            clients: dict[str, tuple[int, int]] = {}
            for fdinfo in Path(f"/proc/{current}/fdinfo").iterdir():
                with contextlib.suppress(OSError):
                    info = dict(
                        line.split(":", 1)
                        for line in fdinfo.read_text().splitlines()
                        if ":" in line
                    )
                    if "drm-memory-vram" in info:
                        size = int(info["drm-memory-vram"].split()[0]) * 1024
                        busy = sum(
                            int(value.split()[0])
                            for key, value in info.items()
                            if key.startswith("drm-engine-")
                        )
                        clients[info.get("drm-client-id", fdinfo.name)] = (size, busy)
            vram += sum(size for size, _ in clients.values())
            gpu += sum(busy for _, busy in clients.values()) / 1e9
    return Usage(cpu, gpu, int(ram), int(vram))


def _temperature(hwmon: Path, label: str) -> int | None:
    """°C of the hwmon sensor with this label, if present."""
    for sensor in hwmon.glob("temp*_label"):
        with contextlib.suppress(OSError, ValueError):
            if sensor.read_text().strip() == label:
                return (
                    int(
                        sensor.with_name(
                            sensor.name.replace("label", "input")
                        ).read_text()
                    )
                    // 1000
                )
    return None


def system() -> dict[str, int]:
    """Whole-machine CPU ticks, RAM and temperature and, for the first GPU that
    reports them, VRAM, load and hotspot temperature."""
    cpu = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
    meminfo = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    found = {
        "cpu_busy": sum(map(int, cpu))
        - int(cpu[3])
        - int(cpu[4]),  # minus idle, iowait
        "cpu_total": sum(map(int, cpu)),
        "ram_used": meminfo["MemTotal"] - meminfo["MemAvailable"],
        "ram_total": meminfo["MemTotal"],
    }
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        with contextlib.suppress(OSError, ValueError):
            found["vram_used"] = int((device / "mem_info_vram_used").read_text())
            found["vram_total"] = int((device / "mem_info_vram_total").read_text())
            found["gpu_busy"] = int((device / "gpu_busy_percent").read_text())
            # The hotspot (junction) is what throttles, not the edge sensor.
            for hwmon in device.glob("hwmon/hwmon*"):
                if (celsius := _temperature(hwmon, "junction")) is not None:
                    found["gpu_temp"] = celsius
            break
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        with contextlib.suppress(OSError):
            is_cpu = (hwmon / "name").read_text().strip() == "coretemp"
            if is_cpu and (celsius := _temperature(hwmon, "Package id 0")) is not None:
                found["cpu_temp"] = celsius
    return found
