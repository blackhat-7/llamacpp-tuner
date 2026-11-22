"""Tests for starting Loom on a model package."""

import os
import resource
import subprocess
from unittest.mock import patch

import pytest

from llamacpp_tuner.backends.loom import run_loom

MIB = 1024**2


def make_package(root):
    package = root / "Inkling-2-Mini"
    package.mkdir()
    (package / "model.json").write_text("{}")
    binary = root / "venv" / "bin" / "loom-serve"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    return package, binary


def start(package, limits, sudo=None):
    """run_loom with the given (soft, hard) memory-lock limits; nothing real runs."""
    with (
        patch("llamacpp_tuner.backends.loom.resource.getrlimit", return_value=limits),
        patch("llamacpp_tuner.backends.loom.resource.setrlimit") as setrlimit,
        patch("llamacpp_tuner.backends.loom.subprocess.run", side_effect=sudo) as run,
        patch("llamacpp_tuner.backends.loom._free_port", return_value=41000),
        patch("llamacpp_tuner.backends.loom.make_server") as make_server,
        patch("llamacpp_tuner.backends.loom.threading.Thread"),
        patch("llamacpp_tuner.backends.loom.run_process") as run_process,
    ):
        run_loom(package, ["--preset=xtx"], env={"A": "1"}, host="0.0.0.0", port=6868)
    # Clients reach the translator on the public port; Loom listens privately.
    make_server.assert_called_once_with("0.0.0.0", 6868, "127.0.0.1:41000")
    return setrlimit, run, run_process


def test_run_loom_serves_the_package_with_its_venv(tmp_path):
    package, binary = make_package(tmp_path)

    setrlimit, sudo, run_process = start(package, (8 * MIB, 64 * MIB))

    # A high enough ceiling needs no sudo: only this server's soft limit goes up.
    setrlimit.assert_called_once_with(resource.RLIMIT_MEMLOCK, (64 * MIB, 64 * MIB))
    sudo.assert_not_called()
    assert run_process.call_args.args[0] == [
        str(binary),
        f"--model={package}",
        "--listen=127.0.0.1:41000",
        "--preset=xtx",
    ]
    assert run_process.call_args.kwargs == {"env": {"A": "1"}, "cwd": tmp_path}


def test_run_loom_asks_sudo_to_raise_its_own_low_ceiling(tmp_path):
    package, _ = make_package(tmp_path)

    _, sudo, run_process = start(package, (8 * MIB, 8 * MIB))

    sudo.assert_called_once_with(
        ["sudo", "prlimit", "--pid", str(os.getpid()), "--memlock=67108864:67108864"],
        check=True,
    )
    run_process.assert_called_once()


def test_run_loom_stops_when_sudo_is_refused(tmp_path):
    package, _ = make_package(tmp_path)
    refused = subprocess.CalledProcessError(1, "sudo")

    with pytest.raises(RuntimeError, match="refused"):
        start(package, (8 * MIB, 8 * MIB), sudo=refused)


def test_run_loom_needs_a_package_and_loom_serve(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.json"):
        run_loom(tmp_path, [])
    package, binary = make_package(tmp_path)
    binary.unlink()
    with pytest.raises(FileNotFoundError, match="hrx-loom"):
        run_loom(package, [])
