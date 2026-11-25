"""Tests for starting Loom on a model package."""

import errno
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from llamacpp_tuner.backends.loom import _parking, run_loom


def make_package(root):
    package = root / "Inkling-2-Mini"
    package.mkdir()
    (package / "model.json").write_text("{}")
    binary = root / "venv" / "bin" / "loom-serve"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    return package, binary


def start(package):
    """run_loom with nothing real running."""
    with (
        patch("llamacpp_tuner.backends.loom._without_io_uring") as without_io_uring,
        patch("llamacpp_tuner.backends.loom._free_port", return_value=41000),
        patch("llamacpp_tuner.backends.loom.make_server") as make_server,
        patch("llamacpp_tuner.backends.loom.threading.Thread"),
        patch("llamacpp_tuner.backends.loom.run_process") as run_process,
    ):
        run_loom(package, ["--preset=xtx"], env={"A": "1"}, host="0.0.0.0", port=6868)
    # Clients reach the translator on the public port; Loom listens privately.
    make_server.assert_called_once_with("0.0.0.0", 6868, "127.0.0.1:41000", None)
    return without_io_uring, run_process


def test_run_loom_serves_the_package_with_its_venv_without_io_uring(tmp_path):
    package, binary = make_package(tmp_path)

    without_io_uring, run_process = start(package)

    without_io_uring.assert_called_once_with()
    assert run_process.call_args.args[0] == [
        str(binary),
        f"--model={package}",
        "--listen=127.0.0.1:41000",
        "--preset=xtx",
    ]
    assert run_process.call_args.kwargs == {"env": {"A": "1"}, "cwd": tmp_path}


def test_the_programs_lct_starts_get_enosys_from_io_uring_setup_only():
    # The filter cannot be taken off, so a child process sets it and starts the probe,
    # as lct serve starts Loom.
    probe = (
        "import ctypes, os\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "result = libc.syscall(425, 1, ctypes.create_string_buffer(120))\n"
        "print(result, ctypes.get_errno(), os.getpid() > 0)\n"
    )
    parent = (
        "import subprocess, sys\n"
        "from llamacpp_tuner.backends.loom import _without_io_uring\n"
        "_without_io_uring()\n"
        f"subprocess.run([sys.executable, '-c', {probe!r}], check=True)\n"
    )

    out = subprocess.run(
        [sys.executable, "-c", parent], capture_output=True, text=True, check=True
    )

    assert out.stdout.split() == ["-1", str(errno.ENOSYS), "True"]


def test_run_loom_needs_a_package_and_loom_serve(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.json"):
        run_loom(tmp_path, [])
    package, binary = make_package(tmp_path)
    binary.unlink()
    with pytest.raises(FileNotFoundError, match="hrx-loom"):
        run_loom(package, [])


def test_parking_needs_checkpoint_slots_and_elastic_memory(tmp_path):
    card = {"presets": {"many": {"execution": {"checkpoint_capacity": 32}}, "none": {}}}
    (tmp_path / "model.json").write_text(json.dumps(card))

    parking = _parking(tmp_path, ["--preset=many", "--device=amdgpu"])

    assert parking is not None and parking.limit == 30
    assert _parking(tmp_path, ["--preset=many", "--pool_backing=fixed"]) is None
    assert _parking(tmp_path, ["--preset=none"]) is None
