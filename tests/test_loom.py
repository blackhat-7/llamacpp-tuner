"""Tests for starting Loom on a model package."""

import resource
from unittest.mock import patch

import pytest

from llamacpp_tuner.loom import run_loom


def make_package(root):
    package = root / "Inkling-2-Mini"
    package.mkdir()
    (package / "model.json").write_text("{}")
    binary = root / "venv" / "bin" / "loom-serve"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    return package, binary


def test_run_loom_serves_the_package_with_its_venv(tmp_path):
    package, binary = make_package(tmp_path)
    unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    with (
        patch("llamacpp_tuner.loom.resource.getrlimit", return_value=unlimited),
        patch("llamacpp_tuner.loom.run_process") as run_process,
    ):
        run_loom(package, ["--preset=xtx"], env={"A": "1"})

    assert run_process.call_args.args[0] == [
        str(binary),
        f"--model={package}",
        "--preset=xtx",
    ]
    assert run_process.call_args.kwargs == {"env": {"A": "1"}, "cwd": tmp_path}


def test_run_loom_needs_a_package_and_loom_serve(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.json"):
        run_loom(tmp_path, [])
    package, binary = make_package(tmp_path)
    binary.unlink()
    with pytest.raises(FileNotFoundError, match="hrx-loom"):
        run_loom(package, [])


def test_run_loom_refuses_the_default_memory_lock_limit(tmp_path):
    package, _ = make_package(tmp_path)
    eight_mib = (8 * 1024**2, 8 * 1024**2)
    with (
        patch("llamacpp_tuner.loom.resource.getrlimit", return_value=eight_mib),
        patch("llamacpp_tuner.loom.run_process") as run_process,
        pytest.raises(RuntimeError, match="limits.d"),
    ):
        run_loom(package, [])

    run_process.assert_not_called()
