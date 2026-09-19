"""Tests for llama-server discovery and execution."""

from pathlib import Path

import pytest

from llamacpp_tuner import llama


def test_finds_system_binary_without_touching_cache(monkeypatch):
    monkeypatch.setattr(llama.shutil, "which", lambda name: "/usr/bin/llama-server")
    monkeypatch.setattr(
        llama,
        "get_llama_dir",
        lambda: (_ for _ in ()).throw(AssertionError("cache accessed")),
    )

    assert llama.get_llama_binary() == Path("/usr/bin/llama-server")


def test_finds_managed_binary(monkeypatch, tmp_path):
    binary = tmp_path / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(llama.shutil, "which", lambda name: None)
    monkeypatch.setattr(llama, "get_llama_dir", lambda: tmp_path)

    assert llama.get_llama_binary() == binary


def test_finds_windows_release_binary(monkeypatch, tmp_path):
    binary = tmp_path / "build" / "bin" / "Release" / "llama-server.exe"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(llama.platform, "system", lambda: "Windows")
    monkeypatch.setattr(llama.shutil, "which", lambda name: None)
    monkeypatch.setattr(llama, "get_llama_dir", lambda: tmp_path)

    assert llama.get_llama_binary() == binary


def test_build_passes_backend_arguments(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    binary = tmp_path / "build" / "bin" / "llama-server"
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool):
        commands.append(command)
        if "--build" in command:
            binary.parent.mkdir(parents=True)
            binary.touch()

    monkeypatch.setattr(llama, "get_llama_dir", lambda: tmp_path)
    monkeypatch.setattr(
        llama.shutil,
        "which",
        lambda name: "/opt/cuda/bin/nvcc" if name == "nvcc" else None,
    )
    monkeypatch.setattr(llama.subprocess, "run", fake_run)

    assert llama.build_from_source(extra_cmake_args=["-DGGML_VULKAN=ON"]) == binary
    assert "-DGGML_CUDA=ON" in commands[0]
    assert "-DGGML_VULKAN=ON" in commands[0]


def test_build_enables_vulkan_without_cuda(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    binary = tmp_path / "build" / "bin" / "llama-server"
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool):
        commands.append(command)
        if "--build" in command:
            binary.parent.mkdir(parents=True)
            binary.touch()

    monkeypatch.setattr(llama, "get_llama_dir", lambda: tmp_path)
    monkeypatch.setattr(
        llama.shutil,
        "which",
        lambda name: "/usr/bin/vulkaninfo" if name == "vulkaninfo" else None,
    )
    monkeypatch.setattr(llama.subprocess, "run", fake_run)

    assert llama.build_from_source() == binary
    assert "-DGGML_VULKAN=ON" in commands[0]
    assert "-DGGML_CUDA=ON" not in commands[0]


def test_run_server_requires_binary(monkeypatch):
    monkeypatch.setattr(llama, "get_llama_binary", lambda: None)

    with pytest.raises(FileNotFoundError):
        llama.run_server([])


def test_run_server_terminates_child_on_signal(monkeypatch):
    handlers = {}
    monkeypatch.setattr(
        llama, "get_llama_binary", lambda: Path("/usr/bin/llama-server")
    )
    monkeypatch.setattr(
        llama.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler)
    )

    class FakeServer:
        def __init__(self):
            self.terminated = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def terminate(self):
            self.terminated = True

        def wait(self):
            handlers[llama.signal.SIGTERM](llama.signal.SIGTERM, None)
            return -15

    server = FakeServer()
    monkeypatch.setattr(llama.subprocess, "Popen", lambda command: server)

    llama.run_server([])

    assert server.terminated


def test_run_server_reports_server_failure(monkeypatch):
    monkeypatch.setattr(
        llama, "get_llama_binary", lambda: Path("/usr/bin/llama-server")
    )
    monkeypatch.setattr(llama.signal, "signal", lambda sig, handler: None)

    class FakeServer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def wait(self):
            return 1

    monkeypatch.setattr(llama.subprocess, "Popen", lambda command: FakeServer())

    with pytest.raises(llama.subprocess.CalledProcessError):
        llama.run_server([])
