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


def test_run_server_requires_binary(monkeypatch):
    monkeypatch.setattr(llama, "get_llama_binary", lambda: None)

    with pytest.raises(FileNotFoundError):
        llama.run_server([])
