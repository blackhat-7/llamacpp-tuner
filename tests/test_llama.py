"""Tests for llama-server discovery and execution."""

import io
import json
import shutil
import tarfile
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


SERVER = "llama-b1/llama-server"


def fake_release(monkeypatch, tmp_path, archives: dict[str, dict[str, str]], gpu=None):
    """Publish one release whose tarballs hold {member path: shell script}."""
    for name, files in archives.items():
        with tarfile.open(tmp_path / name, "w:gz") as tar:
            for member, script in files.items():
                data = f"#!/bin/sh\n{script}\n".encode()
                info = tarfile.TarInfo(member)
                info.size, info.mode = len(data), 0o755
                tar.addfile(info, io.BytesIO(data))
    assets = [{"name": n, "browser_download_url": f"https://x/{n}"} for n in archives]
    monkeypatch.setattr(llama.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        llama.shutil, "which", lambda name: gpu if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(llama, "get_llama_dir", lambda: tmp_path / "llama")
    (tmp_path / "llama").mkdir()
    monkeypatch.setattr(
        llama.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(json.dumps([{"assets": assets}]).encode()),
    )
    monkeypatch.setattr(
        llama.urllib.request,
        "urlretrieve",
        lambda url, path: shutil.copy(tmp_path / url.rsplit("/", 1)[-1], path),
    )


VULKAN = "llama-b1-bin-ubuntu-vulkan-x64.tar.gz"
CUDA = "llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz"
CUDART = "cudart-llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz"


def test_download_release_installs_vulkan_without_nvidia(monkeypatch, tmp_path):
    fake_release(
        monkeypatch,
        tmp_path,
        {VULKAN: {SERVER: "echo vulkan"}, CUDA: {SERVER: "echo cuda"}},
    )

    binary = llama.download_release()

    assert binary == tmp_path / "llama" / "build" / "bin" / "llama-server"
    assert llama.subprocess.run([binary], capture_output=True).stdout == b"vulkan\n"


def test_download_release_puts_the_cuda_runtime_beside_llama_on_nvidia(
    monkeypatch, tmp_path
):
    fake_release(
        monkeypatch,
        tmp_path,
        {
            VULKAN: {SERVER: "echo vulkan"},
            CUDA: {SERVER: "echo cuda"},
            CUDART: {"cudart/libcudart.so.12": ""},
        },
        gpu="/usr/bin/nvidia-smi",
    )

    binary = llama.download_release()

    assert llama.subprocess.run([binary], capture_output=True).stdout == b"cuda\n"
    assert (binary.parent / "libcudart.so.12").is_file()


def test_download_release_fails_when_the_server_cannot_start(monkeypatch, tmp_path):
    fake_release(
        monkeypatch,
        tmp_path,
        {VULKAN: {SERVER: "echo 'libvulkan.so.1: not found' >&2; exit 127"}},
    )

    with pytest.raises(RuntimeError, match="(?s)libvulkan.so.1.*GPU driver"):
        llama.download_release()


@pytest.mark.parametrize(
    ("cmake_args", "expected"), [((), "download"), (("-DX=1",), "build")]
)
def test_install_downloads_unless_given_cmake_args(monkeypatch, cmake_args, expected):
    monkeypatch.setattr(llama, "get_llama_binary", lambda: None)
    monkeypatch.setattr(llama, "download_release", lambda: "download")
    monkeypatch.setattr(llama, "build_from_source", lambda **kwargs: "build")

    assert llama.install_llama(extra_cmake_args=cmake_args) == expected


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
    monkeypatch.setattr(llama.subprocess, "Popen", lambda command, **kwargs: server)

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

    monkeypatch.setattr(
        llama.subprocess, "Popen", lambda command, **kwargs: FakeServer()
    )

    with pytest.raises(llama.subprocess.CalledProcessError):
        llama.run_server([])
