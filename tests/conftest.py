"""Shared test fixtures."""

import pytest

from llamacpp_tuner.hardware import GPUInfo, HardwareProfile


@pytest.fixture
def cpu_only_hardware():
    """Hardware profile with no GPUs."""
    return HardwareProfile(
        gpus=[],
        cpu_cores=8,
        cpu_threads=16,
        ram_mb=32768,
        backend="cpu",
    )


@pytest.fixture
def single_gpu_hardware():
    """Hardware profile with single NVIDIA GPU."""
    return HardwareProfile(
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 4070",
                vram_mb=12288,
                compute_capability=(8, 9),
            ),
        ],
        cpu_cores=8,
        cpu_threads=16,
        ram_mb=32768,
        backend="cuda",
    )


@pytest.fixture
def multi_gpu_hardware():
    """Hardware profile with multiple GPUs."""
    return HardwareProfile(
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 4090",
                vram_mb=24576,
                compute_capability=(8, 9),
            ),
            GPUInfo(
                name="NVIDIA GeForce RTX 4090",
                vram_mb=24576,
                compute_capability=(8, 9),
            ),
        ],
        cpu_cores=16,
        cpu_threads=32,
        ram_mb=65536,
        backend="cuda",
    )
