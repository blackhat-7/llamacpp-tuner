"""Tests for hardware detection module."""


from llamacpp_tuner.hardware import (
    GPUInfo,
    HardwareProfile,
    _detect_backend,
    format_hardware,
)


class TestGPUInfo:
    def test_gpu_with_compute_capability(self):
        gpu = GPUInfo(
            name="RTX 4090",
            vram_mb=24576,
            compute_capability=(8, 9),
        )
        assert gpu.name == "RTX 4090"
        assert gpu.vram_mb == 24576
        assert gpu.compute_capability == (8, 9)

    def test_gpu_without_compute_capability(self):
        gpu = GPUInfo(
            name="Unknown GPU",
            vram_mb=4096,
            compute_capability=None,
        )
        assert gpu.compute_capability is None


class TestHardwareProfile:
    def test_cpu_only_profile(self):
        profile = HardwareProfile(
            gpus=[],
            cpu_cores=8,
            cpu_threads=16,
            ram_mb=32768,
            backend="cpu",
        )
        assert len(profile.gpus) == 0
        assert profile.backend == "cpu"

    def test_single_gpu_profile(self):
        profile = HardwareProfile(
            gpus=[GPUInfo("RTX 4070", 12288, (8, 9))],
            cpu_cores=8,
            cpu_threads=16,
            ram_mb=32768,
            backend="cuda",
        )
        assert len(profile.gpus) == 1
        assert profile.backend == "cuda"


class TestFormatHardware:
    def test_cpu_only_formatting(self, cpu_only_hardware):
        output = format_hardware(cpu_only_hardware)
        assert "Backend: cpu" in output
        assert "CPU: 8 cores, 16 threads" in output
        assert "RAM: 32768 MB" in output

    def test_gpu_formatting_includes_compute(self, single_gpu_hardware):
        output = format_hardware(single_gpu_hardware)
        assert "RTX 4070" in output
        assert "12288 MB VRAM" in output
        assert "Compute: 8.9" in output

    def test_multi_gpu_formatting(self, multi_gpu_hardware):
        output = format_hardware(multi_gpu_hardware)
        assert output.count("GPU 0:") == 1
        assert output.count("GPU 1:") == 1


class TestDetectBackend:
    def test_returns_cuda_for_nvidia_gpus(self):
        gpus = [GPUInfo("RTX 4090", 24576, (8, 9))]
        assert _detect_backend(gpus) == "cuda"

    def test_returns_cpu_for_empty_gpu_list(self):
        assert _detect_backend([]) == "cpu"
