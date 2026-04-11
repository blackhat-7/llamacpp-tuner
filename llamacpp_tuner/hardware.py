"""Hardware detection for GPU, CPU, and memory."""

import warnings
from dataclasses import dataclass

import psutil

from llamacpp_tuner.types import Backend

warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")


@dataclass(frozen=True)
class GPUInfo:
    name: str
    vram_mb: int
    compute_capability: tuple[int, int] | None = None


@dataclass(frozen=True)
class HardwareProfile:
    gpus: list[GPUInfo]
    cpu_cores: int
    cpu_threads: int
    ram_mb: int
    backend: Backend

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareProfile":
        """Reconstruct HardwareProfile from cached dict."""
        gpus = []
        for g in data.get("gpus", []):
            compute = tuple(g["compute"]) if g.get("compute") else None
            gpus.append(
                GPUInfo(
                    name=g["name"],
                    vram_mb=g["vram_mb"],
                    compute_capability=compute,
                )
            )

        return cls(
            gpus=gpus,
            cpu_cores=data["cpu_cores"],
            cpu_threads=data["cpu_threads"],
            ram_mb=data["ram_mb"],
            backend=data["backend"],
        )


def _detect_nvidia() -> list[GPUInfo]:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[GPUInfo] = []

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute = (cc[0], cc[1])
            except Exception:
                compute = None

            gpu_name = name.decode() if isinstance(name, bytes) else name
            vram_mb = int(mem.total) // (1024 * 1024)
            gpus.append(
                GPUInfo(
                    name=gpu_name,
                    vram_mb=vram_mb,
                    compute_capability=compute,
                )
            )

        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []


def _detect_backend(gpus: list[GPUInfo]) -> Backend:
    if gpus:
        return "cuda"

    import platform

    if platform.system() == "Darwin":
        return "metal"

    return "cpu"


def detect_hardware() -> HardwareProfile:
    gpus = _detect_nvidia()

    return HardwareProfile(
        gpus=gpus,
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        cpu_threads=psutil.cpu_count(logical=True) or 1,
        ram_mb=psutil.virtual_memory().total // (1024 * 1024),
        backend=_detect_backend(gpus),
    )


def format_hardware(profile: HardwareProfile) -> str:
    lines = [f"Backend: {profile.backend}"]

    for i, gpu in enumerate(profile.gpus):
        lines.append(f"GPU {i}: {gpu.name} ({gpu.vram_mb} MB VRAM)")
        if gpu.compute_capability:
            lines.append(
                f"  Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
            )

    lines.append(f"CPU: {profile.cpu_cores} cores, {profile.cpu_threads} threads")
    lines.append(f"RAM: {profile.ram_mb} MB")

    return "\n".join(lines)
