"""Calculate optimal llama.cpp arguments based on hardware and model."""

import re
from dataclasses import dataclass

from llamacpp_tuner.constants import (
    BATCH_SIZE_CPU,
    BYTES_PER_PARAM_IQ4_XS,
    BYTES_PER_PARAM_Q4_K_M,
    BYTES_PER_PARAM_Q4_K_S,
    BYTES_PER_PARAM_Q5_K_M,
    BYTES_PER_PARAM_Q5_K_S,
    BYTES_PER_PARAM_Q8_0,
    NUMA_CORE_THRESHOLD,
    RAM_MODEL_LOCK_FACTOR,
    RAM_MODEL_MMAP_FACTOR,
    UBATCH_SIZE_CPU,
    VRAM_HEADROOM_FACTOR,
    VRAM_USAGE_FACTOR,
)
from llamacpp_tuner.hardware import HardwareProfile
from llamacpp_tuner.types import Quant

BYTES_PER_PARAM: dict[Quant, float] = {
    "Q4_K_M": BYTES_PER_PARAM_Q4_K_M,
    "Q4_K_S": BYTES_PER_PARAM_Q4_K_S,
    "Q5_K_M": BYTES_PER_PARAM_Q5_K_M,
    "Q5_K_S": BYTES_PER_PARAM_Q5_K_S,
    "Q8_0": BYTES_PER_PARAM_Q8_0,
    "IQ4_XS": BYTES_PER_PARAM_IQ4_XS,
}

# Model size estimates (in billions)
MODEL_PARAMS: dict[str, float] = {
    "0.5b": 0.5,
    "1b": 1.0,
    "1.5b": 1.5,
    "2b": 2.0,
    "3b": 3.0,
    "4b": 4.0,
    "7b": 7.0,
    "8b": 8.0,
    "9b": 9.0,
    "14b": 14.0,
    "27b": 27.0,
    "32b": 32.0,
    "35b": 35.0,
    "70b": 70.0,
}

# Default layer counts for different model sizes
MODEL_LAYERS: dict[str, int] = {
    "0.5b": 24,
    "1b": 24,
    "2b": 24,
    "3b": 28,
    "4b": 32,
    "7b": 32,
    "8b": 32,
    "9b": 48,
    "14b": 48,
    "27b": 48,
    "32b": 64,
    "35b": 64,
    "70b": 80,
}


@dataclass(frozen=True)
class OptimalArgs:
    """Optimized llama.cpp server arguments."""

    # Core performance args
    ctx_size: int
    n_gpu_layers: int
    batch_size: int
    ubatch_size: int
    threads: int
    threads_batch: int
    parallel: int

    # GPU/parallelism
    flash_attn: bool
    tensor_split: str | None
    split_mode: str  # "layer" or "row"
    main_gpu: int

    # Memory
    mlock: bool
    mmap: bool
    offload_kqv: bool

    # Optimization flags
    numa: bool

    def to_list(self, model_path: str) -> list[str]:
        """Convert to llama-server command-line arguments."""
        has_gpu = self.n_gpu_layers != 0

        args = [
            "-m",
            model_path,
            "-c",
            str(self.ctx_size),
        ]

        # GPU layers: use 'auto' for partial offload, 'all' for full offload
        if has_gpu:
            if self.n_gpu_layers == -1:
                args.extend(["-ngl", "all"])
            elif self.n_gpu_layers == -2:
                args.extend(["-ngl", "auto"])
            else:
                args.extend(["-ngl", str(self.n_gpu_layers)])

        args.extend(
            [
                "-b",
                str(self.batch_size),
                "-ub",
                str(self.ubatch_size),
                "-t",
                str(self.threads),
                "-tb",
                str(self.threads_batch),
                "-np",
                str(self.parallel),
            ]
        )

        # Only add GPU-specific args if we have GPU support
        if has_gpu:
            args.extend(["--split-mode", self.split_mode])
            args.extend(["-mg", str(self.main_gpu)])
            if self.flash_attn:
                args.extend(["--flash-attn", "on"])
            if self.offload_kqv:
                args.append("-kvo")
            if self.tensor_split:
                args.extend(["-ts", self.tensor_split])

        if self.mlock:
            args.append("--mlock")
        if not self.mmap:
            args.append("--no-mmap")
        if self.numa:
            args.append("--numa")

        return args


def _estimate_params(model_name: str) -> float:
    lowered = model_name.lower()
    # Try regex first - more accurate for extracting number before B/b
    match = re.search(r"(\d+\.?\d*)[Bb](?:[^a-z]|$)", model_name)
    if match:
        return float(match.group(1))
    # Fallback to known model sizes
    for key, value in sorted(
        MODEL_PARAMS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if key in lowered:
            return value
    return 7.0


def _estimate_layers(model_name: str) -> int:
    lowered = model_name.lower()
    # Sort keys by length descending to match longer patterns first
    for key, value in sorted(
        MODEL_LAYERS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if key in lowered:
            return value
    # Default based on param count
    params = _estimate_params(model_name)
    if params <= 4:
        return 32
    elif params <= 14:
        return 48
    elif params <= 32:
        return 64
    return 80


def _estimate_vram_usage(
    params_b: float, quant: Quant, ctx_size: int, layers: int
) -> float:
    """Estimate VRAM usage in MB."""
    model_size = params_b * 1_000_000_000 * BYTES_PER_PARAM[quant] / (1024 * 1024)
    # KV cache: 2 (K+V) * layers * ctx * 128 (head_dim) * 2 (bytes per fp16)
    kv_cache = 2 * layers * ctx_size * 128 * 2 / (1024 * 1024)
    # Add overhead for activations, buffers (estimate 10%)
    overhead = model_size * 0.1
    return model_size + kv_cache + overhead


def _calculate_batch_sizes(
    vram_mb: int, has_gpu: bool, available_vram_mb: float | None = None
) -> tuple[int, int]:
    """Calculate optimal batch and micro-batch sizes.

    Args:
        vram_mb: Total GPU VRAM in MB
        has_gpu: Whether GPU is available
        available_vram_mb: Available VRAM after model load (optional)

    Returns:
        Tuple of (batch_size, ubatch_size)
    """
    if not has_gpu:
        return BATCH_SIZE_CPU, UBATCH_SIZE_CPU

    # Use available VRAM if provided, otherwise use total
    effective_vram = available_vram_mb if available_vram_mb else vram_mb

    # Scale batch size based on available VRAM
    # Larger batches improve GPU utilization but need more memory
    if effective_vram > 6000:
        batch_size = 4096
        ubatch_size = 1024
    elif effective_vram > 3000:
        batch_size = 2048
        ubatch_size = 512
    elif effective_vram > 1500:
        batch_size = 1024
        ubatch_size = 256
    else:
        batch_size = 512
        ubatch_size = 128

    return batch_size, ubatch_size


def _calculate_threads(
    cpu_cores: int, cpu_threads: int, has_gpu: bool
) -> tuple[int, int]:
    """Calculate optimal thread counts."""
    threads = max(4, cpu_cores // 2) if has_gpu else cpu_cores

    # Batch threads: can be higher for prompt processing
    threads_batch = min(cpu_threads, threads + 4)

    return threads, threads_batch


def calculate_optimal_args(
    hardware: HardwareProfile,
    model_name: str,
    quant: Quant,
    ctx_size: int,
    llama_has_gpu: bool = True,
) -> tuple[OptimalArgs, list[str]]:
    """Calculate optimal llama.cpp arguments for given hardware and model."""
    warnings: list[str] = []
    params = _estimate_params(model_name)
    n_layers = _estimate_layers(model_name)

    total_vram = sum(g.vram_mb for g in hardware.gpus) if hardware.gpus else 0
    # has_gpu requires both hardware detection AND llama.cpp built with GPU support
    has_gpu = llama_has_gpu and total_vram > 0 and hardware.backend == "cuda"

    # Estimate if model fits in VRAM
    estimated = _estimate_vram_usage(params, quant, ctx_size, n_layers)

    # Calculate model size in MB
    model_size_mb = params * 1_000_000_000 * BYTES_PER_PARAM[quant] / (1024 * 1024)

    # Calculate GPU layers based on VRAM availability
    n_gpu_layers = 0
    if has_gpu:
        # Use full VRAM estimate (model + KV cache + overhead)
        total_vram_needed = estimated

        # Apply VRAM usage factor to get usable amount
        usable_vram = total_vram * VRAM_USAGE_FACTOR

        if total_vram_needed <= usable_vram:
            n_gpu_layers = -1  # All layers fit
        else:
            # Use 'auto' to let llama.cpp fit layers to available VRAM
            n_gpu_layers = -2  # Special value meaning 'auto'
            warnings.append(
                f"Model exceeds VRAM ({total_vram_needed:.0f}MB needed, {usable_vram:.0f}MB usable). "
                f"Using auto GPU layer offloading."
            )

    # Calculate available VRAM after model load
    layers_on_gpu = n_layers if n_gpu_layers == -1 else n_gpu_layers

    available_vram_mb = (
        total_vram - (model_size_mb * layers_on_gpu / n_layers)
        if has_gpu and n_layers > 0
        else 0
    )

    # Batch sizes based on available VRAM (not total)
    batch_size, ubatch_size = _calculate_batch_sizes(
        total_vram, has_gpu, available_vram_mb
    )

    # Thread counts
    threads, threads_batch = _calculate_threads(
        hardware.cpu_cores, hardware.cpu_threads, has_gpu
    )

    # Parallel sequences: default to 1 (single user)
    parallel = 1

    # Flash attention: requires CUDA compute capability >= 7.0
    flash_attn = False
    if has_gpu:
        for gpu in hardware.gpus:
            if gpu.compute_capability and gpu.compute_capability[0] >= 7:
                flash_attn = True
                break

    # Memory settings
    mlock = has_gpu and hardware.ram_mb > model_size_mb * RAM_MODEL_LOCK_FACTOR
    mmap = hardware.ram_mb > model_size_mb * RAM_MODEL_MMAP_FACTOR

    offload_kqv = has_gpu and estimated < total_vram * VRAM_HEADROOM_FACTOR

    # Multi-GPU settings (only for CUDA builds with GPU support)
    tensor_split = None
    split_mode = "layer"  # Default: split by layers
    main_gpu = 0

    if has_gpu and len(hardware.gpus) > 1:
        ratios = [g.vram_mb / total_vram for g in hardware.gpus]
        tensor_split = ",".join(f"{r:.2f}" for r in ratios)
        # Use row splitting for better performance on multi-GPU
        split_mode = "row"

    numa = hardware.cpu_cores > NUMA_CORE_THRESHOLD and not has_gpu

    args = OptimalArgs(
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        threads=threads,
        threads_batch=threads_batch,
        parallel=parallel,
        flash_attn=flash_attn if has_gpu else False,
        tensor_split=tensor_split if has_gpu else None,
        split_mode=split_mode if has_gpu else "layer",
        main_gpu=main_gpu if has_gpu else 0,
        mlock=mlock,
        mmap=mmap,
        offload_kqv=offload_kqv if has_gpu else False,
        numa=numa,
    )

    return args, warnings


def format_args(args: OptimalArgs) -> str:
    """Format optimal args for display."""
    if args.n_gpu_layers == -1:
        gpu_layers_str = "all"
    elif args.n_gpu_layers == -2:
        gpu_layers_str = "auto"
    else:
        gpu_layers_str = str(args.n_gpu_layers)

    lines = [
        f"Context size: {args.ctx_size}",
        f"GPU layers: {gpu_layers_str}",
        f"Batch size: {args.batch_size}",
        f"Micro-batch: {args.ubatch_size}",
        f"Threads: {args.threads}",
        f"Threads (batch): {args.threads_batch}",
        f"Parallel sequences: {args.parallel}",
        f"Flash attention: {args.flash_attn}",
        f"Memory lock: {args.mlock}",
        f"Memory map: {args.mmap}",
        f"Offload KQV: {args.offload_kqv}",
        f"Split mode: {args.split_mode}",
    ]
    if args.tensor_split:
        lines.append(f"Tensor split: {args.tensor_split}")
    if args.numa:
        lines.append("NUMA: enabled")
    return "\n".join(lines)
