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

# KV cache bytes per element by type (relative to f16 = 2 bytes)
KV_BYTES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
    "q4_1": 0.5,
    "q5_0": 0.625,
    "q5_1": 0.625,
    "iq4_nl": 0.5,
}

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

    ctx_size: int
    n_gpu_layers: int
    tensor_split: str | None
    split_mode: str
    mlock: bool
    mmap: bool
    numa: bool
    cache_type_k: str
    cache_type_v: str
    flash_attn: str  # "on", "off", or "auto"
    batch_size: int | None  # None = use llama.cpp default
    ubatch_size: int | None  # None = use llama.cpp default

    def to_list(self, model_path: str) -> list[str]:
        """Convert to llama-server command-line arguments."""
        args = [
            "-m",
            model_path,
            "-c",
            str(self.ctx_size),
        ]

        if self.n_gpu_layers != 0:
            if self.n_gpu_layers == -1:
                args.extend(["-ngl", "all"])
            elif self.n_gpu_layers == -2:
                args.extend(["-ngl", "auto"])
            else:
                args.extend(["-ngl", str(self.n_gpu_layers)])

        if self.tensor_split:
            args.extend(["-ts", self.tensor_split])

        if self.split_mode != "layer":
            args.extend(["--split-mode", self.split_mode])

        if self.flash_attn != "auto":
            args.extend(["-fa", self.flash_attn])

        if self.cache_type_k != "f16":
            args.extend(["-ctk", self.cache_type_k])
        if self.cache_type_v != "f16":
            args.extend(["-ctv", self.cache_type_v])

        if self.batch_size is not None:
            args.extend(["-b", str(self.batch_size)])
        if self.ubatch_size is not None:
            args.extend(["-ub", str(self.ubatch_size)])

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
    params_b: float,
    quant: Quant,
    ctx_size: int,
    layers: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
) -> float:
    """Estimate VRAM usage in MB."""
    model_size = params_b * 1_000_000_000 * BYTES_PER_PARAM[quant] / (1024 * 1024)
    # KV cache: 2 (K+V) * layers * ctx * 128 (head_dim) * bytes_per_element
    kv_bytes_k = KV_BYTES.get(cache_type_k, 2.0)
    kv_bytes_v = KV_BYTES.get(cache_type_v, 2.0)
    kv_cache = layers * ctx_size * 128 * (kv_bytes_k + kv_bytes_v) / (1024 * 1024)
    # Add overhead for activations, buffers (estimate 10%)
    overhead = model_size * 0.1
    return model_size + kv_cache + overhead


def calculate_optimal_args(
    hardware: HardwareProfile,
    model_name: str,
    quant: Quant,
    ctx_size: int,
    llama_has_gpu: bool = True,
) -> tuple[OptimalArgs, list[str]]:
    """Calculate optimal llama.cpp arguments for given hardware and model.

    Strategy: llama.cpp defaults are already near-optimal for GPU inference.
    Only override for specific scenarios (CPU, multi-GPU, memory-constrained).
    """
    warnings: list[str] = []
    params = _estimate_params(model_name)
    n_layers = _estimate_layers(model_name)

    total_vram = sum(g.vram_mb for g in hardware.gpus) if hardware.gpus else 0
    has_gpu = llama_has_gpu and total_vram > 0 and hardware.backend == "cuda"

    n_gpu_layers = 0

    mlock = False
    mmap = True
    tensor_split = None
    split_mode = "layer"
    numa = hardware.cpu_cores > NUMA_CORE_THRESHOLD and not has_gpu

    # KV cache quantization: q8_0 on GPU (no quality loss, ~50% VRAM savings)
    if has_gpu:
        cache_type_k = "q8_0"
        cache_type_v = "q8_0"
    else:
        cache_type_k = "f16"
        cache_type_v = "f16"

    # Flash attention: explicit on GPU (guarantees it; also required for KV quant)
    flash_attn = "on" if has_gpu else "auto"

    # Batch sizes: only override for CPU (GPU defaults of 2048/512 are fine)
    batch_size: int | None = None
    ubatch_size: int | None = None
    if not has_gpu:
        batch_size = BATCH_SIZE_CPU
        ubatch_size = UBATCH_SIZE_CPU

    # Re-check VRAM fit with actual KV cache quantization applied
    if has_gpu:
        estimated = _estimate_vram_usage(
            params, quant, ctx_size, n_layers, cache_type_k, cache_type_v
        )
        usable_vram = total_vram * VRAM_USAGE_FACTOR
        if estimated <= usable_vram:
            n_gpu_layers = -1
        else:
            n_gpu_layers = -2
            warnings.append(
                f"Model exceeds VRAM ({estimated:.0f}MB needed, {usable_vram:.0f}MB usable). "
                f"Using auto GPU layer offloading."
            )

    if has_gpu and len(hardware.gpus) > 1:
        ratios = [g.vram_mb / total_vram for g in hardware.gpus]
        tensor_split = ",".join(f"{r:.2f}" for r in ratios)
        split_mode = "row"

    args = OptimalArgs(
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        split_mode=split_mode,
        mlock=mlock,
        mmap=mmap,
        numa=numa,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        flash_attn=flash_attn,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
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
        f"Flash attention: {args.flash_attn}",
        f"KV cache (K): {args.cache_type_k}",
        f"KV cache (V): {args.cache_type_v}",
    ]
    if args.batch_size is not None:
        lines.append(f"Batch size: {args.batch_size} / {args.ubatch_size}")
    if args.tensor_split:
        lines.append(f"Tensor split: {args.tensor_split}")
    if args.split_mode != "layer":
        lines.append(f"Split mode: {args.split_mode}")
    if args.numa:
        lines.append("NUMA: enabled")
    return "\n".join(lines)
