"""Shared type definitions."""

from typing import Literal

Quant = Literal["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "IQ4_XS"]
Backend = Literal["cuda", "rocm", "metal", "vulkan", "cpu"]

QUANT_CHOICES = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "IQ4_XS"]
