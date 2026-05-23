"""CUDA mixed-precision helpers (bf16 autocast, fp32 optimizer steps)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Literal

import torch

MixedPrecisionMode = Literal["bf16", "fp32"]


def cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())


def resolve_bf16_enabled(
    mode: MixedPrecisionMode,
    device: torch.device,
) -> bool:
    """True when bf16 autocast should run on CUDA."""
    return mode == "bf16" and device.type == "cuda" and cuda_supports_bf16()


def autocast_dtype_name(enabled: bool) -> str:
    return "bfloat16" if enabled else "float32"


@contextmanager
def cuda_autocast(
    *,
    enabled: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> Generator[None, None, None]:
    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=enabled,
    ):
        yield


@contextmanager
def frozen_cuda_autocast(
    *,
    enabled: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> Generator[None, None, None]:
    """Frozen foundation encoders: no grads, optional bf16 autocast on CUDA."""
    with torch.no_grad():
        with cuda_autocast(enabled=enabled, dtype=dtype):
            yield
