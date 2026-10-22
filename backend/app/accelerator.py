"""Accelerator selection for PyTorch dependency installs and diagnostics."""

from __future__ import annotations

import os
import sys

from .nvml import nvidia_available

__all__ = [
    "ACCELERATORS",
    "ACCELERATOR_ENV",
    "CUDA_INDEX_ENV",
    "DEFAULT_ACCELERATOR",
    "DEFAULT_CUDA_INDEX",
    "PYPI_INDEX",
    "TORCH_CPU_INDEX",
    "configured_accelerator",
    "cuda_index",
    "effective_accelerator",
    "torch_index",
]

ACCELERATOR_ENV = "COMPLEXUI_ACCELERATOR"
CUDA_INDEX_ENV = "COMPLEXUI_TORCH_CUDA_INDEX"
DEFAULT_ACCELERATOR = "auto"
DEFAULT_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
ACCELERATORS = ("auto", "cpu", "cuda")


def configured_accelerator() -> str:
    """Return the configured accelerator mode, defaulting to ``auto``."""
    raw = os.environ.get(ACCELERATOR_ENV, DEFAULT_ACCELERATOR).strip().lower()
    return raw if raw in ACCELERATORS else DEFAULT_ACCELERATOR


def cuda_index() -> str:
    """Return the CUDA wheel index, allowing future CUDA releases to be used."""
    return os.environ.get(CUDA_INDEX_ENV, "").strip() or DEFAULT_CUDA_INDEX


def _platform_supports_accelerator(platform: str) -> bool:
    return platform.startswith(("win32", "linux"))


def effective_accelerator(
    accelerator: str | None = None,
    *,
    platform: str | None = None,
) -> str:
    """Resolve ``auto`` into ``cuda`` or ``cpu`` for the current machine."""
    target = sys.platform if platform is None else platform
    mode = (accelerator or configured_accelerator()).strip().lower()
    if mode not in ACCELERATORS:
        mode = DEFAULT_ACCELERATOR
    if not _platform_supports_accelerator(target):
        return "cpu"
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return "cuda"
    return "cuda" if nvidia_available() else "cpu"


def torch_index(
    accelerator: str | None = None,
    *,
    platform: str | None = None,
) -> str:
    """Return the PyTorch wheel index for the selected accelerator."""
    return (
        cuda_index()
        if effective_accelerator(accelerator, platform=platform) == "cuda"
        else TORCH_CPU_INDEX
    )
