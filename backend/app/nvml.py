"""Best-effort NVIDIA GPU discovery through NVML.

The module stays importable when no NVIDIA driver is available. Initialization
is attempted once; later calls reuse the result so a missing driver does not
add repeated startup latency to every metrics request.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

__all__ = ["nvidia_available", "nvml_devices"]

_LOCK = threading.Lock()
_MODULE: Any | None = None
_INITIALIZED = False
_DEVICE_COUNT = 0


def _initialize() -> Any | None:
    global _INITIALIZED, _MODULE, _DEVICE_COUNT
    if _INITIALIZED:
        return _MODULE
    with _LOCK:
        if _INITIALIZED:
            return _MODULE
        try:
            module = importlib.import_module("pynvml")
            module.nvmlInit()
            count = int(module.nvmlDeviceGetCount())
        except Exception:
            _MODULE = None
            _DEVICE_COUNT = 0
        else:
            _MODULE = module
            _DEVICE_COUNT = max(count, 0)
        _INITIALIZED = True
        return _MODULE


def nvidia_available() -> bool:
    """Return whether NVML initialized and exposed at least one GPU."""
    _initialize()
    return _DEVICE_COUNT > 0


def _name(module: Any, handle: Any) -> str:
    try:
        value = module.nvmlDeviceGetName(handle)
    except Exception:
        return "NVIDIA GPU"
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _memory(module: Any, handle: Any) -> tuple[int | None, int | None]:
    try:
        memory = module.nvmlDeviceGetMemoryInfo(handle)
        return int(memory.total), int(memory.used)
    except Exception:
        return None, None


def _utilization(module: Any, handle: Any) -> float | None:
    try:
        rates = module.nvmlDeviceGetUtilizationRates(handle)
        return float(rates.gpu)
    except Exception:
        return None


def _temperature(module: Any, handle: Any) -> float | None:
    try:
        value = module.nvmlDeviceGetTemperature(
            handle,
            module.NVML_TEMPERATURE_GPU,
        )
        return float(value)
    except Exception:
        return None


def nvml_devices() -> list[dict[str, Any]]:
    """Return per-GPU utilization, VRAM and temperature metadata."""
    module = _initialize()
    if module is None:
        return []
    devices: list[dict[str, Any]] = []
    for index in range(_DEVICE_COUNT):
        try:
            handle = module.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            continue
        total, used = _memory(module, handle)
        devices.append(
            {
                "index": index,
                "name": _name(module, handle),
                "utilization": _utilization(module, handle),
                "vram_total_bytes": total,
                "vram_used_bytes": used,
                "temperature_c": _temperature(module, handle),
            }
        )
    return devices
