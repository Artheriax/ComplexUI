"""Live CPU, memory and NVIDIA GPU metrics for the system monitor."""

from __future__ import annotations

from datetime import UTC, datetime

import psutil

from .nvml import nvml_devices
from .schemas import CpuMetrics, GpuMetrics, MemoryMetrics, SystemMetricsResponse

__all__ = ["collect_system_metrics"]


def _cpu_metrics() -> CpuMetrics:
    utilization: float | None
    cores: int | None
    try:
        utilization = float(psutil.cpu_percent(interval=0.1))
    except (AttributeError, NotImplementedError, OSError):
        utilization = None
    try:
        cores = psutil.cpu_count(logical=True)
    except (AttributeError, NotImplementedError, OSError):
        cores = None
    return CpuMetrics(utilization=utilization, cores=cores)


def _memory_metrics() -> MemoryMetrics:
    try:
        memory = psutil.virtual_memory()
    except (AttributeError, NotImplementedError, OSError):
        return MemoryMetrics(total_bytes=0, used_bytes=0, utilization=0.0)
    total = max(int(memory.total), 0)
    used = max(total - int(memory.available), 0)
    utilization = float(memory.percent) if memory.percent is not None else 0.0
    return MemoryMetrics(
        total_bytes=total,
        used_bytes=used,
        utilization=min(max(utilization, 0.0), 100.0),
    )


def _gpu_metrics() -> list[GpuMetrics]:
    gpus: list[GpuMetrics] = []
    for device in nvml_devices():
        gpus.append(
            GpuMetrics(
                index=int(device.get("index", len(gpus))),
                name=str(device.get("name", "NVIDIA GPU")),
                utilization=device.get("utilization"),
                vram_total_bytes=device.get("vram_total_bytes"),
                vram_used_bytes=device.get("vram_used_bytes"),
                temperature_c=device.get("temperature_c"),
            )
        )
    return gpus


def collect_system_metrics() -> SystemMetricsResponse:
    """Collect one system-monitor snapshot without failing on missing sensors."""
    return SystemMetricsResponse(
        cpu=_cpu_metrics(),
        memory=_memory_metrics(),
        gpus=_gpu_metrics(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )
