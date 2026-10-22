"""Tests for the live system monitor metrics."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import system_metrics
from app.main import app
from app.schemas import CpuMetrics, GpuMetrics, MemoryMetrics, SystemMetricsResponse


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def _patch_basic_metrics(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cpu_percent: float = 37.5,
    cores: int = 16,
    total: int = 64 * 1024**3,
    available: int = 48 * 1024**3,
    percent: float = 25.0,
) -> None:
    monkeypatch.setattr(
        system_metrics.psutil,
        "cpu_percent",
        lambda interval=None: cpu_percent,
    )
    monkeypatch.setattr(system_metrics.psutil, "cpu_count", lambda logical=True: cores)
    monkeypatch.setattr(
        system_metrics.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=total, available=available, percent=percent),
    )


def test_collect_system_metrics_includes_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_basic_metrics(monkeypatch)
    monkeypatch.setattr(
        system_metrics,
        "nvml_devices",
        lambda: [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 5080",
                "utilization": 72.0,
                "vram_total_bytes": 16 * 1024**3,
                "vram_used_bytes": 8 * 1024**3,
                "temperature_c": 64.0,
            }
        ],
    )

    metrics = system_metrics.collect_system_metrics()

    assert metrics.cpu.utilization == pytest.approx(37.5)
    assert metrics.cpu.cores == 16
    assert metrics.memory.total_bytes == 64 * 1024**3
    assert metrics.memory.used_bytes == 16 * 1024**3
    assert metrics.memory.utilization == pytest.approx(25.0)
    assert metrics.gpus == [
        GpuMetrics(
            index=0,
            name="NVIDIA GeForce RTX 5080",
            utilization=72.0,
            vram_total_bytes=16 * 1024**3,
            vram_used_bytes=8 * 1024**3,
            temperature_c=64.0,
        )
    ]
    assert metrics.timestamp


def test_collect_system_metrics_handles_missing_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_basic_metrics(monkeypatch)
    monkeypatch.setattr(system_metrics, "nvml_devices", lambda: [])

    metrics = system_metrics.collect_system_metrics()

    assert metrics.gpus == []


def test_system_metrics_endpoint_returns_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = SystemMetricsResponse(
        cpu=CpuMetrics(utilization=10.0, cores=8),
        memory=MemoryMetrics(total_bytes=16, used_bytes=4, utilization=25.0),
        gpus=[],
        timestamp="2026-10-22T00:00:00+00:00",
    )
    monkeypatch.setattr("app.main.collect_system_metrics", lambda: payload)

    response = client.get("/api/system/metrics")

    assert response.status_code == 200
    assert response.json() == payload.model_dump(mode="json")
