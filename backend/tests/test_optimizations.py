"""Contract tests for runtime optimization configuration."""

from __future__ import annotations

import pytest

from app import optimizations


@pytest.fixture(autouse=True)
def _clear_package_cache() -> None:
    optimizations._package_version.cache_clear()
    yield
    optimizations._package_version.cache_clear()


def test_defaults_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        optimizations.ATTENTION_ENV,
        optimizations.MEMORY_PROFILE_ENV,
        optimizations.CUDA_ALLOCATOR_ENV,
        optimizations.TORCH_COMPILE_ENV,
        optimizations.FP16_ACCUM_ENV,
        optimizations.ALLOC_CONF_ENV,
        optimizations.LEGACY_ALLOC_CONF_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    settings = optimizations.configured_settings()

    assert settings.attention == "auto"
    assert settings.memory_profile == "auto"
    assert settings.allocator == "native"
    assert settings.torch_compile == "off"
    assert settings.fp16_accumulation is False


def test_invalid_values_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(optimizations.ATTENTION_ENV, "magic")
    monkeypatch.setenv(optimizations.MEMORY_PROFILE_ENV, "tiny")
    monkeypatch.setenv(optimizations.CUDA_ALLOCATOR_ENV, "custom")
    monkeypatch.setenv(optimizations.TORCH_COMPILE_ENV, "turbo")

    settings = optimizations.configured_settings()

    assert settings.attention == "auto"
    assert settings.memory_profile == "auto"
    assert settings.allocator == "native"
    assert settings.torch_compile == "off"


def test_fp16_accumulation_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(optimizations.FP16_ACCUM_ENV, "1")
    assert optimizations.fp16_accumulation_enabled() is True
    monkeypatch.setenv(optimizations.FP16_ACCUM_ENV, "off")
    assert optimizations.fp16_accumulation_enabled() is False


@pytest.mark.parametrize(
    ("vram_gib", "expected"),
    [
        (6, "low"),
        (8, "low"),
        (12, "balanced"),
        (16, "balanced"),
        (24, "resident"),
        (80, "resident"),
    ],
)
def test_resolve_memory_profile_tiers(vram_gib: int, expected: str) -> None:
    total = int(vram_gib * (1024**3))
    assert optimizations.resolve_memory_profile("auto", total_vram_bytes=total) == expected


def test_resolve_memory_profile_explicit_and_unknown() -> None:
    assert optimizations.resolve_memory_profile("low", total_vram_bytes=80 * 1024**3) == "low"
    assert optimizations.resolve_memory_profile("resident") == "resident"
    assert optimizations.resolve_memory_profile("unknown") == "resident"


def test_allocator_conf_honors_existing_environment() -> None:
    environment = {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128"}

    assert (
        optimizations.allocator_conf(
            allocator="native",
            expandable_segments=True,
            environment=environment,
        )
        is None
    )


def test_allocator_conf_builds_native_and_async_profiles() -> None:
    environment: dict[str, str] = {}

    assert (
        optimizations.allocator_conf(
            allocator="native",
            expandable_segments=True,
            environment=environment,
        )
        == "expandable_segments:True"
    )
    assert (
        optimizations.allocator_conf(
            allocator="native",
            expandable_segments=False,
            raw_conf="garbage_collection_threshold:0.8",
            environment=environment,
        )
        == "garbage_collection_threshold:0.8"
    )
    assert (
        optimizations.allocator_conf(
            allocator="cudaMallocAsync",
            expandable_segments=True,
            environment=environment,
        )
        == "backend:cudaMallocAsync"
    )


def test_optimization_health_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(optimizations.ATTENTION_ENV, "sdpa-math")
    monkeypatch.setenv(optimizations.ATTENTION_BENCHMARK_ENV, "1")
    monkeypatch.setenv(optimizations.MEMORY_PROFILE_ENV, "low")
    monkeypatch.setenv(optimizations.CUDA_ALLOCATOR_ENV, "cudaMallocAsync")
    monkeypatch.setenv(optimizations.TORCH_COMPILE_ENV, "default")

    health = optimizations.optimization_health()

    assert health["attention"]["configured"] == "sdpa-math"
    assert health["attention"]["benchmark"] is True
    assert set(health["attention"]["packages"]) == set(
        optimizations.OPTIONAL_ATTENTION_PACKAGES
    )
    assert health["memory"]["profile"] == "low"
    assert health["memory"]["allocator"] == "cudaMallocAsync"
    assert health["compile"]["mode"] == "default"
    assert health["precision"]["quantized_runtime"] is False


def test_optional_package_detection_handles_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> str:
        raise optimizations.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(optimizations.importlib.metadata, "version", missing)
    optimizations._package_version.cache_clear()

    health = optimizations.attention_health()

    assert health["packages"] == {
        name: None for name in optimizations.OPTIONAL_ATTENTION_PACKAGES
    }


def test_summary_mentions_effective_settings() -> None:
    summary = optimizations.optimization_summary()

    assert "attention=" in summary
    assert "memory=" in summary
    assert "compile=" in summary
