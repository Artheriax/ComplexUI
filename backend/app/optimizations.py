"""Runtime optimization configuration and diagnostics.

The module deliberately never imports torch. The HTTP server must be able to
report and validate attention, precision, compilation, allocator and memory
settings without initializing CUDA or requiring the optional ML toolchain.
Node templates and the isolated execution worker apply the actual kernels and
dtype conversions.
"""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "ALLOCATORS",
    "ALLOC_CONF_ENV",
    "ATTENTION_BACKENDS",
    "ATTENTION_BENCHMARK_ENV",
    "ATTENTION_ENV",
    "COMPILE_CACHE_ENV",
    "COMPILE_MODES",
    "CUDA_ALLOCATOR_ENV",
    "FP16_ACCUM_ENV",
    "LEGACY_ALLOC_CONF_ENV",
    "MEMORY_PROFILE_ENV",
    "MEMORY_PROFILES",
    "OPTIONAL_ATTENTION_PACKAGES",
    "QUANTIZATION_FORMATS",
    "TORCH_COMPILE_ENV",
    "OptimizationSettings",
    "allocator_conf",
    "attention_health",
    "attention_benchmark_enabled",
    "compile_cache_dir",
    "compile_health",
    "configured_allocator",
    "configured_attention",
    "configured_compile_mode",
    "configured_memory_profile",
    "configured_settings",
    "fp16_accumulation_enabled",
    "memory_health",
    "memory_profile_offloads_model",
    "memory_profile_offloads_vae",
    "optimization_health",
    "optimization_summary",
    "precision_health",
    "resolve_memory_profile",
]

ATTENTION_ENV = "COMPLEXUI_ATTENTION"
ATTENTION_BENCHMARK_ENV = "COMPLEXUI_ATTENTION_BENCHMARK"
MEMORY_PROFILE_ENV = "COMPLEXUI_MEMORY_PROFILE"
CUDA_ALLOCATOR_ENV = "COMPLEXUI_CUDA_ALLOCATOR"
TORCH_COMPILE_ENV = "COMPLEXUI_TORCH_COMPILE"
FP16_ACCUM_ENV = "COMPLEXUI_EXPERIMENTAL_FP16_ACCUM"
COMPILE_CACHE_ENV = "COMPLEXUI_COMPILE_CACHE_DIR"
ALLOC_CONF_ENV = "PYTORCH_ALLOC_CONF"
LEGACY_ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"

ATTENTION_BACKENDS: tuple[str, ...] = (
    "auto",
    "sdpa",
    "sdpa-flash",
    "sdpa-mem-efficient",
    "sdpa-cudnn",
    "sdpa-math",
    "xformers",
    "flash-attn",
    "flash-attn4",
    "sageattention",
    "comfy_kitchen_sol",
    "comfy_kitchen_int8",
)
MEMORY_PROFILES: tuple[str, ...] = ("auto", "low", "balanced", "resident")
ALLOCATORS: tuple[str, ...] = ("native", "cudaMallocAsync")
COMPILE_MODES: tuple[str, ...] = (
    "off",
    "auto",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)

OPTIONAL_ATTENTION_PACKAGES: tuple[str, ...] = (
    "xformers",
    "flash-attn",
    "flash-attn-4",
    "sageattention",
    "comfy-kitchen",
    "torchao",
)
QUANTIZATION_FORMATS: tuple[str, ...] = (
    "fp8-e4m3",
    "fp8-e5m2",
    "mxfp8",
    "nvfp4",
    "int8-w8a8",
    "convrot-w4a4",
    "nf4",
    "gguf",
)


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """Effective process-wide optimization settings."""

    attention: str
    memory_profile: str
    allocator: str
    torch_compile: str
    fp16_accumulation: bool
    cache_dir: str | None
    alloc_conf: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "attention": self.attention,
            "memory_profile": self.memory_profile,
            "allocator": self.allocator,
            "torch_compile": self.torch_compile,
            "fp16_accumulation": self.fp16_accumulation,
            "cache_dir": self.cache_dir,
            "alloc_conf": self.alloc_conf,
        }


def _environment(name: str) -> str:
    return os.environ.get(name, "").strip()


def _choice(name: str, choices: tuple[str, ...], default: str) -> str:
    value = _environment(name).lower()
    return value if value in choices else default


def configured_attention() -> str:
    """Return the configured attention mode, defaulting to ``auto``."""
    return _choice(ATTENTION_ENV, ATTENTION_BACKENDS, "auto")


def configured_memory_profile() -> str:
    """Return the configured memory profile, defaulting to ``auto``."""
    return _choice(MEMORY_PROFILE_ENV, MEMORY_PROFILES, "auto")


def configured_allocator() -> str:
    """Return the configured CUDA allocator backend."""
    value = _environment(CUDA_ALLOCATOR_ENV).lower()
    return "cudaMallocAsync" if value == "cudamallocasync" else "native"


def configured_compile_mode() -> str:
    """Return the configured torch.compile mode, defaulting to ``off``."""
    return _choice(TORCH_COMPILE_ENV, COMPILE_MODES, "off")


def fp16_accumulation_enabled() -> bool:
    """Return whether the experimental FP16-accumulation gate is enabled."""
    value = _environment(FP16_ACCUM_ENV).lower()
    return value not in {"", "0", "false", "no", "off"}


def attention_benchmark_enabled() -> bool:
    """Return whether startup attention benchmarking is enabled."""
    value = _environment(ATTENTION_BENCHMARK_ENV).lower()
    return value in {"1", "true", "on", "yes"}


def compile_cache_dir() -> str | None:
    """Return the configured stable Inductor cache directory, if any."""
    value = _environment(COMPILE_CACHE_ENV)
    if value:
        return value
    inductor = _environment("TORCHINDUCTOR_CACHE_DIR")
    return inductor or None


def _alloc_conf() -> str | None:
    return _environment(ALLOC_CONF_ENV) or _environment(LEGACY_ALLOC_CONF_ENV) or None


def allocator_conf(
    *,
    allocator: str,
    expandable_segments: bool,
    raw_conf: str | None = None,
    environment: dict[str, str] | None = None,
) -> str | None:
    """Build a ``PYTORCH_ALLOC_CONF`` value without overriding explicit users.

    ``environment`` defaults to ``os.environ`` and is used to honor an existing
    allocator configuration. ``cudaMallocAsync`` is an all-or-nothing backend;
    callers must reject it when native-only options are supplied.
    """
    env = os.environ if environment is None else environment
    if ALLOC_CONF_ENV in env or LEGACY_ALLOC_CONF_ENV in env:
        return None
    if allocator == "cudaMallocAsync":
        return "backend:cudaMallocAsync"
    if raw_conf and raw_conf.strip():
        return raw_conf.strip()
    return "expandable_segments:True" if expandable_segments else None


def resolve_memory_profile(
    profile: str | None = None,
    *,
    total_vram_bytes: int | None = None,
) -> str:
    """Resolve ``auto`` into a concrete memory profile.

    ``auto`` follows the research VRAM tiers: under 10 GiB is ``low``, under
    20 GiB is ``balanced`` and larger cards stay ``resident``. Without CUDA
    information the safe default is ``resident`` (no offload).
    """
    value = (profile or configured_memory_profile()).strip().lower()
    if value in MEMORY_PROFILES and value != "auto":
        return value
    if not total_vram_bytes or total_vram_bytes <= 0:
        return "resident"
    gib = float(total_vram_bytes) / (1024.0**3)
    if gib < 10.0:
        return "low"
    if gib < 20.0:
        return "balanced"
    return "resident"


def memory_profile_offloads_model(profile: str) -> bool:
    """Return whether a profile offloads the diffusion transformer."""
    return profile == "low"


def memory_profile_offloads_vae(profile: str) -> bool:
    """Return whether a profile offloads VAE weights between calls."""
    return profile in {"low", "balanced"}


@lru_cache(maxsize=32)
def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _package_versions() -> dict[str, str | None]:
    return {name: _package_version(name) for name in OPTIONAL_ATTENTION_PACKAGES}


def attention_health() -> dict[str, Any]:
    """Return configured attention settings and optional package availability."""
    return {
        "configured": configured_attention(),
        "benchmark": attention_benchmark_enabled(),
        "backends": list(ATTENTION_BACKENDS),
        "packages": _package_versions(),
    }


def precision_health() -> dict[str, Any]:
    """Return precision tracking and quantization policy information."""
    return {
        "default": "keep",
        "fp16_accumulation": fp16_accumulation_enabled(),
        "quantization_formats": list(QUANTIZATION_FORMATS),
        "quantized_runtime": False,
    }


def memory_health() -> dict[str, Any]:
    """Return configured memory and allocator settings."""
    return {
        "profile": configured_memory_profile(),
        "allocator": configured_allocator(),
        "alloc_conf": _alloc_conf(),
        "pinned_memory": False,
    }


def compile_health() -> dict[str, Any]:
    """Return configured compilation settings."""
    return {
        "mode": configured_compile_mode(),
        "cache_dir": compile_cache_dir(),
    }


def configured_settings() -> OptimizationSettings:
    """Return the full effective optimization settings."""
    return OptimizationSettings(
        attention=configured_attention(),
        memory_profile=configured_memory_profile(),
        allocator=configured_allocator(),
        torch_compile=configured_compile_mode(),
        fp16_accumulation=fp16_accumulation_enabled(),
        cache_dir=compile_cache_dir(),
        alloc_conf=_alloc_conf(),
    )


def optimization_health() -> dict[str, Any]:
    """Return the ``/api/health`` optimization payload."""
    return {
        "attention": attention_health(),
        "precision": precision_health(),
        "memory": memory_health(),
        "compile": compile_health(),
    }


def optimization_summary() -> str:
    """Return a compact one-line summary for startup logs and banners."""
    settings = configured_settings()
    parts = [
        f"attention={settings.attention}",
        f"memory={settings.memory_profile}",
        f"allocator={settings.allocator}",
        f"compile={settings.torch_compile}",
    ]
    if settings.cache_dir:
        parts.append(f"compile_cache={settings.cache_dir}")
    if settings.fp16_accumulation:
        parts.append("fp16_accumulation=experimental")
    return " ".join(parts)
