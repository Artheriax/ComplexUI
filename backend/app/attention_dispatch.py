"""Capability-checked attention dispatch for diffusion workloads.

The server reports available backends without importing torch. Node templates
call :func:`apply_attention` from the disposable execution worker, where torch
is already present. The universal API is PyTorch's
``scaled_dot_product_attention``; optional kernels are installed as a guarded
wrapper around it and fall back to exact SDPA whenever their per-request
capabilities do not match.

Exported standalone scripts catch the optional import and continue with SDPA,
so optional kernels remain a ComplexUI runtime feature rather than a new
dependency of exported notebooks or scripts.
"""

from __future__ import annotations

import contextvars
import importlib.metadata
import importlib.util
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    "ATTENTION_BACKENDS",
    "AttentionSelection",
    "apply_attention",
    "attention_context",
    "available_attention_backends",
    "benchmark_attention",
    "ensure_attention_dispatch",
    "reset_attention_dispatch",
    "resolve_attention",
]

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
OPTIONAL_BACKENDS = {
    "xformers",
    "flash-attn",
    "flash-attn4",
    "sageattention",
    "comfy_kitchen_sol",
    "comfy_kitchen_int8",
}
_ORIGINAL_SDPA = F.scaled_dot_product_attention
_INSTALLED = False
_GLOBAL_SELECTION: dict[str, Any] | None = None
_ATTENTION_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("complexui_attention", default=None)
)
LOGGER = logging.getLogger("complexui.attention")


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    """One resolved attention selection."""

    requested: str
    backend: str
    reason: str
    approximates: bool
    flags: dict[str, bool]
    device: str
    capability: tuple[int, int] | None
    arch: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "backend": self.backend,
            "reason": self.reason,
            "approximates": self.approximates,
            "flags": dict(self.flags),
            "device": self.device,
            "capability": self.capability,
            "arch": self.arch,
        }


def _flags(flash=False, mem=False, cudnn=False, math=True) -> dict[str, bool]:
    return {
        "flash": bool(flash),
        "mem_efficient": bool(mem),
        "cudnn": bool(cudnn),
        "math": bool(math),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _package_available(name: str) -> bool:
    module = name.replace("-", "_")
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _cc_at_least(
    capability: tuple[int, int] | None, major: int, minor: int
) -> bool:
    if capability is None:
        return False
    return capability[0] > major or (
        capability[0] == major and capability[1] >= minor
    )


def _device_info() -> tuple[str, tuple[int, int] | None, str]:
    if not torch.cuda.is_available():
        return "cpu", None, "cpu"
    try:
        major, minor = torch.cuda.get_device_capability()
        capability: tuple[int, int] = (int(major), int(minor))
    except Exception:
        capability = (0, 0)
    if _cc_at_least(capability, 10, 0):
        arch = "blackwell"
    elif _cc_at_least(capability, 9, 0):
        arch = "hopper"
    elif _cc_at_least(capability, 8, 9):
        arch = "ada"
    elif _cc_at_least(capability, 8, 0):
        arch = "ampere"
    else:
        arch = f"sm{capability[0]}{capability[1]}"
    return "cuda", capability, arch


def _unavailable(requested: str, device: str, capability, reason: str) -> ValueError:
    device_text = (
        f"sm{capability[0]}{capability[1]}"
        if capability
        else device
    )
    hints = {
        "xformers": "install xformers; it is unsupported on sm120",
        "flash-attn": "install flash-attn with a wheel for your torch/CUDA build",
        "flash-attn4": "install flash-attn-4[cu13]; CuTeDSL currently needs Linux",
        "sageattention": "install sageattention; requires sm80+",
        "comfy_kitchen_sol": "install comfy-kitchen; SOL requires head_dim 128",
        "comfy_kitchen_int8": "install comfy-kitchen; INT8 attention requires CUDA",
    }
    hint = hints.get(requested, "choose auto, sdpa or sdpa-math")
    return ValueError(
        f"Attention backend '{requested}' is unavailable ({reason}; device="
        f"{device_text}). {hint}."
    )


def _optional_selection(
    requested: str,
    device: str,
    capability: tuple[int, int] | None,
    arch: str,
) -> AttentionSelection:
    if device != "cuda":
        raise _unavailable(requested, device, capability, "CUDA is required")
    if requested == "xformers":
        if not _package_available("xformers"):
            raise _unavailable(requested, device, capability, "package is not installed")
        if _cc_at_least(capability, 12, 0):
            raise _unavailable(requested, device, capability, "no sm120 build")
        approximates = False
    elif requested == "flash-attn":
        if not _package_available("flash-attn"):
            raise _unavailable(requested, device, capability, "package is not installed")
        if not _cc_at_least(capability, 8, 0):
            raise _unavailable(requested, device, capability, "requires sm80+")
        if _cc_at_least(capability, 12, 0):
            raise _unavailable(
                requested,
                device,
                capability,
                "official wheels do not cover sm120",
            )
        approximates = False
    elif requested == "sageattention":
        if not _package_available("sageattention"):
            raise _unavailable(requested, device, capability, "package is not installed")
        if not _cc_at_least(capability, 8, 0):
            raise _unavailable(requested, device, capability, "requires sm80+")
        approximates = True
    elif requested == "flash-attn4":
        if not _package_available("flash_attn"):
            raise _unavailable(requested, device, capability, "package is not installed")
        if not _cc_at_least(capability, 9, 0):
            raise _unavailable(requested, device, capability, "requires sm90+")
        try:
            importlib.import_module("flash_attn.cute")
        except Exception as exc:
            raise _unavailable(
                requested,
                device,
                capability,
                f"CuTeDSL import failed: {exc}",
            ) from exc
        approximates = False
    elif requested == "comfy_kitchen_sol":
        if not _package_available("comfy_kitchen"):
            raise _unavailable(requested, device, capability, "package is not installed")
        if not _cc_at_least(capability, 8, 0):
            raise _unavailable(requested, device, capability, "requires sm80+")
        approximates = True
    elif requested == "comfy_kitchen_int8":
        if not _package_available("comfy_kitchen"):
            raise _unavailable(requested, device, capability, "package is not installed")
        if not _cc_at_least(capability, 8, 0):
            raise _unavailable(requested, device, capability, "requires sm80+")
        approximates = True
    else:
        raise ValueError(f"Unsupported optional attention backend: {requested}")
    version_names = {
        "flash-attn4": "flash-attn-4",
        "comfy_kitchen_sol": "comfy-kitchen",
        "comfy_kitchen_int8": "comfy-kitchen",
    }
    version = _package_version(version_names.get(requested, requested))
    return AttentionSelection(
        requested=requested,
        backend=requested,
        reason=f"explicit optional kernel ({version or 'unknown'})",
        approximates=approximates,
        flags=_flags(math=True),
        device=device,
        capability=capability,
        arch=arch,
    )


def resolve_attention(
    requested: str,
    *,
    device: str | None = None,
    capability: tuple[int, int] | None = None,
    arch: str | None = None,
) -> AttentionSelection:
    """Resolve a backend request against device and package capabilities."""
    if device is None:
        device, capability, arch = _device_info()
    else:
        if capability is None and device == "cuda":
            try:
                major, minor = torch.cuda.get_device_capability()
                capability = (int(major), int(minor))
            except Exception:
                capability = None
        if arch is None:
            if _cc_at_least(capability, 10, 0):
                arch = "blackwell"
            elif _cc_at_least(capability, 9, 0):
                arch = "hopper"
            elif _cc_at_least(capability, 8, 9):
                arch = "ada"
            elif _cc_at_least(capability, 8, 0):
                arch = "ampere"
            else:
                arch = "cpu" if device == "cpu" else "unknown"
    value = (requested or "auto").strip().lower()
    if value == "server":
        value = os.environ.get("COMPLEXUI_ATTENTION", "auto").strip().lower()
    if value not in ATTENTION_BACKENDS:
        raise ValueError(f"Unsupported attention backend: {requested}")
    if value in OPTIONAL_BACKENDS:
        return _optional_selection(value, device, capability, arch)
    if value in {"auto", "sdpa"}:
        if device == "cpu":
            return AttentionSelection(
                value,
                "sdpa",
                "cpu default dispatch",
                False,
                _flags(),
                device,
                capability,
                arch,
            )
        if _cc_at_least(capability, 9, 0):
            return AttentionSelection(
                value,
                "sdpa",
                f"{arch} prefers flash/cuDNN SDPA",
                False,
                _flags(flash=True, mem=True, cudnn=True),
                device,
                capability,
                arch,
            )
        if _cc_at_least(capability, 8, 0):
            return AttentionSelection(
                value,
                "sdpa",
                f"{arch} prefers flash SDPA",
                False,
                _flags(flash=True, mem=True),
                device,
                capability,
                arch,
            )
        return AttentionSelection(
            value,
            "sdpa",
            "pre-Ampere safe SDPA fallback",
            False,
            _flags(mem=True),
            device,
            capability,
            arch,
        )
    if value == "sdpa-math":
        return AttentionSelection(
            value,
            value,
            "explicit reference math attention",
            False,
            _flags(),
            device,
            capability,
            arch,
        )
    if value == "sdpa-flash":
        if device != "cuda":
            raise _unavailable(value, device, capability, "CUDA is required")
        if not _cc_at_least(capability, 8, 0):
            raise _unavailable(value, device, capability, "requires sm80+")
        return AttentionSelection(
            value,
            value,
            "explicit flash SDPA",
            False,
            _flags(flash=True),
            device,
            capability,
            arch,
        )
    if value == "sdpa-mem-efficient":
        if device != "cuda":
            raise _unavailable(value, device, capability, "CUDA is required")
        return AttentionSelection(
            value,
            value,
            "explicit memory-efficient SDPA",
            False,
            _flags(mem=True),
            device,
            capability,
            arch,
        )
    if value == "sdpa-cudnn":
        if device != "cuda":
            raise _unavailable(value, device, capability, "CUDA is required")
        if not _cc_at_least(capability, 9, 0):
            raise _unavailable(value, device, capability, "requires sm90+")
        if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            raise _unavailable(
                value,
                device,
                capability,
                "installed torch has no cuDNN SDPA switch",
            )
        return AttentionSelection(
            value,
            value,
            "explicit cuDNN SDPA",
            False,
            _flags(cudnn=True),
            device,
            capability,
            arch,
        )
    raise ValueError(f"Unsupported attention backend: {requested}")


def _optional_call(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
    sol_tau: float = 1.0,
) -> torch.Tensor | None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return None
    if enable_gqa or dropout_p:
        return None
    if query.dtype not in {torch.float16, torch.bfloat16}:
        return None
    if query.shape[1] != key.shape[1] or key.shape[1] != value.shape[1]:
        return None
    head_dim = int(query.shape[-1])
    if backend == "sageattention":
        if (
            attn_mask is not None
            or scale is not None
            or query.shape[-2] != key.shape[-2]
            or head_dim not in {40, 64, 128, 256}
        ):
            return None
        sage = importlib.import_module("sageattention")

        return sage.sageattn(
            query,
            key,
            value,
            tensor_layout="HND",
            is_causal=is_causal,
        )
    if backend == "comfy_kitchen_int8":
        if is_causal or head_dim not in {64, 128, 256}:
            return None
        ck = importlib.import_module("comfy_kitchen")

        return ck.int8_attention(
            query,
            key,
            value,
            scale=scale,
            attn_mask=attn_mask,
        )
    if backend == "comfy_kitchen_sol":
        if (
            attn_mask is not None
            or scale is not None
            or is_causal
            or query.shape[-2] != key.shape[-2]
            or head_dim != 128
        ):
            return None
        ck = importlib.import_module("comfy_kitchen")
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        return ck.sol_attn(
            q,
            k,
            v,
            tau=float(sol_tau),
        ).transpose(1, 2)
    if attn_mask is not None or scale is not None:
        return None
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if backend == "flash-attn":
        if query.shape[-2] != key.shape[-2]:
            return None
        flash = importlib.import_module("flash_attn")

        return flash.flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            causal=is_causal,
        ).transpose(1, 2)
    if backend == "flash-attn4":
        if query.shape[-2] != key.shape[-2]:
            return None
        cute = importlib.import_module("flash_attn.cute")

        return cute.flash_attn_func(
            q,
            k,
            v,
            causal=is_causal,
        ).transpose(1, 2)
    if backend == "xformers":
        if is_causal and query.shape[-2] != key.shape[-2]:
            return None
        xformers_ops = importlib.import_module("xformers.ops")

        bias = None
        if is_causal:
            bias = xformers_ops.LowerTriangularMask()
        return xformers_ops.memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=bias,
        ).transpose(1, 2)
    return None


def _sdpa_context(selection: dict[str, Any] | None):
    if not isinstance(selection, dict):
        return nullcontext()
    backend = str(selection.get("backend", "sdpa"))
    if backend in {"auto", "sdpa"}:
        return nullcontext()
    mapping = {
        "sdpa-flash": SDPBackend.FLASH_ATTENTION,
        "sdpa-mem-efficient": SDPBackend.EFFICIENT_ATTENTION,
        "sdpa-cudnn": SDPBackend.CUDNN_ATTENTION,
        "sdpa-math": SDPBackend.MATH,
    }
    selected = mapping.get(backend)
    if selected is None:
        return nullcontext()
    if selected is SDPBackend.MATH:
        return sdpa_kernel([SDPBackend.MATH])
    return sdpa_kernel([selected, SDPBackend.MATH])


def _dispatch(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    config = _ATTENTION_CONTEXT.get() or _GLOBAL_SELECTION
    if not isinstance(config, dict):
        config = {}
    backend = str(config.get("backend", "sdpa"))
    if backend in OPTIONAL_BACKENDS:
        try:
            result = _optional_call(
                backend,
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
                sol_tau=float(config.get("sol_tau", 1.0)),
            )
            if result is not None:
                return result
            config["fallback"] = True
            config["fallback_reason"] = "request shape/mask is unsupported"
        except Exception as exc:
            config["fallback"] = True
            config["fallback_reason"] = str(exc)
            LOGGER.warning(
                "attention backend %s failed; falling back to SDPA: %s",
                backend,
                exc,
            )
    with _sdpa_context(config):
        return _ORIGINAL_SDPA(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


def _install_dispatch() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    torch.nn.functional.scaled_dot_product_attention = _dispatch
    _INSTALLED = True


def ensure_attention_dispatch() -> None:
    """Install the scoped SDPA hook without changing selection."""
    _install_dispatch()


@contextmanager
def attention_context(
    torch_module,
    selection: dict[str, Any] | str | None,
) -> Iterator[None]:
    """Apply a MODEL-scoped attention override for the duration of the block."""
    if selection is None or selection == "server":
        yield
        return
    if isinstance(selection, str):
        selection = resolve_attention(selection).as_dict()
    if not isinstance(selection, dict):
        yield
        return
    _install_dispatch()
    token = _ATTENTION_CONTEXT.set(dict(selection))
    try:
        yield
    finally:
        _ATTENTION_CONTEXT.reset(token)


def reset_attention_dispatch() -> None:
    """Restore the original SDPA hook (used by tests and worker cleanup)."""
    global _INSTALLED, _GLOBAL_SELECTION
    torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA
    _INSTALLED = False
    _GLOBAL_SELECTION = None
    _ATTENTION_CONTEXT.set(None)
    if hasattr(torch, "_complexui_attention"):
        delattr(torch, "_complexui_attention")


def apply_attention(torch_module, requested: str, device: str) -> dict[str, Any]:
    """Resolve, validate and activate the global launch attention default."""
    global _GLOBAL_SELECTION, _ORIGINAL_SDPA
    if _ORIGINAL_SDPA is not getattr(
        torch.nn.functional, "scaled_dot_product_attention", None
    ) and not _INSTALLED:
        _ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention
    selection = resolve_attention(requested, device=device)
    _GLOBAL_SELECTION = {
        **selection.as_dict(),
        "fallback": False,
        "fallback_reason": None,
    }
    if selection.backend not in {"auto", "sdpa"}:
        _install_dispatch()
    elif _INSTALLED:
        torch.nn.functional.scaled_dot_product_attention = _dispatch
    setattr(  # noqa: B010 - dynamic torch runtime configuration
        torch,
        "_complexui_attention",
        _GLOBAL_SELECTION,
    )
    message = (
        f"attention: global backend={selection.backend} "
        f"reason={selection.reason} device={selection.device} "
        f"approximates={selection.approximates} flags={selection.flags}"
    )
    print(message)
    LOGGER.info(message)
    return selection.as_dict()


def _benchmark_call(
    selection: AttentionSelection,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor | None:
    if selection.backend in OPTIONAL_BACKENDS:
        return _optional_call(
            selection.backend,
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        )
    with _sdpa_context(selection.as_dict()):
        return _ORIGINAL_SDPA(query, key, value)


def benchmark_attention(
    torch_module,
    device: str,
    requested: str = "auto",
    *,
    batch: int = 1,
    heads: int = 8,
    sequence: int | None = None,
    head_dim: int = 64,
    repeats: int = 3,
) -> dict[str, Any]:
    """Time available backends on synthetic tensors and return the fastest."""
    if sequence is None:
        sequence = 1024 if device == "cuda" else 256
    target = torch.device("cuda" if device == "cuda" else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    query = torch.randn(batch, heads, sequence, head_dim, device=target, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    reference = torch.softmax(
        (query.float() @ key.float().transpose(-2, -1)) / (head_dim**0.5),
        dim=-1,
    ) @ value.float()
    if requested not in {"auto", "server", "sdpa"}:
        candidates = [requested]
    else:
        candidates = [
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
        ]
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            selection = resolve_attention(candidate, device=device)
        except ValueError:
            continue
        if selection.backend in OPTIONAL_BACKENDS and not _package_available(
            selection.backend
        ):
            continue
        try:
            result = _benchmark_call(selection, query, key, value)
        except Exception as exc:
            LOGGER.debug("attention benchmark skipped %s: %s", candidate, exc)
            continue
        if result is None:
            continue
        tolerance = 5e-2 if selection.approximates else 2e-2
        correct = bool(
            torch.allclose(
                result.float(),
                reference,
                rtol=tolerance,
                atol=tolerance,
            )
        )
        if not correct:
            results.append(
                {"backend": selection.backend, "ms": None, "correct": False}
            )
            continue
        if target.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(max(1, repeats)):
            _benchmark_call(selection, query, key, value)
        if target.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = round(
            (time.perf_counter() - started) * 1000 / max(1, repeats), 3
        )
        entry = {
            "backend": selection.backend,
            "ms": elapsed_ms,
            "correct": True,
        }
        results.append(entry)
        if best is None or elapsed_ms < float(best["ms"]):
            best = entry
    if best is None:
        best = {"backend": "sdpa", "ms": None, "correct": True}
    return {
        "selected": best["backend"],
        "results": results,
        "shapes": {
            "batch": batch,
            "heads": heads,
            "sequence": sequence,
            "head_dim": head_dim,
            "device": device,
        },
    }


def available_attention_backends() -> dict[str, Any]:
    """Best-effort package availability for diagnostics and tests."""
    return {
        "backends": list(ATTENTION_BACKENDS),
        "packages": {
            name: _package_version(name)
            for name in (
                "xformers",
                "flash-attn",
                "flash-attn-4",
                "sageattention",
                "comfy-kitchen",
            )
        },
    }
