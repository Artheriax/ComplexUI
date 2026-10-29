# /// script
# dependencies = ["torch>=2.2", "diffusers>=0.42.0"]
# ///
import logging
import os
import time
from typing import Literal

import torch

model: MODEL
mode: Literal[
    "server",
    "off",
    "auto",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
] = "server"
regional: bool = False
fullgraph: bool = False
dynamic: Literal["auto", "on", "off"] = "auto"

LOGGER = logging.getLogger("complexui.compile")
BLOCK_NAMES = (
    "transformer_blocks",
    "single_transformer_blocks",
    "joint_transformer_blocks",
    "blocks",
)


def _server_mode():
    value = os.environ.get("COMPLEXUI_TORCH_COMPILE", "off").strip().lower()
    allowed = {
        "off",
        "auto",
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    return value if value in allowed else "off"


def _recompile_limit():
    raw = os.environ.get("COMPLEXUI_TORCH_COMPILE_RECOMPILE_LIMIT", "8")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 8
    return max(1, min(value, 64))


def _configure_dynamo():
    limit = _recompile_limit()
    dynamo = getattr(torch, "_dynamo", None)
    config = getattr(dynamo, "config", None) if dynamo is not None else None
    if config is not None and hasattr(config, "cache_size_limit"):
        config.cache_size_limit = limit
    return limit


def _dynamic_value(value):
    if value == "on":
        return True
    if value == "off":
        return False
    return None


def _compile_callable(callable_value, compile_mode, is_fullgraph, is_dynamic):
    kwargs = {
        "mode": compile_mode,
        "fullgraph": bool(is_fullgraph),
        "dynamic": _dynamic_value(is_dynamic),
    }
    try:
        return torch.compile(
            callable_value,
            recompile_limit=_recompile_limit(),
            **kwargs,
        )
    except TypeError:
        return torch.compile(callable_value, **kwargs)


def _blocks(component):
    found = []
    for name in BLOCK_NAMES:
        value = getattr(component, name, None)
        if isinstance(value, torch.nn.ModuleList) and len(value) > 0:
            found.extend(list(value))
    return found


def _compile_regional(component, compile_mode, is_fullgraph, is_dynamic):
    blocks = _blocks(component)
    if not blocks:
        return 0
    for block in blocks:
        block.forward = _compile_callable(
            block.forward,
            compile_mode,
            is_fullgraph,
            is_dynamic,
        )
    return len(blocks)


def _apply_compile(wrapper, requested_mode, use_regional, use_fullgraph, use_dynamic):
    if wrapper.get("compiled"):
        print("compile: model is already compiled; leaving the existing graph")
        return wrapper
    component = wrapper.get("transformer") or wrapper.get("unet")
    if component is None:
        raise ValueError("MODEL wrapper is missing its transformer/UNet")
    requested = str(requested_mode)
    resolved = _server_mode() if requested == "server" else requested
    if resolved == "off":
        patched = dict(wrapper)
        patched["compiled"] = False
        patched["compile_mode"] = "off"
        patched["compile"] = {"mode": "off", "compiled": False}
        return patched
    if resolved in {"server", "auto"}:
        resolved = "default"
    if resolved not in {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }:
        raise ValueError("Unsupported compile mode: " + str(requested_mode))
    if wrapper.get("offload"):
        if requested in {"server", "auto"} and _server_mode() == "auto":
            patched = dict(wrapper)
            patched["compiled"] = False
            patched["compile_mode"] = "auto"
            patched["compile"] = {
                "mode": "auto",
                "compiled": False,
                "error": "skipped because the model is offloaded",
            }
            return patched
        raise ValueError(
            "torch.compile and model-level offload cannot be combined safely in "
            "this build; load with a resident memory profile before compiling"
        )
    if not hasattr(component, "forward"):
        raise TypeError("MODEL transformer cannot be compiled")
    limit = _configure_dynamo()
    started = time.perf_counter()
    compiled_blocks = 0
    error_message = None
    try:
        if use_regional:
            compiled_blocks = _compile_regional(
                component,
                resolved,
                use_fullgraph,
                use_dynamic,
            )
            if compiled_blocks == 0:
                component.forward = _compile_callable(
                    component.forward,
                    resolved,
                    use_fullgraph,
                    use_dynamic,
                )
        else:
            component.forward = _compile_callable(
                component.forward,
                resolved,
                use_fullgraph,
                use_dynamic,
            )
    except Exception as exc:
        error_message = str(exc)
        dynamo = getattr(torch, "_dynamo", None)
        if dynamo is not None and hasattr(dynamo, "reset"):
            dynamo.reset()
        if requested not in {"server", "auto"}:
            raise ValueError(
                f"torch.compile failed for mode '{resolved}': {error_message}"
            ) from exc
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        print(f"compile: failed, continuing eager ({error_message})")
        LOGGER.warning("compile: eager fallback after failure: %s", error_message)
        patched = dict(wrapper)
        patched["compiled"] = False
        patched["compile_mode"] = resolved
        patched["compile"] = {
            "mode": resolved,
            "regional": bool(use_regional),
            "compiled": False,
            "compile_ms": elapsed,
            "recompile_limit": limit,
            "error": error_message,
        }
        return patched
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    summary = (
        f"compile: mode={resolved} regional={bool(use_regional)} "
        f"blocks={compiled_blocks} compile_ms={elapsed}"
    )
    print(summary)
    LOGGER.info(summary)
    patched = dict(wrapper)
    patched["compiled"] = True
    patched["compile_mode"] = resolved
    patched["compile"] = {
        "mode": resolved,
        "regional": bool(use_regional),
        "blocks": compiled_blocks,
        "fullgraph": bool(use_fullgraph),
        "dynamic": str(use_dynamic),
        "compiled": True,
        "compile_ms": elapsed,
        "recompile_limit": limit,
        "error": None,
    }
    return patched


if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
model = _apply_compile(model, mode, regional, fullgraph, dynamic)
return model: MODEL
