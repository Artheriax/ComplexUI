"""Single-file pre-quantized checkpoint compatibility loading.

Comfy single-file checkpoints store quantized tensors under Comfy's LDM key
names. Diffusers' single-file conversion is key-based, so this path dequantizes
the quantized tensors first, lets diffusers build the pipeline, and leaves the
native component loader for diffusers-format folders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .quantized_loading import (
    _NATIVE_LINEAR_LAYOUTS,
    QuantizedLoadReport,
    _build_params,
    _quantized_tensor,
    _QuantModule,
    _replace_module,
    _resolve_module,
    dequantize_state_dict,
    read_quant_modules,
)

__all__ = [
    "apply_single_file_native_unet",
    "load_quantized_single_file",
    "quantization_mode",
]

LOGGER = logging.getLogger("complexui.single_file_quantization")

_LDM_UNET_PREFIX = "model.diffusion_model."


def apply_single_file_native_unet(
    unet: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    compute_dtype: torch.dtype,
) -> dict[str, QuantizedLoadReport]:
    """Re-apply native quantized weights to a loaded SD/SDXL UNet.

    Diffusers' LDM-to-diffusers conversion is key-based, so a marker state dict
    can recover which diffusers module each Comfy checkpoint module maps to
    without touching weights.
    """
    sources = [
        source
        for source in read_quant_modules(state)
        if source.prefix.startswith(_LDM_UNET_PREFIX)
    ]
    if not sources:
        return {}
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint

    markers: dict[str, torch.Tensor] = {}
    marker_sources: dict[int, Any] = {}
    for index, source in enumerate(sources):
        marker = torch.tensor([index + 1], dtype=torch.int64)
        key = f"{source.prefix}.weight"
        markers[key] = marker
        marker_sources[id(marker)] = source
    try:
        converted = convert_ldm_unet_checkpoint(markers, unet.config)
    except Exception as exc:
        LOGGER.warning("single-file quantized UNet mapping failed: %s", exc)
        return {}
    applied: dict[str, QuantizedLoadReport] = {}
    for target_key, marker in converted.items():
        source = marker_sources.get(id(marker))
        if source is None:
            continue
        if not target_key.endswith(".weight"):
            continue
        target_prefix = target_key[: -len(".weight")]
        module = _resolve_module(unet, target_prefix)
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            LOGGER.warning(
                "single-file quantized module %s maps to unsupported %s; "
                "keeping dequantized weights",
                source.prefix,
                type(module).__name__,
            )
            applied[source.prefix] = QuantizedLoadReport(
                module=target_prefix,
                format=source.config.get("format", "unknown"),
                runtime="dequantized",
                native=False,
            )
            continue
        try:
            layout_name, storage = _layout_entry(
                str(source.config.get("format", ""))
            )
            if layout_name not in _NATIVE_LINEAR_LAYOUTS:
                continue
            params = _build_params(
                layout_name,
                source.config,
                source.tensors,
                module,
                compute_dtype,
            )
            format_name = str(source.config.get("format", "")).lower()
            quantized = _quantized_tensor(
                layout_name,
                storage,
                source.tensors["weight"],
                params,
                device="cpu",
            )
            _replace_module(
                unet,
                _QuantModule(
                    prefix=target_prefix,
                    config=source.config,
                    tensors=source.tensors,
                    module=module,
                ),
                quantized,
                compute_dtype,
            )
            applied[source.prefix] = QuantizedLoadReport(
                module=target_prefix,
                format=format_name,
                runtime="comfy_kitchen",
                native=True,
            )
        except Exception as exc:
            LOGGER.warning(
                "native single-file quantization failed for %s: %s; "
                "keeping dequantized weights",
                target_prefix,
                exc,
            )
            applied[source.prefix] = QuantizedLoadReport(
                module=target_prefix,
                format=str(source.config.get("format", "unknown")),
                runtime="dequantized",
                native=False,
            )
    return applied


def _layout_entry(format_name: str) -> tuple[str, torch.dtype]:
    from .quantized_loading import _FORMAT_LAYOUTS

    entry = _FORMAT_LAYOUTS.get(format_name.lower().replace("-", "_"))
    if entry is None:
        raise ValueError(f"Unsupported quantization format: {format_name}")
    return entry


def quantization_mode(report: list[QuantizedLoadReport]) -> str:
    """Describe whether every quantized module kept native storage."""
    if not report:
        return "emulated"
    native = sum(1 for entry in report if entry.native)
    if native == len(report):
        return "pre_quantized"
    return "mixed" if native else "emulated"


def load_quantized_single_file(
    path: Path,
    pipeline_class: Any,
    *,
    torch_dtype: torch.dtype,
    use_safetensors: bool = True,
) -> tuple[Any, list[QuantizedLoadReport]]:
    """Load a single-file quantized checkpoint through diffusers.

    SD/SDXL UNets are re-patched with native Comfy Kitchen quantized weights
    after diffusers builds the pipeline. Other components stay dequantized.
    """
    state = load_file(str(path))
    dequantized, report = dequantize_state_dict(
        state, compute_dtype=torch_dtype
    )
    from diffusers.loaders import single_file, single_file_model, single_file_utils

    loaders = [
        cast(Any, single_file),
        cast(Any, single_file_model),
        cast(Any, single_file_utils),
    ]
    originals = [
        loader.load_single_file_checkpoint for loader in loaders
    ]

    def _load_checkpoint(*_args: Any, **_kwargs: Any) -> dict[str, torch.Tensor]:
        return dequantized

    for loader in loaders:
        loader.load_single_file_checkpoint = _load_checkpoint
    try:
        pipeline = pipeline_class.from_single_file(
            str(path),
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
        )
    finally:
        for loader, original in zip(loaders, originals, strict=True):
            loader.load_single_file_checkpoint = original
    unet = getattr(pipeline, "unet", None)
    if isinstance(unet, nn.Module):
        native = apply_single_file_native_unet(
            unet, state, compute_dtype=torch_dtype
        )
        if native:
            merged: list[QuantizedLoadReport] = []
            for entry in report:
                merged.append(native.get(entry.module, entry))
            report = merged
    LOGGER.info(
        "single-file quantized checkpoint loaded: %d native, %d emulated",
        sum(1 for entry in report if entry.native),
        sum(1 for entry in report if not entry.native),
    )
    return pipeline, report
