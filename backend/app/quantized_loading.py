"""Comfy-format pre-quantized component loading for the execution worker.

The loader keeps Comfy's per-module ``.comfy_quant`` JSON and scale tensors,
builds ``comfy_kitchen`` quantized tensors, and replaces only matching
``nn.Linear``/``nn.Embedding`` modules. Everything else loads through the normal
PyTorch state-dict path. Formats or layers without a native layout can be
dequantized explicitly with ``full_precision_matrix_mult``; weights are never
quantized here.

This module imports torch and optional kernels, so the HTTP server must not
import it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

__all__ = [
    "QuantizedEmbedding",
    "QuantizedLinear",
    "QuantizedLoadingError",
    "QuantizedLoadReport",
    "dequantize_state_dict",
    "load_quantized_component",
]

LOGGER = logging.getLogger("complexui.quantized_loading")

_MAX_WEIGHT_FILES = 128
_QUANT_MARKER = ".comfy_quant"
_QUANT_KEYS = (
    "weight",
    "weight_scale",
    "weight_scale_2",
    "input_scale",
    "pre_quant_scale",
    "weight_s_rel",
    "weight_s_channel",
    "weight_codebook",
    "bias",
    "comfy_quant",
)
_FORMAT_LAYOUTS: dict[str, tuple[str, torch.dtype]] = {
    "float8_e4m3fn": ("TensorCoreFP8Layout", torch.float8_e4m3fn),
    "float8_e4m3": ("TensorCoreFP8Layout", torch.float8_e4m3fn),
    "fp8_e4m3": ("TensorCoreFP8Layout", torch.float8_e4m3fn),
    "float8_e5m2": ("TensorCoreFP8Layout", torch.float8_e5m2),
    "fp8_e5m2": ("TensorCoreFP8Layout", torch.float8_e5m2),
    "int8_tensorwise": ("TensorWiseINT8Layout", torch.int8),
    "int8": ("TensorWiseINT8Layout", torch.int8),
    "int8_w8a8": ("TensorWiseINT8Layout", torch.int8),
    "convrot_w4a4": ("TensorCoreConvRotW4A4Layout", torch.int8),
    "convrot": ("TensorCoreConvRotW4A4Layout", torch.int8),
    "int8_convrot": ("TensorCoreConvRotW4A4Layout", torch.int8),
    "int8_convrot_w4a4": ("TensorCoreConvRotW4A4Layout", torch.int8),
    "nvfp4": ("TensorCoreNVFP4Layout", torch.uint8),
    "mxfp8": ("TensorCoreMXFP8Layout", torch.float8_e4m3fn),
}
# These layouts implement linear_input_act and can avoid dequantizing weights.
_NATIVE_LINEAR_LAYOUTS = {
    "TensorWiseINT8Layout",
    "TensorCoreConvRotW4A4Layout",
}


class QuantizedLoadingError(Exception):
    """Raised when a pre-quantized component cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class QuantizedLoadReport:
    """One loaded or emulated quantized layer."""

    module: str
    format: str
    runtime: str
    native: bool


@dataclass(slots=True)
class _QuantModule:
    prefix: str
    config: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    module: nn.Module


@dataclass(slots=True)
class _QuantSource:
    prefix: str
    config: dict[str, Any]
    tensors: dict[str, torch.Tensor]


def _comfy() -> Any:
    try:
        from comfy_kitchen import tensor as comfy_tensor
    except ImportError as exc:
        raise QuantizedLoadingError(
            "comfy-kitchen is required to load pre-quantized weights; "
            "install the matching Comfy Kitchen wheel"
        ) from exc
    return comfy_tensor


def _decode_config(value: torch.Tensor) -> dict[str, Any]:
    if value.numel() > 65_536:
        raise QuantizedLoadingError(".comfy_quant payload is too large")
    try:
        payload = bytes(value.detach().to("cpu", torch.uint8).numpy().tobytes())
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError, RuntimeError) as exc:
        raise QuantizedLoadingError("Invalid .comfy_quant JSON payload") from exc
    if not isinstance(decoded, dict) or not isinstance(decoded.get("format"), str):
        raise QuantizedLoadingError(".comfy_quant payload is missing a format")
    return decoded


def _format_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for part in path.split("."):
        if isinstance(current, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            current = current[int(part)]
        elif isinstance(current, nn.ModuleDict):
            current = current[part]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise QuantizedLoadingError(f"'{path}' is not a module")
    return current


def _set_module(root: nn.Module, path: str, value: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = _resolve_module(root, parent_path) if parent_path else root
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and name.isdigit():
        parent[int(name)] = value
    elif isinstance(parent, nn.ModuleDict):
        parent[name] = value
    else:
        setattr(parent, name, value)


def _module_shape(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        return (int(module.out_features), int(module.in_features))
    if isinstance(module, nn.Embedding):
        return (int(module.num_embeddings), int(module.embedding_dim))
    raise QuantizedLoadingError(
        f"Quantized module '{type(module).__name__}' is not a Linear or Embedding"
    )


def read_quant_modules(state: dict[str, torch.Tensor]) -> list[_QuantSource]:
    """Read per-module Comfy quantization metadata without resolving modules."""
    modules: list[_QuantSource] = []
    for key in sorted(state):
        if not key.endswith(_QUANT_MARKER):
            continue
        prefix = key[: -len(_QUANT_MARKER)].rstrip(".")
        if not prefix:
            raise QuantizedLoadingError("File-level .comfy_quant is not supported")
        config = _decode_config(state[key])
        tensors = {
            name: state[f"{prefix}.{name}"]
            for name in _QUANT_KEYS
            if f"{prefix}.{name}" in state
        }
        modules.append(
            _QuantSource(prefix=prefix, config=config, tensors=tensors)
        )
    if modules:
        return modules
    # Legacy scaled-FP8 checkpoints have no per-module JSON. Infer the format
    # from the stored FP8 weight dtype and require a matching weight scale.
    for key in sorted(state):
        if not key.endswith(".weight"):
            continue
        weight = state[key]
        if weight.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}:
            continue
        prefix = key[: -len(".weight")]
        scale = state.get(f"{prefix}.weight_scale")
        bias = state.get(f"{prefix}.bias")
        if scale is None:
            continue
        tensors = {"weight": weight, "weight_scale": scale}
        if bias is not None:
            tensors["bias"] = bias
        modules.append(
            _QuantSource(
                prefix=prefix,
                config={
                    "format": (
                        "float8_e4m3fn"
                        if weight.dtype == torch.float8_e4m3fn
                        else "float8_e5m2"
                    )
                },
                tensors=tensors,
            )
        )
    return modules


def _collect_quant_modules(
    state: dict[str, torch.Tensor], model: nn.Module
) -> list[_QuantModule]:
    return [
        _QuantModule(
            prefix=source.prefix,
            config=source.config,
            tensors=source.tensors,
            module=_resolve_module(model, source.prefix),
        )
        for source in read_quant_modules(state)
    ]


def _scale(
    tensors: dict[str, torch.Tensor],
    name: str,
    *,
    view: torch.dtype | None = None,
) -> torch.Tensor | None:
    value = tensors.get(name)
    if value is None:
        return None
    value = value.detach()
    if view is not None and value.dtype == torch.uint8:
        value = value.view(view)
    return value


def _params_for_shape(
    layout_name: str,
    config: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    shape: tuple[int, int],
    compute_dtype: torch.dtype,
) -> Any:
    comfy_tensor = _comfy()
    layout_cls = comfy_tensor.get_layout_class(layout_name)
    if layout_cls is None:
        raise QuantizedLoadingError(f"Unknown Comfy Kitchen layout: {layout_name}")
    options = config.get("params")
    options = options if isinstance(options, dict) else {}
    common = {"orig_dtype": compute_dtype, "orig_shape": shape}
    if layout_name == "TensorCoreFP8Layout":
        scale = _scale(tensors, "weight_scale")
        if scale is None:
            raise QuantizedLoadingError("Missing FP8 weight_scale")
        return layout_cls.Params(scale=scale.float(), **common)
    if layout_name == "TensorWiseINT8Layout":
        scale = _scale(tensors, "weight_scale")
        if scale is None:
            raise QuantizedLoadingError("Missing INT8 weight_scale")
        convrot = bool(config.get("convrot", options.get("convrot", False)))
        groupsize = int(
            config.get("convrot_groupsize", options.get("convrot_groupsize", 256))
        )
        return layout_cls.Params(
            scale=scale,
            convrot=convrot,
            convrot_groupsize=groupsize,
            **common,
        )
    if layout_name == "TensorCoreConvRotW4A4Layout":
        scale = _scale(tensors, "weight_scale")
        if scale is None:
            raise QuantizedLoadingError("Missing ConvRot W4A4 weight_scale")
        return layout_cls.Params(
            scale=scale,
            convrot_groupsize=int(
                config.get("convrot_groupsize", options.get("convrot_groupsize", 256))
            ),
            quant_group_size=64,
            linear_dtype=str(
                config.get("linear_dtype", options.get("linear_dtype", "int4"))
            ),
            **common,
        )
    if layout_name == "TensorCoreNVFP4Layout":
        block_scale = _scale(tensors, "weight_scale", view=torch.float8_e4m3fn)
        global_scale = _scale(tensors, "weight_scale_2")
        if block_scale is None or global_scale is None:
            raise QuantizedLoadingError("Missing NVFP4 scales")
        return layout_cls.Params(
            scale=global_scale.float(),
            block_scale=block_scale,
            **common,
        )
    if layout_name == "TensorCoreMXFP8Layout":
        block_scale = _scale(tensors, "weight_scale", view=torch.float8_e8m0fnu)
        if block_scale is None:
            raise QuantizedLoadingError("Missing MXFP8 block scales")
        return layout_cls.Params(scale=block_scale, **common)
    raise QuantizedLoadingError(f"Unsupported Comfy Kitchen layout: {layout_name}")


def _build_params(
    layout_name: str,
    config: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    module: nn.Module,
    compute_dtype: torch.dtype,
) -> Any:
    return _params_for_shape(
        layout_name,
        config,
        tensors,
        _module_shape(module),
        compute_dtype,
    )


def _quantized_tensor(
    layout_name: str,
    storage: torch.dtype,
    tensor: torch.Tensor,
    params: Any,
    *,
    device: torch.device | str,
) -> Any:
    comfy_tensor = _comfy()
    qdata = tensor.detach().to(device=device, dtype=storage).contiguous()
    quantized = comfy_tensor.QuantizedTensor(qdata, layout_name, params)
    if str(device) != "cpu":
        quantized = quantized.to(device=device)
    return quantized


def _dequantize(
    layout_name: str,
    storage: torch.dtype,
    tensor: torch.Tensor,
    params: Any,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    comfy_tensor = _comfy()
    layout_cls = comfy_tensor.get_layout_class(layout_name)
    qdata = tensor.detach().to(dtype=storage)
    return layout_cls.dequantize(qdata, params).to(compute_dtype)


def _source_shape(source: _QuantSource) -> tuple[int, int]:
    weight = source.tensors.get("weight")
    if weight is None:
        raise QuantizedLoadingError(f"Missing quantized weight at '{source.prefix}'")
    format_name = _format_key(source.config["format"])
    if format_name in {"convrot_w4a4", "convrot", "int8_convrot_w4a4", "nvfp4"}:
        return (int(weight.shape[0]), int(weight.shape[1]) * 2)
    return (int(weight.shape[0]), int(weight.shape[1]))


def dequantize_state_dict(
    state: dict[str, torch.Tensor],
    *,
    compute_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], list[QuantizedLoadReport]]:
    """Dequantize every Comfy-quantized tensor to a float state dict.

    This is the compatibility path for single-file checkpoints. Native component
    loading should use :func:`load_quantized_component` instead.
    """
    sources = read_quant_modules(state)
    if not sources:
        raise QuantizedLoadingError("No pre-quantized modules found")
    for name in list(state):
        if ".comfy_attention" in name:
            state.pop(name)
    report: list[QuantizedLoadReport] = []
    for source in sources:
        format_name = _format_key(source.config["format"])
        layout_entry = _FORMAT_LAYOUTS.get(format_name)
        if layout_entry is None:
            raise QuantizedLoadingError(
                f"Unsupported quantization format '{format_name}' at '{source.prefix}'"
            )
        layout_name, storage = layout_entry
        weight = source.tensors.get("weight")
        if weight is None:
            raise QuantizedLoadingError(
                f"Missing quantized weight at '{source.prefix}'"
            )
        params = _params_for_shape(
            layout_name,
            source.config,
            source.tensors,
            _source_shape(source),
            compute_dtype,
        )
        state[f"{source.prefix}.weight"] = _dequantize(
            layout_name, storage, weight, params, compute_dtype=compute_dtype
        )
        for name in list(state):
            if name.startswith(f"{source.prefix}.") and name != f"{source.prefix}.weight":
                state.pop(name)
        report.append(
            QuantizedLoadReport(
                module=source.prefix,
                format=format_name,
                runtime="dequantized",
                native=False,
            )
        )
    return state, report


class QuantizedLinear(nn.Linear):
    """Linear layer whose weight is a Comfy Kitchen quantized tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight: Any,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
        pre_quant_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias is not None,
            device="meta",
            dtype=compute_dtype,
        )
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().to(compute_dtype), requires_grad=False)
        if pre_quant_scale is not None:
            self.register_buffer(
                "pre_quant_scale",
                pre_quant_scale.detach().to(torch.float32),
                persistent=False,
            )
        else:
            self.pre_quant_scale = None
        self._native_error: str | None = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast(Any, self.weight)
        if self.pre_quant_scale is not None:
            input = input * self.pre_quant_scale.to(
                device=input.device, dtype=input.dtype
            )
        try:
            result = weight.linear_input_act(input, self.bias)
        except Exception as exc:
            if self._native_error is None:
                self._native_error = str(exc)
                LOGGER.warning(
                    "quantized linear kernel failed; using dequantized fallback: %s",
                    exc,
                )
            result = NotImplemented
        if result is not NotImplemented:
            return result
        dequantized = weight.dequantize()
        if dequantized.dtype != input.dtype:
            dequantized = dequantized.to(input.dtype)
        return F.linear(input, dequantized, self.bias)


class QuantizedEmbedding(nn.Embedding):
    """Embedding layer that dequantizes its table on lookup."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        weight: Any,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, device="meta", dtype=compute_dtype)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast(Any, self.weight)
        layout = weight.layout_cls
        if hasattr(layout, "dequantize_embedding"):
            flat = input.reshape(-1)
            values = layout.dequantize_embedding(
                weight._qdata,
                weight.params,
                flat,
            )
            return values.reshape(*input.shape, values.shape[-1])
        dequantized = weight.dequantize()
        if dequantized.dtype != input.dtype and input.dtype.is_floating_point:
            dequantized = dequantized.to(input.dtype)
        return F.embedding(
            input,
            dequantized,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def _replace_module(
    model: nn.Module,
    item: _QuantModule,
    weight: Any,
    compute_dtype: torch.dtype,
) -> None:
    module = item.module
    bias = item.tensors.get("bias")
    if (
        bias is None
        and isinstance(module, nn.Linear)
        and module.bias is not None
    ):
        raise QuantizedLoadingError(
            f"Quantized linear '{item.prefix}' is missing its bias tensor"
        )
    if isinstance(module, nn.Linear):
        replacement: nn.Module = QuantizedLinear(
            module.in_features,
            module.out_features,
            weight=weight,
            bias=bias,
            compute_dtype=compute_dtype,
            pre_quant_scale=item.tensors.get("pre_quant_scale"),
        )
    elif isinstance(module, nn.Embedding):
        replacement = QuantizedEmbedding(
            module.num_embeddings,
            module.embedding_dim,
            weight=weight,
            compute_dtype=compute_dtype,
        )
    else:
        raise QuantizedLoadingError(
            f"Unsupported quantized module type at '{item.prefix}': "
            f"{type(module).__name__}"
        )
    _set_module(model, item.prefix, replacement)


def _load_state_files(path: Path) -> dict[str, torch.Tensor]:
    files = sorted(path.glob("*.safetensors"))
    if len(files) > _MAX_WEIGHT_FILES:
        raise QuantizedLoadingError(
            f"Component has too many weight shards ({len(files)} > {_MAX_WEIGHT_FILES})"
        )
    state: dict[str, torch.Tensor] = {}
    for file in files:
        state.update(load_file(str(file)))
    return state


def load_quantized_component(
    model: nn.Module,
    path: Path,
    *,
    compute_dtype: torch.dtype,
    device: torch.device | str = "cpu",
    disabled_formats: tuple[str, ...] = (),
) -> tuple[nn.Module, list[QuantizedLoadReport]]:
    """Load Comfy-format quantized weights into an instantiated component."""
    state = _load_state_files(path)
    if not state:
        raise QuantizedLoadingError(f"No SafeTensors weights found in {path}")
    for name in list(state):
        if ".comfy_attention" in name:
            state.pop(name)
    items = _collect_quant_modules(state, model)
    if not items:
        raise QuantizedLoadingError(
            f"No .comfy_quant markers found in {path}; use the normal loader"
        )
    disabled = {_format_key(value) for value in disabled_formats}
    report: list[QuantizedLoadReport] = []
    native_prefixes: set[str] = set()
    emulated_prefixes: set[str] = set()
    for item in items:
        format_name = _format_key(item.config["format"])
        layout_entry = _FORMAT_LAYOUTS.get(format_name)
        if layout_entry is None:
            raise QuantizedLoadingError(
                f"Unsupported quantization format '{format_name}' at '{item.prefix}'"
            )
        layout_name, storage = layout_entry
        options = item.config.get("params")
        options = options if isinstance(options, dict) else {}
        emulated = bool(
            item.config.get(
                "full_precision_matrix_mult",
                options.get("full_precision_matrix_mult", False),
            )
        ) or format_name in disabled
        weight = item.tensors.get("weight")
        if weight is None:
            raise QuantizedLoadingError(f"Missing quantized weight at '{item.prefix}'")
        ignored = [
            name
            for name in (("input_scale", "pre_quant_scale") if emulated else ("input_scale",))
            if name in item.tensors
        ]
        if ignored:
            LOGGER.warning(
                "quantized layer %s stores %s; this loader keeps activation "
                "scaling dynamic and ignores the calibrated values",
                item.prefix,
                ", ".join(ignored),
            )
        params = _build_params(
            layout_name, item.config, item.tensors, item.module, compute_dtype
        )
        if emulated:
            dequantized = _dequantize(
                layout_name, storage, weight, params, compute_dtype=compute_dtype
            )
            state[f"{item.prefix}.weight"] = dequantized
            emulated_prefixes.add(item.prefix)
            report.append(
                QuantizedLoadReport(
                    module=item.prefix,
                    format=format_name,
                    runtime="dequantized",
                    native=False,
                )
            )
        else:
            quantized = _quantized_tensor(
                layout_name, storage, weight, params, device=device
            )
            _replace_module(model, item, quantized, compute_dtype)
            native_prefixes.add(item.prefix)
            native = layout_name in _NATIVE_LINEAR_LAYOUTS
            report.append(
                QuantizedLoadReport(
                    module=item.prefix,
                    format=format_name,
                    runtime="comfy_kitchen" if native else "quantized_storage",
                    native=native,
                )
            )
    # Remove every key owned by a replaced or emulated quantized module. The
    # emulated path intentionally writes back one float weight per module.
    for prefix in native_prefixes | emulated_prefixes:
        keep_weight = prefix in emulated_prefixes
        for name in list(state):
            if not name.startswith(f"{prefix}."):
                continue
            if keep_weight and name == f"{prefix}.weight":
                continue
            state.pop(name)
    model.to(dtype=compute_dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    remaining_missing = [
        key
        for key in missing
        if not any(
            key.startswith(f"{prefix}.")
            for prefix in native_prefixes | emulated_prefixes
        )
    ]
    if remaining_missing or unexpected:
        raise QuantizedLoadingError(
            "Quantized component does not match its config: "
            f"missing={remaining_missing[:8]} unexpected={unexpected[:8]}"
        )
    model.eval()
    LOGGER.info(
        "quantized component loaded: %d layer(s), %d native, %d emulated",
        len(report),
        sum(1 for entry in report if entry.native),
        sum(1 for entry in report if not entry.native),
    )
    return model, report
