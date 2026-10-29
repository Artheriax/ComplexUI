"""Pre-quantized checkpoint metadata discovery and capability gates.

Only headers and metadata are read here. Model weights are never imported or
loaded by the HTTP server. Float-only loaders call
:func:`require_float_checkpoint` before touching weights; the execution worker
can reuse the same metadata to pick a runtime once native quantized loading is
enabled.
"""

from __future__ import annotations

import importlib.util
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import QuantizationInfo

__all__ = [
    "FORMAT_SPECS",
    "QuantizationError",
    "detect_quantization",
    "normalize_format",
    "quantization_health",
    "read_parameter_count",
    "require_float_checkpoint",
    "validate_quantization",
]

MAX_HEADER_BYTES = 16_000_000
MAX_METADATA_BYTES = 32_000_000
MAX_METADATA_FILES = 16
MAX_WEIGHT_FILES = 64
MAX_GGUF_ITEMS = 100_000
QUANT_SUFFIXES = {".safetensors", ".gguf"}
_COMPONENT_DIRS = (
    "transformer",
    "unet",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "vae",
)


class QuantizationError(Exception):
    """Raised when a checkpoint's quantization metadata is invalid."""


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """Static capability metadata for one pre-quantized format."""

    name: str
    label: str
    bits: int | None
    runtime: str
    min_cc: tuple[int, int] | None = None
    package: str | None = None
    components: tuple[str, ...] = ("diffusion", "clip")
    notes: str = ""


FORMAT_SPECS: dict[str, FormatSpec] = {
    "int8_w8a8": FormatSpec(
        "int8_w8a8",
        "INT8 W8A8",
        8,
        "comfy_kitchen",
        (8, 0),
        "comfy-kitchen",
    ),
    "int8_convrot_w4a4": FormatSpec(
        "int8_convrot_w4a4",
        "INT8 ConvRot W4A4",
        4,
        "comfy_kitchen",
        (8, 0),
        "comfy-kitchen",
    ),
    "gguf": FormatSpec(
        "gguf",
        "GGUF",
        None,
        "gguf",
        None,
        "gguf",
        ("diffusion", "clip"),
        "Weight-only; low-VRAM compatibility tier",
    ),
    "nvfp4": FormatSpec(
        "nvfp4",
        "NVFP4",
        4,
        "comfy_kitchen",
        (10, 0),
        "comfy-kitchen",
        notes="Requires cu130/cublas and Blackwell; experimental quality tier",
    ),
    "mxfp8": FormatSpec(
        "mxfp8",
        "MXFP8",
        8,
        "comfy_kitchen",
        (10, 0),
        "comfy-kitchen",
    ),
    "fp8_e4m3": FormatSpec(
        "fp8_e4m3",
        "FP8 E4M3",
        8,
        "comfy_kitchen",
        (8, 9),
        "comfy-kitchen",
    ),
    "fp8_e5m2": FormatSpec(
        "fp8_e5m2",
        "FP8 E5M2",
        8,
        "comfy_kitchen",
        (8, 9),
        "comfy-kitchen",
    ),
    "nf4": FormatSpec(
        "nf4",
        "NF4",
        4,
        "bitsandbytes",
        (8, 0),
        "bitsandbytes",
        notes="Weight-only; text-encoder/low-VRAM tier",
    ),
}

_ALIASES = {
    "int8": "int8_w8a8",
    "w8a8": "int8_w8a8",
    "int8_tensorwise": "int8_w8a8",
    "int8_convrot": "int8_convrot_w4a4",
    "convrot": "int8_convrot_w4a4",
    "convrot_w4a4": "int8_convrot_w4a4",
    "float8_e4m3fn": "fp8_e4m3",
    "float8_e4m3": "fp8_e4m3",
    "e4m3fn": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2",
    "e5m2": "fp8_e5m2",
    "mxfp8_e4m3": "mxfp8",
    "nf4_4bit": "nf4",
}


def _package_available(name: str | None) -> bool:
    if not name:
        return True
    module = name.replace("-", "_")
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def normalize_format(value: Any) -> str:
    """Normalize a checkpoint-declared format name."""
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw or raw in {"none", "float16", "bfloat16", "float32"}:
        return "none"
    return _ALIASES.get(raw, raw)


def _read_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _safetensors_header(path: Path) -> tuple[dict[str, Any], int] | None:
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return None
            length = struct.unpack("<Q", raw)[0]
            if length <= 0 or length > MAX_HEADER_BYTES:
                return None
            payload = handle.read(length)
            if len(payload) != length:
                return None
        header = json.loads(payload)
    except (OSError, ValueError, TypeError):
        return None
    return (header, length) if isinstance(header, dict) else None


def _tensor_payload(path: Path, header_length: int, entry: dict[str, Any]) -> bytes | None:
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        return None
    try:
        start, end = int(offsets[0]), int(offsets[1])
    except (TypeError, ValueError):
        return None
    if start < 0 or end < start or end - start > MAX_HEADER_BYTES:
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(8 + header_length + start)
            return handle.read(end - start)
    except OSError:
        return None


def _quant_json_from_payload(payload: bytes | None) -> dict[str, Any] | None:
    if not payload:
        return None
    try:
        decoded = payload.decode("utf-8").strip("\x00")
        value = json.loads(decoded)
    except (UnicodeDecodeError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _info_from_metadata(metadata: dict[str, Any], source: str) -> QuantizationInfo:
    options = metadata.get("options")
    options = options if isinstance(options, dict) else {}
    fmt = normalize_format(metadata.get("format") or metadata.get("quant_format"))
    spec = FORMAT_SPECS.get(fmt)
    runtime = str(metadata.get("runtime") or (spec.runtime if spec else "unknown"))
    warnings: list[str] = []
    if fmt == "none":
        warnings.append("Checkpoint metadata declares no quantization format")
    if spec is None and fmt != "none":
        warnings.append(f"Unknown quantization format: {fmt}")
    if spec and spec.package and not _package_available(spec.package):
        warnings.append(f"Runtime package '{spec.package}' is not installed")
    excluded = metadata.get("excluded_layers") or options.get("excluded_layers") or []
    if not isinstance(excluded, list):
        excluded = []
    return QuantizationInfo(
        format=fmt,
        label=spec.label if spec else fmt,
        runtime=runtime,
        source=source,
        mode="pre_quantized",
        variant=_optional_str(metadata.get("variant") or options.get("variant")),
        weight_bits=spec.bits if spec else None,
        group_size=_optional_int(metadata.get("group_size") or options.get("group_size")),
        block_size=_optional_int(metadata.get("block_size") or options.get("block_size")),
        scale_dtype=_optional_str(metadata.get("scale_dtype") or options.get("scale_dtype")),
        compute_dtype=_optional_str(metadata.get("compute_dtype")),
        activation=_optional_str(metadata.get("activation")),
        accumulation=_optional_str(metadata.get("accumulation")),
        quantized_layer_count=_optional_int(metadata.get("quantized_layer_count")),
        excluded_layers=[str(item)[:120] for item in excluded[:256]],
        valid=fmt != "none" and spec is not None,
        warnings=warnings[:32],
    )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:80] or None


def _detect_safetensors(path: Path) -> QuantizationInfo | None:
    header_result = _safetensors_header(path)
    if header_result is None:
        return None
    header, header_length = header_result
    quant_entries = [
        name
        for name in header
        if name == "comfy_quant"
        or name.endswith(".comfy_quant")
        or name.endswith(".quantization")
    ]
    for name in quant_entries[:MAX_METADATA_FILES]:
        entry = header.get(name)
        if not isinstance(entry, dict):
            continue
        metadata = _quant_json_from_payload(
            _tensor_payload(path, header_length, entry)
        )
        if metadata:
            info = _info_from_metadata(metadata, "comfy_quant")
            if info.quantized_layer_count is None:
                info = info.model_copy(
                    update={"quantized_layer_count": len(quant_entries)}
                )
            return info
    sidecar = path.with_name(path.name + ".quant.json")
    if not sidecar.is_file():
        sidecar = path.with_suffix(".quant.json")
    if sidecar.is_file():
        metadata = _read_json(sidecar)
        if isinstance(metadata, dict):
            return _info_from_metadata(metadata, "sidecar")
    dtypes = {
        str(entry.get("dtype", "")).upper()
        for entry in header.values()
        if isinstance(entry, dict)
    }
    if "F8_E4M3" in dtypes or "F8_E5M2" in dtypes:
        fmt = "fp8_e4m3" if "F8_E4M3" in dtypes else "fp8_e5m2"
        info = _info_from_metadata({"format": fmt}, "scaled_fp8")
        if info.quantized_layer_count is None:
            info = info.model_copy(
                update={
                    "quantized_layer_count": sum(
                        1
                        for name in header
                        if "scale" in name.lower() or "weight" in name.lower()
                    )
                }
            )
        return info
    return None


def _detect_gguf(path: Path) -> QuantizationInfo:
    metadata = _gguf_metadata(path)
    file_type = metadata.get("general.file_type")
    variant = _GGUF_FILE_TYPES.get(file_type) if isinstance(file_type, int) else None
    bits = None
    if variant:
        digits = ""
        for character in variant[1:]:
            if character.isdigit():
                digits += character
            else:
                break
        if digits:
            bits = max(1, min(32, int(digits)))
    warnings: list[str] = []
    if not _package_available("gguf"):
        warnings.append("GGUF runtime package is not installed")
    architecture = metadata.get("general.architecture")
    if architecture:
        warnings.append(f"GGUF architecture: {architecture}")
    if metadata.get("general.file_type") is None:
        warnings.append("GGUF header does not declare general.file_type")
    return QuantizationInfo(
        format="gguf",
        label=f"GGUF {variant}" if variant else "GGUF",
        runtime="gguf",
        source="gguf_header",
        mode="pre_quantized",
        variant=variant,
        weight_bits=bits,
        activation="weight_only",
        valid=_package_available("gguf") and variant is not None,
        warnings=warnings[:32],
    )


_GGUF_VALUE_SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}
_GGUF_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q8_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4",
}
_GGUF_KEYS = {"general.architecture", "general.file_type"}


def _gguf_read(handle, size: int) -> bytes:
    if size < 0 or size > MAX_METADATA_BYTES:
        raise ValueError("GGUF metadata value is too large")
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("GGUF header is truncated")
    return payload


def _gguf_string(handle) -> str:
    (length,) = struct.unpack("<Q", _gguf_read(handle, 8))
    if length > 1_048_576:
        raise ValueError("GGUF string is too large")
    return _gguf_read(handle, length).decode("utf-8", "replace")


def _gguf_skip(handle, value_type: int) -> None:
    size = _GGUF_VALUE_SIZES.get(value_type)
    if size is not None:
        _gguf_read(handle, size)
        return
    if value_type == 8:
        _gguf_string(handle)
        return
    if value_type == 9:
        array_type, count = struct.unpack("<IQ", _gguf_read(handle, 12))
        if count > MAX_GGUF_ITEMS:
            raise ValueError("GGUF metadata array is too large")
        if array_type == 8:
            for _ in range(count):
                _gguf_string(handle)
                if handle.tell() > MAX_METADATA_BYTES:
                    raise ValueError("GGUF metadata array is too large")
        elif array_type in _GGUF_VALUE_SIZES:
            _gguf_read(handle, _GGUF_VALUE_SIZES[array_type] * count)
        else:
            raise ValueError(f"Unsupported GGUF array type: {array_type}")
        return
    raise ValueError(f"Unsupported GGUF metadata type: {value_type}")


def _gguf_metadata(path: Path) -> dict[str, Any]:
    """Read bounded top-level GGUF metadata without loading tensor data."""
    try:
        with path.open("rb") as handle:
            magic, version = struct.unpack("<II", _gguf_read(handle, 8))
            if magic != 0x46554747 or version not in {2, 3}:
                return {}
            struct.unpack("<Q", _gguf_read(handle, 8))  # tensor count
            (kv_count,) = struct.unpack("<Q", _gguf_read(handle, 8))
            if kv_count > MAX_GGUF_ITEMS:
                return {}
            metadata: dict[str, Any] = {}
            for _ in range(int(kv_count)):
                if handle.tell() > MAX_METADATA_BYTES:
                    return metadata
                key = _gguf_string(handle)
                (value_type,) = struct.unpack("<I", _gguf_read(handle, 4))
                if key in _GGUF_KEYS and value_type == 8:
                    metadata[key] = _gguf_string(handle)
                elif key in _GGUF_KEYS and value_type in {4, 5, 10, 11}:
                    raw = _gguf_read(handle, _GGUF_VALUE_SIZES[value_type])
                    signed = value_type in {5, 11}
                    metadata[key] = int.from_bytes(
                        raw, "little", signed=signed
                    )
                else:
                    _gguf_skip(handle, value_type)
                if _GGUF_KEYS <= metadata.keys():
                    break
            return metadata
    except (OSError, ValueError, struct.error):
        return {}


def _weight_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in QUANT_SUFFIXES else []
    files: list[Path] = []
    for name in _COMPONENT_DIRS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.safetensors")):
            if path.is_file() and not path.is_symlink():
                files.append(path)
                if len(files) >= MAX_WEIGHT_FILES:
                    return files
    if not files:
        for path in sorted(root.glob("*.safetensors")):
            if path.is_file() and not path.is_symlink():
                files.append(path)
                if len(files) >= MAX_WEIGHT_FILES:
                    break
    return files


def _file_parameter_count(path: Path) -> int | None:
    header_result = _safetensors_header(path)
    if header_result is None:
        return None
    header, _ = header_result
    total = 0
    found = False
    for name, entry in header.items():
        if not isinstance(entry, dict):
            continue
        if name.endswith(("comfy_quant", "quantization")):
            continue
        shape = entry.get("shape")
        if not isinstance(shape, list) or not shape:
            continue
        try:
            count = math.prod(int(value) for value in shape)
        except (TypeError, ValueError):
            continue
        if count > 0:
            total += count
            found = True
    return total if found else None


def read_parameter_count(path: Path) -> int | None:
    """Count logical elements from SafeTensors headers (no tensor data)."""
    if not path.exists():
        return None
    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            return None
        return _file_parameter_count(path)
    total = 0
    found = False
    for file in _weight_files(path):
        count = _file_parameter_count(file)
        if count is not None:
            total += count
            found = True
    return total if found else None


def detect_quantization(path: Path, model_type: str = "") -> QuantizationInfo | None:
    """Inspect a checkpoint or component folder for pre-quantization metadata."""
    if not path.exists():
        return None
    if path.is_file() and path.suffix.lower() == ".gguf":
        return _detect_gguf(path)
    files = _weight_files(path)
    for file in files:
        info = _detect_safetensors(file)
        if info is not None:
            return info
    return None


def validate_quantization(
    info: QuantizationInfo | None,
    *,
    model_type: str,
    device: str = "cuda",
    capability: tuple[int, int] | None = None,
) -> QuantizationInfo | None:
    """Validate discovered metadata against runtime and component support."""
    if info is None or info.format == "none":
        return info
    spec = FORMAT_SPECS.get(info.format)
    if spec is None:
        raise QuantizationError(
            f"Unsupported quantization format '{info.format}' for {model_type}"
        )
    if model_type and model_type not in spec.components:
        raise QuantizationError(
            f"Quantization format '{spec.label}' is not supported for {model_type}"
        )
    if spec.package and not _package_available(spec.package):
        raise QuantizationError(
            f"Quantization format '{spec.label}' needs the '{spec.package}' "
            "runtime; install it in the backend environment"
        )
    if device == "cuda" and spec.min_cc is not None:
        if capability is None or (
            capability[0] < spec.min_cc[0]
            or (capability[0] == spec.min_cc[0] and capability[1] < spec.min_cc[1])
        ):
            raise QuantizationError(
                f"Quantization format '{spec.label}' requires a GPU with "
                f"compute capability {spec.min_cc[0]}.{spec.min_cc[1]} or newer"
            )
    return info


def require_float_checkpoint(
    path: Path,
    *,
    model_type: str,
    device: str = "cuda",
    capability: tuple[int, int] | None = None,
) -> QuantizationInfo | None:
    """Gate float-only loaders before they touch pre-quantized weights.

    Returns the detected metadata for float checkpoints (``None`` when no
    quantization metadata exists) and raises an actionable error otherwise.
    Runtime and architecture gates run first so a missing kernel package is
    reported before the loader limitation.
    """
    info = detect_quantization(path, model_type)
    if info is None or info.format == "none":
        return info
    validate_quantization(
        info,
        model_type=model_type,
        device=device,
        capability=capability,
    )
    raise QuantizationError(
        f"Pre-quantized {info.label} checkpoint detected. This loader keeps "
        "single-file checkpoints on the float path; use Load Diffusion Model "
        "or Load CLIP with a diffusers component folder containing "
        ".comfy_quant metadata for native quantized loading, or choose a "
        "float32/float16/bfloat16 checkpoint."
    )


@lru_cache(maxsize=1)
def quantization_health() -> dict[str, Any]:
    return {
        "formats": {
            key: {
                "label": spec.label,
                "bits": spec.bits,
                "runtime": spec.runtime,
                "package": spec.package,
                "installed": _package_available(spec.package),
                "components": list(spec.components),
                "notes": spec.notes,
            }
            for key, spec in FORMAT_SPECS.items()
        }
    }
