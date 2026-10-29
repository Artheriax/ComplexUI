"""Contract tests for pre-quantized checkpoint metadata (no real checkpoints)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from app import quantization
from app.models import ModelStore
from app.quantization import (
    QuantizationError,
    detect_quantization,
    normalize_format,
    quantization_health,
    read_parameter_count,
    validate_quantization,
)
from app.schemas import QuantizationInfo


def _quant_payload(value: dict) -> torch.Tensor:
    encoded = json.dumps(value).encode("utf-8")
    return torch.tensor(list(encoded), dtype=torch.uint8)


def _write_comfy_checkpoint(path: Path, config: dict) -> None:
    save_file(
        {
            "model.diffusion_model.layer.weight": torch.zeros(
                (8, 16), dtype=torch.int8
            ),
            "model.diffusion_model.layer.weight_scale": torch.ones(
                (8, 1), dtype=torch.float16
            ),
            "model.diffusion_model.layer.comfy_quant": _quant_payload(config),
        },
        path,
    )


def _write_gguf(path: Path, *, architecture: str, file_type: int) -> None:
    def text(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    payload = struct.pack("<IIQQ", 0x46554747, 3, 0, 2)
    payload += text("general.architecture") + struct.pack("<I", 8) + text(
        architecture
    )
    payload += text("general.file_type") + struct.pack("<I", 4) + struct.pack(
        "<I", file_type
    )
    path.write_bytes(payload)


def test_comfy_quant_metadata_is_detected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "int8.safetensors"
    _write_comfy_checkpoint(
        checkpoint,
        {
            "format": "int8",
            "options": {"group_size": 128, "activation": "dynamic"},
        },
    )

    info = detect_quantization(checkpoint, "diffusion")

    assert info is not None
    assert info.format == "int8_w8a8"
    assert info.label == "INT8 W8A8"
    assert info.runtime == "comfy_kitchen"
    assert info.source == "comfy_quant"
    assert info.weight_bits == 8
    assert info.group_size == 128
    assert info.quantized_layer_count == 1
    assert info.valid is True


def test_scaled_fp8_dtype_is_detected_as_legacy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fp8.safetensors"
    save_file(
        {
            "model.diffusion_model.layer.weight": torch.zeros(
                (4, 4), dtype=torch.float8_e4m3fn
            ),
            "model.diffusion_model.layer.weight_scale": torch.ones(
                (1,), dtype=torch.float32
            ),
        },
        checkpoint,
    )

    info = detect_quantization(checkpoint, "diffusion")

    assert info is not None
    assert info.format == "fp8_e4m3"
    assert info.source == "scaled_fp8"
    assert info.weight_bits == 8


def test_sidecar_metadata_is_used(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.zeros((2, 2))}, checkpoint)
    sidecar = tmp_path / "model.safetensors.quant.json"
    sidecar.write_text(
        json.dumps({"format": "nvfp4", "runtime": "comfy_kitchen"}),
        encoding="utf-8",
    )

    info = detect_quantization(checkpoint, "diffusion")

    assert info is not None
    assert info.format == "nvfp4"
    assert info.source == "sidecar"


def test_gguf_header_variant_and_architecture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model-Q4_K_M.gguf"
    _write_gguf(checkpoint, architecture="flux", file_type=15)

    info = detect_quantization(checkpoint, "diffusion")

    assert info is not None
    assert info.format == "gguf"
    assert info.variant == "Q4_K_M"
    assert info.weight_bits == 4
    assert info.source == "gguf_header"
    assert "GGUF architecture: flux" in info.warnings
    assert info.valid is True


def test_parameter_count_reads_headers_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "params.safetensors"
    save_file(
        {
            "a": torch.zeros((2, 3)),
            "b": torch.zeros((4,), dtype=torch.float16),
            "layer.comfy_quant": _quant_payload({"format": "int8"}),
        },
        checkpoint,
    )

    assert read_parameter_count(checkpoint) == 10


def test_folder_quantization_scans_component_directories(tmp_path: Path) -> None:
    folder = tmp_path / "diffusers-model"
    component = folder / "transformer"
    component.mkdir(parents=True)
    _write_comfy_checkpoint(
        component / "weights.safetensors",
        {"format": "convrot", "options": {"convrot_groupsize": 256}},
    )

    info = detect_quantization(folder, "diffusion")

    assert info is not None
    assert info.format == "int8_convrot_w4a4"
    assert info.label == "INT8 ConvRot W4A4"


def test_model_summary_carries_quantization_and_memory(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "library")
    checkpoint = store.root / "diffusion" / "int8.safetensors"
    _write_comfy_checkpoint(
        checkpoint, {"format": "int8", "options": {"group_size": 64}}
    )

    summary = store.describe("diffusion", "diffusion/int8.safetensors")

    assert summary.quantization is not None
    assert summary.quantization.format == "int8_w8a8"
    assert summary.parameter_count is None
    assert summary.memory is not None
    assert summary.memory.weights_bytes == checkpoint.stat().st_size
    assert summary.memory.unknown is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("int8", "int8_w8a8"),
        ("W8A8", "int8_w8a8"),
        ("convrot", "int8_convrot_w4a4"),
        ("float8_e4m3fn", "fp8_e4m3"),
        ("mxfp8_e4m3", "mxfp8"),
        ("none", "none"),
    ],
)
def test_normalize_format_aliases(raw: str, expected: str) -> None:
    assert normalize_format(raw) == expected


def test_validation_rejects_unknown_and_incompatible_formats() -> None:
    unknown = QuantizationInfo(format="made-up")
    with pytest.raises(QuantizationError, match="Unsupported quantization format"):
        validate_quantization(unknown, model_type="diffusion")

    gguf = QuantizationInfo(format="gguf", runtime="gguf")
    with pytest.raises(QuantizationError, match="not supported for vae"):
        validate_quantization(gguf, model_type="vae")


def test_validation_gates_blackwell_formats() -> None:
    info = QuantizationInfo(format="nvfp4", runtime="comfy_kitchen")

    with pytest.raises(QuantizationError, match="compute capability 10.0"):
        validate_quantization(
            info, model_type="diffusion", device="cuda", capability=(8, 9)
        )

    assert (
        validate_quantization(
            info, model_type="diffusion", device="cuda", capability=(12, 0)
        )
        is info
    )


def test_validation_reports_missing_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quantization, "_package_available", lambda _name: False)
    info = QuantizationInfo(format="int8_w8a8", runtime="comfy_kitchen")

    with pytest.raises(QuantizationError, match="comfy-kitchen"):
        validate_quantization(info, model_type="diffusion", device="cpu")


def test_require_float_checkpoint_accepts_float_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "float.safetensors"
    save_file({"weight": torch.zeros((2, 2), dtype=torch.float16)}, checkpoint)

    assert (
        quantization.require_float_checkpoint(
            checkpoint, model_type="diffusion", device="cpu"
        )
        is None
    )


def test_require_float_checkpoint_rejects_prequantized_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "int8.safetensors"
    _write_comfy_checkpoint(checkpoint, {"format": "int8"})

    with pytest.raises(QuantizationError, match="diffusers component folder"):
        quantization.require_float_checkpoint(
            checkpoint, model_type="diffusion", device="cpu"
        )


def test_health_lists_every_format_with_install_state() -> None:
    health = quantization_health()

    assert {"int8_w8a8", "int8_convrot_w4a4", "gguf", "nvfp4"} <= set(
        health["formats"]
    )
    assert health["formats"]["gguf"]["installed"] is True
