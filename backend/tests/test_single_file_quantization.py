"""Contract tests for single-file pre-quantized compatibility loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import comfy_kitchen as ck
import torch
from diffusers.loaders import single_file
from safetensors.torch import save_file

from app.quantized_loading import dequantize_state_dict
from app.single_file_quantization import (
    apply_single_file_native_unet,
    load_quantized_single_file,
    quantization_mode,
)


def _marker(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config).encode("utf-8")), dtype=torch.uint8
    )


def test_dequantize_state_dict_expands_int8_and_removes_metadata() -> None:
    torch.manual_seed(11)
    weight = torch.randn(16, 32, dtype=torch.float16)
    quantized, scale = ck.quantize_int8_rowwise(weight)
    state = {
        "model.diffusion_model.block.weight": quantized,
        "model.diffusion_model.block.weight_scale": scale,
        "model.diffusion_model.block.comfy_quant": _marker(
            {"format": "int8_tensorwise"}
        ),
        "model.diffusion_model.block.bias": torch.zeros(16),
        "keep.weight": torch.ones(2, 2),
    }

    dequantized, report = dequantize_state_dict(
        state, compute_dtype=torch.float16
    )

    assert dequantized["model.diffusion_model.block.weight"].shape == (16, 32)
    assert dequantized["model.diffusion_model.block.weight"].dtype == torch.float16
    assert "model.diffusion_model.block.weight_scale" not in dequantized
    assert "model.diffusion_model.block.comfy_quant" not in dequantized
    assert dequantized["keep.weight"].shape == (2, 2)
    assert report[0].format == "int8_tensorwise"
    assert report[0].native is False


def test_load_quantized_single_file_uses_dequantized_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(12)
    weight = torch.randn(8, 16, dtype=torch.float16)
    quantized, scale = ck.quantize_int8_rowwise(weight)
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {
            "model.diffusion_model.layer.weight": quantized,
            "model.diffusion_model.layer.weight_scale": scale,
            "model.diffusion_model.layer.comfy_quant": _marker(
                {"format": "int8_tensorwise"}
            ),
        },
        checkpoint,
    )

    class FakePipeline:
        captured = None

        @classmethod
        def from_single_file(cls, path, **kwargs):
            state = single_file.load_single_file_checkpoint(path)
            cls.captured = state
            return SimpleNamespace(unet=state)

    pipeline, report = load_quantized_single_file(
        checkpoint,
        FakePipeline,
        torch_dtype=torch.float16,
    )

    assert pipeline.unet is FakePipeline.captured
    assert FakePipeline.captured["model.diffusion_model.layer.weight"].shape == (
        8,
        16,
    )
    assert not any(
        name.endswith((".weight_scale", ".comfy_quant"))
        for name in FakePipeline.captured
    )
    assert report[0].native is False


def test_native_single_file_unet_mapping() -> None:
    from diffusers import UNet2DConditionModel

    from app.quantized_loading import QuantizedLinear

    unet = UNet2DConditionModel(
        block_out_channels=(8, 16),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=8,
        out_channels=4,
        in_channels=4,
        norm_num_groups=4,
    )
    weight = unet.time_embedding.linear_1.weight.detach().clone()
    quantized, scale = ck.quantize_int8_rowwise(weight)
    state = {
        "model.diffusion_model.time_embed.0.weight": quantized,
        "model.diffusion_model.time_embed.0.weight_scale": scale,
        "model.diffusion_model.time_embed.0.bias": unet.time_embedding.linear_1.bias.detach().clone(),
        "model.diffusion_model.time_embed.0.comfy_quant": _marker(
            {"format": "int8_tensorwise"}
        ),
    }

    report = apply_single_file_native_unet(
        unet, state, compute_dtype=torch.float32
    )

    assert isinstance(unet.time_embedding.linear_1, QuantizedLinear)
    assert report["model.diffusion_model.time_embed.0"].native is True
    assert (
        report["model.diffusion_model.time_embed.0"].module
        == "time_embedding.linear_1"
    )
    assert quantization_mode(list(report.values())) == "pre_quantized"

