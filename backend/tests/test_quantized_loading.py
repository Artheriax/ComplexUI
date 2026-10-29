"""Contract tests for Comfy-format pre-quantized component loading."""

from __future__ import annotations

import json
from pathlib import Path

import comfy_kitchen as ck
import pytest
import torch
from safetensors.torch import save_file

from app.quantized_loading import (
    QuantizedEmbedding,
    QuantizedLinear,
    QuantizedLoadingError,
    load_quantized_component,
)
from app.security import get_security_registry
from app.transpiler import transpile_node


def _marker(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config).encode("utf-8")), dtype=torch.uint8
    )


def _write_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file(state, path / "model.safetensors")


def _int8_state(
    weight: torch.Tensor, bias: torch.Tensor | None, *, emulated: bool = False
) -> dict[str, torch.Tensor]:
    quantized, scale = ck.quantize_int8_rowwise(weight.float())
    config = {"format": "int8_tensorwise"}
    if emulated:
        config["full_precision_matrix_mult"] = True
    state = {
        "0.weight": quantized,
        "0.weight_scale": scale.float(),
        "0.comfy_quant": _marker(config),
    }
    if bias is not None:
        state["0.bias"] = bias
    return state


def test_int8_linear_loads_native_and_forwards(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(32, 16))
    weight = model[0].weight.detach().clone()
    _write_state(
        tmp_path,
        _int8_state(weight, model[0].bias.detach().clone()),
    )

    loaded, report = load_quantized_component(
        model, tmp_path, compute_dtype=torch.float32
    )

    layer = loaded[0]
    assert isinstance(layer, QuantizedLinear)
    assert report[0].native is True
    query = torch.randn(5, 32)
    torch.testing.assert_close(
        layer(query),
        torch.nn.functional.linear(query, weight),
        rtol=0.1,
        atol=0.25,
    )


def test_int8_full_precision_matrix_mult_dequantizes(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Linear(32, 16))
    weight = model[0].weight.detach().clone()
    _write_state(
        tmp_path,
        _int8_state(
            weight,
            model[0].bias.detach().clone(),
            emulated=True,
        ),
    )

    loaded, report = load_quantized_component(
        model, tmp_path, compute_dtype=torch.float32
    )

    assert type(loaded[0]) is torch.nn.Linear
    assert report[0].native is False
    assert report[0].runtime == "dequantized"
    assert loaded[0].weight.dtype == torch.float32


def test_convrot_linear_loads_and_runs(tmp_path: Path) -> None:
    torch.manual_seed(6)
    model = torch.nn.Sequential(torch.nn.Linear(256, 32))
    weight = model[0].weight.detach().clone()
    quantized, scale = ck.quantize_convrot_w4a4_weight(weight.float())
    config = {"format": "convrot_w4a4"}
    _write_state(
        tmp_path,
        {
            "0.weight": quantized,
            "0.weight_scale": scale.float(),
            "0.comfy_quant": _marker(config),
            "0.bias": model[0].bias.detach().clone(),
        },
    )

    loaded, report = load_quantized_component(
        model, tmp_path, compute_dtype=torch.float32
    )

    assert isinstance(loaded[0], QuantizedLinear)
    assert report[0].native is True
    query = torch.randn(2, 256)
    if torch.cuda.is_available():
        loaded = loaded.to("cuda")
        query = query.to("cuda")
    result = loaded[0](query)
    assert result.shape == (2, 32)
    assert torch.isfinite(result).all()


def test_quantized_embedding_loads(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Embedding(16, 8))
    weight = model[0].weight.detach().clone()
    quantized, scale = ck.quantize_int8_rowwise(weight.float())
    _write_state(
        tmp_path,
        {
            "0.weight": quantized,
            "0.weight_scale": scale.float(),
            "0.comfy_quant": _marker({"format": "int8_tensorwise"}),
        },
    )

    loaded, report = load_quantized_component(
        model, tmp_path, compute_dtype=torch.float32
    )

    assert isinstance(loaded[0], QuantizedEmbedding)
    assert report[0].native is True
    assert loaded[0](torch.tensor([1, 3])).shape == (2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 needs CUDA")
def test_nvfp4_linear_loads_on_cuda(tmp_path: Path) -> None:
    torch.manual_seed(8)
    model = torch.nn.Sequential(torch.nn.Linear(64, 32))
    weight = model[0].weight.detach().clone().to("cuda", torch.float16)
    scale = (torch.amax(weight.abs()) / (448.0 * 6.0)).float()
    quantized, block_scale = ck.quantize_nvfp4(weight, scale, pad_16x=False)
    _write_state(
        tmp_path,
        {
            "0.weight": quantized.cpu(),
            "0.weight_scale": block_scale.view(torch.uint8).cpu(),
            "0.weight_scale_2": scale.cpu(),
            "0.comfy_quant": _marker({"format": "nvfp4"}),
            "0.bias": model[0].bias.detach().to(torch.float16),
        },
    )

    loaded, report = load_quantized_component(
        model, tmp_path, compute_dtype=torch.float16
    )
    loaded = loaded.to("cuda")
    result = loaded[0](torch.randn(2, 64, device="cuda", dtype=torch.float16))

    assert report[0].native is False
    assert report[0].runtime == "quantized_storage"
    assert result.shape == (2, 32)
    assert torch.isfinite(result).all()


def test_pre_quant_scale_is_applied(tmp_path: Path) -> None:
    torch.manual_seed(10)
    model = torch.nn.Sequential(torch.nn.Linear(16, 8))
    weight = model[0].weight.detach().clone()
    quantized, scale = ck.quantize_int8_rowwise(weight.float())
    pre_scale = torch.full((16,), 2.0)
    _write_state(
        tmp_path,
        {
            "0.weight": quantized,
            "0.weight_scale": scale,
            "0.pre_quant_scale": pre_scale,
            "0.comfy_quant": _marker({"format": "int8_tensorwise"}),
            "0.bias": model[0].bias.detach().clone(),
        },
    )

    loaded, _ = load_quantized_component(
        model, tmp_path, compute_dtype=torch.float32
    )

    query = torch.randn(2, 16)
    torch.testing.assert_close(
        loaded[0](query),
        torch.nn.functional.linear(query * pre_scale, weight),
        rtol=0.1,
        atol=0.3,
    )


def test_legacy_scaled_fp8_linear_loads(tmp_path: Path) -> None:
    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(32, 16))
    weight = model[0].weight.detach().clone()
    scale = torch.tensor(1.0)
    quantized = ck.quantize_per_tensor_fp8(
        weight.float(), scale, torch.float8_e4m3fn
    )
    _write_state(
        tmp_path,
        {
            "0.weight": quantized,
            "0.weight_scale": scale,
            "0.bias": model[0].bias.detach().clone(),
        },
    )

    loaded, report = load_quantized_component(
        model, tmp_path, compute_dtype=torch.float32
    )

    assert isinstance(loaded[0], QuantizedLinear)
    assert report[0].format == "float8_e4m3fn"
    assert loaded[0](torch.randn(2, 32)).shape == (2, 16)


def test_missing_marker_is_rejected(tmp_path: Path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    save_file({"0.weight": torch.zeros(4, 4)}, tmp_path / "model.safetensors")

    with pytest.raises(QuantizedLoadingError, match="comfy_quant"):
        load_quantized_component(model, tmp_path, compute_dtype=torch.float32)


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    _write_state(
        tmp_path,
        {
            "0.weight": torch.zeros(4, 4, dtype=torch.int8),
            "0.comfy_quant": _marker({"format": "asym_w4a8_int8"}),
        },
    )

    with pytest.raises(QuantizedLoadingError, match="Unsupported quantization format"):
        load_quantized_component(model, tmp_path, compute_dtype=torch.float32)


def test_diffusion_template_loads_quantized_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from diffusers import UNet2DConditionModel

    root = tmp_path / "repo"
    component = root / "unet"
    component.mkdir(parents=True)
    source = UNet2DConditionModel(
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
    source.save_pretrained(component)
    state = source.state_dict()
    for name, module in source.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = state.pop(f"{name}.weight")
        quantized, scale = ck.quantize_int8_rowwise(weight.float())
        state[f"{name}.weight"] = quantized
        state[f"{name}.weight_scale"] = scale.float()
        state[f"{name}.comfy_quant"] = _marker(
            {"format": "int8_tensorwise"}
        )
    for file in component.glob("*.safetensors"):
        file.unlink()
    save_file(state, component / "diffusion_pytorch_model.safetensors")
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "unet": ["diffusers", "UNet2DConditionModel"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COMPLEXUI_MEMORY_PROFILE", "resident")
    monkeypatch.setenv("COMPLEXUI_ATTENTION", "sdpa-math")
    record = get_security_registry().get_template("load_diffusion_model")
    loader = transpile_node(record.code, "load_diffusion_model").func

    (model,) = loader(
        unet_name=str(root),
        type="sd15",
        weight_dtype="fp32",
        allow_custom_code=False,
    )

    assert model["quantization"]["format"] == "int8_w8a8"
    assert isinstance(
        model["unet"].time_embedding.linear_1, QuantizedLinear
    )


def test_clip_quantized_component_helper(tmp_path: Path) -> None:
    from transformers import CLIPTextConfig, CLIPTextModel

    component = tmp_path / "text_encoder"
    component.mkdir()
    config = CLIPTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=100,
        max_position_embeddings=32,
    )
    source = CLIPTextModel(config)
    state = source.state_dict()
    for name, module in source.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = state.pop(f"{name}.weight")
        quantized, scale = ck.quantize_int8_rowwise(weight.float())
        state[f"{name}.weight"] = quantized
        state[f"{name}.weight_scale"] = scale.float()
        state[f"{name}.comfy_quant"] = _marker(
            {"format": "int8_tensorwise"}
        )
    save_file(state, component / "model.safetensors")

    model, report = load_quantized_component(
        CLIPTextModel(config),
        component,
        compute_dtype=torch.float32,
    )

    assert report
    assert all(entry.native for entry in report)
    assert isinstance(
        model.encoder.layers[0].self_attn.q_proj,
        QuantizedLinear,
    )
