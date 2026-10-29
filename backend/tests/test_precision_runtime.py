"""Precision tracking and guarded accumulation contracts."""

from __future__ import annotations

import pytest
import torch

from app.security import get_security_registry
from app.transpiler import transpile_node


def _func(name: str):
    record = get_security_registry().get_template(name)
    return transpile_node(record.code, name).func


def _model(**overrides) -> dict:
    transformer = overrides.pop("transformer", torch.nn.Linear(4, 4))
    value = {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "storage_dtype": "fp32",
        "transformer": transformer,
        "unet": transformer,
    }
    value.update(overrides)
    return value


def test_precision_converts_weights_and_tracks_storage() -> None:
    (model,) = _func("model_precision")(
        model=_model(),
        compute_dtype="bf16",
        accumulation="default",
    )

    assert model["transformer"].weight.dtype == torch.bfloat16
    assert model["dtype"] is torch.bfloat16
    assert model["compute_dtype"] == str(torch.bfloat16)
    assert model["storage_dtype"] == "fp32"
    assert model["precision"]["accumulation"] == "default"


def test_precision_keep_preserves_weights() -> None:
    (model,) = _func("model_precision")(
        model=_model(),
        compute_dtype="keep",
        accumulation="default",
    )

    assert model["transformer"].weight.dtype == torch.float32
    assert model["dtype"] is torch.float32


def test_fp16_on_cpu_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        _func("model_precision")(
            model=_model(),
            compute_dtype="fp16",
            accumulation="default",
        )


def test_fp16_accumulation_needs_explicit_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPLEXUI_EXPERIMENTAL_FP16_ACCUM", raising=False)

    with pytest.raises(ValueError, match="experimental"):
        _func("model_precision")(
            model=_model(),
            compute_dtype="keep",
            accumulation="fp16",
        )


def test_fp16_accumulation_sets_backend_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPLEXUI_EXPERIMENTAL_FP16_ACCUM", "1")

    (model,) = _func("model_precision")(
        model=_model(),
        compute_dtype="keep",
        accumulation="fp16",
    )

    assert model["accumulation"] == "fp16"
    assert (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        is True
    )


def test_precision_after_compile_is_rejected() -> None:
    with pytest.raises(ValueError, match="before Compile Model"):
        _func("model_precision")(
            model=_model(compiled=True),
            compute_dtype="bf16",
            accumulation="default",
        )
