"""Attention backend selection and correctness contracts."""

from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn.functional as F

from app import attention_dispatch, optimizations
from app.security import get_security_registry
from app.transpiler import parse_node, transpile_node


def _func(name: str):
    record = get_security_registry().get_template(name)
    return transpile_node(record.code, name).func


def _model() -> dict:
    return {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "transformer": torch.nn.Linear(2, 2),
    }


def test_attention_backend_widget_matches_contract() -> None:
    record = get_security_registry().get_template("attention_backend")
    schema = parse_node(record.code, "attention_backend")

    assert set(schema.widgets[0].options) - {"server"} == set(
        optimizations.ATTENTION_BACKENDS
    )


def test_auto_attention_on_cpu_selects_sdpa() -> None:
    (model,) = _func("attention_backend")(model=_model(), backend="auto")

    assert model["attention"]["backend"] == "sdpa"
    assert model["attention"]["approximates"] is False
    assert torch.backends.cuda.math_sdp_enabled() is True


def test_explicit_math_attention_selects_math() -> None:
    (model,) = _func("attention_backend")(model=_model(), backend="sdpa-math")

    assert model["attention"]["backend"] == "sdpa-math"
    assert model["attention"]["approximates"] is False
    with attention_dispatch.attention_context(torch, model["attention"]):
        query = torch.randn(1, 2, 4, 16)
        result = F.scaled_dot_product_attention(query, query, query)
        assert result.shape == query.shape


def test_explicit_flash_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is required"):
        _func("attention_backend")(model=_model(), backend="sdpa-flash")


def test_server_mode_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPLEXUI_ATTENTION", "sdpa-math")

    (model,) = _func("attention_backend")(model=_model(), backend="server")

    assert model["attention"]["backend"] == "sdpa-math"


def test_cudnn_requires_hopper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))

    with pytest.raises(ValueError, match="sm90"):
        _func("attention_backend")(model=_model(), backend="sdpa-cudnn")


def test_sdpa_matches_math_with_masks_and_cross_attention() -> None:
    torch.manual_seed(7)
    query = torch.randn(2, 4, 8, 16)
    key = torch.randn(2, 4, 12, 16)
    value = torch.randn(2, 4, 12, 16)
    mask = torch.rand(2, 4, 8, 12) > 0.2

    result = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
    )
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ value

    torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)


@pytest.fixture(autouse=True)
def _reset_dispatch():
    attention_dispatch.reset_attention_dispatch()
    yield
    attention_dispatch.reset_attention_dispatch()


def test_optional_backend_requires_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_dispatch, "_package_available", lambda _name: False)

    with pytest.raises(ValueError, match="not installed"):
        attention_dispatch.resolve_attention(
            "sageattention",
            device="cuda",
            capability=(8, 9),
            arch="ada",
        )


def test_optional_backend_requires_cuda() -> None:
    with pytest.raises(ValueError, match="CUDA is required"):
        attention_dispatch.resolve_attention(
            "xformers",
            device="cpu",
            capability=None,
            arch="cpu",
        )


def test_flash_attention_rejects_blackwell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_dispatch, "_package_available", lambda _name: True)

    with pytest.raises(ValueError, match="sm120"):
        attention_dispatch.resolve_attention(
            "flash-attn",
            device="cuda",
            capability=(12, 0),
            arch="blackwell",
        )


def test_sageattention_dispatch_and_mask_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class _FakeSage:
        @staticmethod
        def sageattn(query, key, value, **kwargs):
            calls.append({"shape": tuple(query.shape), **kwargs})
            return query

    monkeypatch.setitem(sys.modules, "sageattention", _FakeSage)
    monkeypatch.setattr(attention_dispatch, "_package_available", lambda _name: True)
    monkeypatch.setattr(attention_dispatch, "_package_version", lambda _name: "2.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))

    selection = attention_dispatch.apply_attention(torch, "sageattention", "cuda")
    query = torch.randn(1, 2, 4, 64, dtype=torch.float16)
    key = torch.randn(1, 2, 4, 64, dtype=torch.float16)
    value = torch.randn(1, 2, 4, 64, dtype=torch.float16)
    result = F.scaled_dot_product_attention(query, key, value)

    assert selection["backend"] == "sageattention"
    assert selection["approximates"] is True
    assert result.shape == query.shape
    assert calls and calls[0]["tensor_layout"] == "HND"

    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    fallback = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    assert fallback.shape == query.shape
    assert len(calls) == 1

    cross_key = torch.randn(1, 2, 8, 64, dtype=torch.float16)
    cross_value = torch.randn(1, 2, 8, 64, dtype=torch.float16)
    cross = F.scaled_dot_product_attention(query, cross_key, cross_value)
    assert cross.shape == query.shape
    assert len(calls) == 1


def test_optional_apply_does_not_install_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_dispatch, "_package_available", lambda _name: False)

    with pytest.raises(ValueError, match="xformers"):
        attention_dispatch.apply_attention(torch, "xformers", "cuda")

    assert F.scaled_dot_product_attention is attention_dispatch._ORIGINAL_SDPA


def test_patch_node_reports_missing_optional_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_dispatch, "_package_available", lambda _name: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))

    with pytest.raises(ValueError, match="not installed"):
        _func("attention_backend")(
            model=_model(),
            backend="sageattention",
        )


def test_xformers_adapter_transposes_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import ModuleType

    calls: list[tuple[int, ...]] = []
    ops = ModuleType("xformers.ops")

    def memory_efficient_attention(query, key, value, **_kwargs):
        calls.append(tuple(query.shape))
        return query

    ops.memory_efficient_attention = memory_efficient_attention
    ops.LowerTriangularMask = lambda: None
    package = ModuleType("xformers")
    package.ops = ops
    monkeypatch.setitem(sys.modules, "xformers", package)
    monkeypatch.setitem(sys.modules, "xformers.ops", ops)
    monkeypatch.setattr(attention_dispatch, "_package_available", lambda _name: True)
    monkeypatch.setattr(attention_dispatch, "_package_version", lambda _name: "0.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))

    attention_dispatch.apply_attention(torch, "xformers", "cuda")
    query = torch.randn(1, 2, 4, 64, dtype=torch.float16)
    result = F.scaled_dot_product_attention(query, query, query)

    assert result.shape == query.shape
    assert calls == [(1, 4, 2, 64)]


def test_flash_adapter_transposes_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import ModuleType

    calls: list[tuple[int, ...]] = []
    flash = ModuleType("flash_attn")

    def flash_attn_func(query, key, value, **_kwargs):
        calls.append(tuple(query.shape))
        return query

    flash.flash_attn_func = flash_attn_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash)
    monkeypatch.setattr(attention_dispatch, "_package_available", lambda _name: True)
    monkeypatch.setattr(attention_dispatch, "_package_version", lambda _name: "2.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))

    attention_dispatch.apply_attention(torch, "flash-attn", "cuda")
    query = torch.randn(1, 2, 4, 64, dtype=torch.float16)
    result = F.scaled_dot_product_attention(query, query, query)

    assert result.shape == query.shape
    assert calls == [(1, 4, 2, 64)]


def test_attention_benchmark_returns_a_correct_backend() -> None:
    result = attention_dispatch.benchmark_attention(
        torch,
        "cpu",
        "auto",
        batch=1,
        heads=2,
        sequence=32,
        head_dim=16,
        repeats=1,
    )

    assert result["selected"] in {"sdpa-math", "sdpa"}
    assert any(entry["correct"] for entry in result["results"])


def test_attention_benchmark_node_attaches_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_benchmark(*_args, **_kwargs):
        return {
            "selected": "sdpa-math",
            "results": [{"backend": "sdpa-math", "ms": 1.0, "correct": True}],
            "shapes": {"sequence": 32},
        }

    monkeypatch.setattr(attention_dispatch, "benchmark_attention", fake_benchmark)

    (model,) = _func("attention_backend")(
        model=_model(),
        backend="auto",
        benchmark=True,
    )

    assert model["attention"]["backend"] == "sdpa-math"
    assert model["attention"]["benchmark"]["selected"] == "sdpa-math"


@pytest.mark.parametrize("dim_head", [40, 64])
def test_sageattention_matches_sdpa_on_cuda(dim_head: int) -> None:
    pytest.importorskip("sageattention")
    if not torch.cuda.is_available():
        pytest.skip("SageAttention integration needs CUDA")

    from diffusers.models.attention_processor import Attention

    torch.manual_seed(3)
    attention = Attention(
        query_dim=dim_head,
        heads=1,
        dim_head=dim_head,
        out_dim=dim_head,
        bias=False,
    ).to("cuda", dtype=torch.float16)
    attention.eval()
    value = torch.randn(1, 32, dim_head, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        reference = attention(value)
    attention_dispatch.apply_attention(torch, "sageattention", "cuda")
    with torch.no_grad():
        result = attention(value)

    torch.testing.assert_close(result, reference, rtol=5e-2, atol=5e-2)


def test_model_attention_context_does_not_change_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_optional(backend, query, *_args, **_kwargs):
        calls.append(backend)
        return query

    monkeypatch.setattr(attention_dispatch, "_optional_call", fake_optional)
    attention_dispatch.apply_attention(torch, "sdpa-math", "cpu")
    global_selection = attention_dispatch._GLOBAL_SELECTION
    query = torch.randn(1, 2, 4, 16)
    selection = {
        "backend": "sageattention",
        "device": "cpu",
        "approximates": True,
        "sol_tau": 1.0,
    }

    with attention_dispatch.attention_context(torch, selection):
        result = F.scaled_dot_product_attention(query, query, query)

    assert result.shape == query.shape
    assert calls == ["sageattention"]
    assert attention_dispatch._GLOBAL_SELECTION == global_selection
    F.scaled_dot_product_attention(query, query, query)
    assert calls == ["sageattention"]
