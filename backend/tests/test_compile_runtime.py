"""torch.compile patch-node contracts and safe fallback behavior."""

from __future__ import annotations

import pytest
import torch

from app.security import get_security_registry
from app.transpiler import transpile_node


def _func(name: str):
    record = get_security_registry().get_template(name)
    return transpile_node(record.code, name).func


class _Transformer(torch.nn.Module):
    def __init__(self, blocks: int = 0) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4) for _ in range(blocks)]
        )

    def forward(self, value):
        return self.linear(value)


def _model(**overrides) -> dict:
    transformer = overrides.pop("transformer", _Transformer())
    value = {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "transformer": transformer,
        "unet": transformer,
    }
    value.update(overrides)
    return value


@pytest.fixture
def fake_compile(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def compile_fn(function, **kwargs):
        calls.append({"function": function, **kwargs})

        def wrapped(*args, **inner):
            return function(*args, **inner)

        return wrapped

    monkeypatch.setattr(torch, "compile", compile_fn)
    return calls


def test_off_mode_does_not_compile(fake_compile) -> None:
    (model,) = _func("compile_model")(
        model=_model(),
        mode="off",
        regional=False,
        fullgraph=False,
        dynamic="auto",
    )

    assert model["compiled"] is False
    assert model["compile"]["mode"] == "off"
    assert fake_compile == []


def test_default_mode_compiles_forward_in_place(fake_compile) -> None:
    transformer = _Transformer()
    (model,) = _func("compile_model")(
        model=_model(transformer=transformer),
        mode="default",
        regional=False,
        fullgraph=False,
        dynamic="auto",
    )

    assert model["compiled"] is True
    assert model["compile_mode"] == "default"
    assert "forward" in transformer.__dict__
    assert fake_compile[0]["mode"] == "default"
    assert fake_compile[0]["dynamic"] is None


def test_regional_compiles_repeated_blocks(fake_compile) -> None:
    transformer = _Transformer(blocks=2)
    (model,) = _func("compile_model")(
        model=_model(transformer=transformer),
        mode="default",
        regional=True,
        fullgraph=False,
        dynamic="off",
    )

    assert model["compile"]["blocks"] == 2
    assert len(fake_compile) == 2
    assert fake_compile[0]["dynamic"] is False


def test_server_auto_falls_back_to_eager(
    fake_compile, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("unsupported graph")

    monkeypatch.setattr(torch, "compile", explode)
    monkeypatch.setenv("COMPLEXUI_TORCH_COMPILE", "auto")

    (model,) = _func("compile_model")(
        model=_model(),
        mode="server",
        regional=False,
        fullgraph=False,
        dynamic="auto",
    )

    assert model["compiled"] is False
    assert "unsupported graph" in model["compile"]["error"]


def test_explicit_failure_raises(fake_compile, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("unsupported graph")

    monkeypatch.setattr(torch, "compile", explode)

    with pytest.raises(ValueError, match="torch.compile failed"):
        _func("compile_model")(
            model=_model(),
            mode="default",
            regional=False,
            fullgraph=False,
            dynamic="auto",
        )


def test_compile_and_offload_are_rejected(fake_compile) -> None:
    with pytest.raises(ValueError, match="offload"):
        _func("compile_model")(
            model=_model(offload=True),
            mode="default",
            regional=False,
            fullgraph=False,
            dynamic="auto",
        )


def test_server_auto_skips_compile_when_offloaded(
    fake_compile, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPLEXUI_TORCH_COMPILE", "auto")

    (model,) = _func("compile_model")(
        model=_model(offload=True),
        mode="server",
        regional=False,
        fullgraph=False,
        dynamic="auto",
    )

    assert model["compiled"] is False
    assert "offloaded" in model["compile"]["error"]
    assert fake_compile == []
