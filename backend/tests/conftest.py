"""Shared pytest configuration."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ["COMPLEXUI_LOG_DIR"] = "0"


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from app.datasets import get_dataset_store, get_output_store
    from app.models import get_model_store
    from app.run_history import get_run_history_store

    monkeypatch.setenv("COMPLEXUI_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("COMPLEXUI_INPUT_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("COMPLEXUI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("COMPLEXUI_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("COMPLEXUI_ATTENTION", "auto")
    monkeypatch.setenv("COMPLEXUI_MEMORY_PROFILE", "resident")
    monkeypatch.setenv("COMPLEXUI_TORCH_COMPILE", "off")
    monkeypatch.delenv("COMPLEXUI_EXPERIMENTAL_FP16_ACCUM", raising=False)
    get_run_history_store.cache_clear()
    get_dataset_store.cache_clear()
    get_output_store.cache_clear()
    get_model_store.cache_clear()
    yield
    get_run_history_store.cache_clear()
    get_dataset_store.cache_clear()
    get_output_store.cache_clear()
    get_model_store.cache_clear()
