"""Shared pytest configuration."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ["COMPLEXUI_LOG_DIR"] = "0"


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from app.run_history import get_run_history_store

    monkeypatch.setenv("COMPLEXUI_WORKSPACE", str(tmp_path / "workspace"))
    get_run_history_store.cache_clear()
    yield
    get_run_history_store.cache_clear()
