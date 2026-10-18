"""Tests for the launcher helpers."""

from __future__ import annotations

from pathlib import Path

import launch


def _theme() -> launch.Theme:
    return launch.Theme(False)


def test_ensure_frontend_dependencies_noop_when_installed(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "node_modules").mkdir()
    monkeypatch.setattr(launch, "FRONTEND_DIR", tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("run_streamed must not be called")

    monkeypatch.setattr(launch, "run_streamed", explode)

    assert launch.ensure_frontend_dependencies("npm", _theme()) == 0


def test_ensure_frontend_dependencies_runs_npm_install(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(launch, "FRONTEND_DIR", tmp_path)
    calls: list[tuple[str, list[str], Path]] = []

    def fake_run_streamed(label: str, command: list[str], cwd: Path, env) -> int:
        calls.append((label, command, cwd))
        return 0

    monkeypatch.setattr(launch, "run_streamed", fake_run_streamed)

    assert launch.ensure_frontend_dependencies("npm", _theme()) == 0
    assert calls == [("npm install", ["npm", "install"], tmp_path)]


def test_ensure_frontend_dependencies_propagates_failure(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(launch, "FRONTEND_DIR", tmp_path)
    monkeypatch.setattr(
        launch, "run_streamed", lambda label, command, cwd, env: 7
    )

    assert launch.ensure_frontend_dependencies("npm", _theme()) == 7
