"""Tests for the launcher helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_non_loopback_host_requires_allow_network() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args(["backend", "--host", "0.0.0.0"])


def test_loopback_host_is_allowed_without_flag() -> None:
    args = launch.parse_args(["backend", "--host", "localhost"])

    assert args.allow_network is False


def test_allow_network_enables_wildcard_hosts(monkeypatch) -> None:
    monkeypatch.delenv("COMPLEXUI_ALLOWED_HOSTS", raising=False)
    args = launch.parse_args(["backend", "--host", "0.0.0.0", "--allow-network"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_ALLOWED_HOSTS"] == "*"


def test_allow_network_keeps_loopback_guard(monkeypatch) -> None:
    monkeypatch.delenv("COMPLEXUI_ALLOWED_HOSTS", raising=False)
    args = launch.parse_args(["backend", "--host", "localhost", "--allow-network"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert "COMPLEXUI_ALLOWED_HOSTS" not in environment


def test_allow_network_preserves_explicit_hosts(monkeypatch) -> None:
    monkeypatch.setenv("COMPLEXUI_ALLOWED_HOSTS", "ui.local")
    args = launch.parse_args(["backend", "--host", "0.0.0.0", "--allow-network"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_ALLOWED_HOSTS"] == "ui.local"


def test_backend_environment_sets_log_format() -> None:
    args = launch.parse_args(["backend", "--log-format", "json"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_LOG_FORMAT"] == "json"


def test_accelerator_defaults_to_environment(monkeypatch) -> None:
    monkeypatch.setenv("COMPLEXUI_ACCELERATOR", "cuda")

    args = launch.parse_args(["backend"])

    assert args.accelerator == "cuda"


def test_backend_environment_sets_accelerator() -> None:
    args = launch.parse_args(["backend", "--accelerator", "cpu"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_ACCELERATOR"] == "cpu"


def test_unknown_accelerator_is_rejected() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args(["backend", "--accelerator", "gpu"])


def test_log_format_defaults_to_environment(monkeypatch) -> None:
    monkeypatch.setenv("COMPLEXUI_LOG_FORMAT", "json")

    args = launch.parse_args(["backend"])

    assert args.log_format == "json"


def test_backend_environment_sets_log_dir(tmp_path: Path) -> None:
    args = launch.parse_args(["backend", "--log-dir", str(tmp_path)])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_LOG_DIR"] == str(tmp_path)


def test_backend_environment_can_disable_file_logs() -> None:
    args = launch.parse_args(["backend", "--no-log-file"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_LOG_DIR"] == "0"


def test_unknown_log_format_is_rejected() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args(["backend", "--log-format", "xml"])


def test_resolve_color_respects_disable_environment(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert launch._resolve_color(False) is False

    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "0")
    assert launch._resolve_color(False) is False


def test_frontend_binds_network_host() -> None:
    args = launch.parse_args(
        ["dev", "--host", "192.168.1.5", "--allow-network"]
    )

    command = launch.build_frontend_command("npm", args)

    assert command[-2:] == ["--host", "192.168.1.5"]


def test_frontend_keeps_loopback_binding_by_default() -> None:
    args = launch.parse_args(["dev"])

    command = launch.build_frontend_command("npm", args)

    assert "--host" not in command
