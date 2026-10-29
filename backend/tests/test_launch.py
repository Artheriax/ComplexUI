"""Tests for the launcher helpers."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import launch


def _theme() -> launch.Theme:
    return launch.Theme(False)


def test_console_renderer_keeps_progress_at_bottom() -> None:
    stream = io.StringIO()
    renderer = launch.ConsoleRenderer(stream, enabled=True, ansi=False)

    renderer.update_progress("progress 1")
    renderer.write("[backend] log line\n")
    renderer.update_progress("progress 2")
    renderer.clear_progress()

    output = stream.getvalue()
    assert "progress 1" in output
    assert "[backend] log line" in output
    assert "progress 2" in output
    assert output.endswith("\r")


def test_console_renderer_falls_back_to_plain_lines_without_tty() -> None:
    stream = io.StringIO()
    renderer = launch.ConsoleRenderer(stream, enabled=False)

    renderer.update_progress("progress 1")
    renderer.write("plain line\n")

    assert stream.getvalue() == "progress 1\nplain line\n"


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


def test_ensure_frontend_dependencies_uses_npm_ci_for_fresh_clone(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launch, "FRONTEND_DIR", tmp_path)
    calls: list[list[str]] = []

    def fake_run_streamed(label: str, command: list[str], cwd: Path, env) -> int:
        calls.append(command)
        (cwd / "node_modules").mkdir(exist_ok=True)
        return 0

    monkeypatch.setattr(launch, "run_streamed", fake_run_streamed)

    assert launch.ensure_frontend_dependencies("npm", _theme()) == 0
    assert calls == [["npm", "ci"]]
    assert launch._installed_digest() == launch._lockfile_digest()


def test_ensure_frontend_dependencies_refreshes_changed_lockfile(
    tmp_path: Path, monkeypatch
) -> None:
    lockfile = tmp_path / "package-lock.json"
    lockfile.write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    monkeypatch.setattr(launch, "FRONTEND_DIR", tmp_path)
    calls: list[list[str]] = []

    def fake_run_streamed(label: str, command: list[str], cwd: Path, env) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(launch, "run_streamed", fake_run_streamed)

    assert launch.ensure_frontend_dependencies("npm", _theme()) == 0
    assert calls == [["npm", "install"]]

    lockfile.write_text('{"lockfileVersion": 3, "changed": true}\n', encoding="utf-8")
    assert launch.ensure_frontend_dependencies("npm", _theme()) == 0
    assert calls == [["npm", "install"], ["npm", "install"]]

    assert launch.ensure_frontend_dependencies("npm", _theme()) == 0
    assert len(calls) == 2


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


def test_backend_command_skips_optional_dev_toolchain() -> None:
    args = launch.parse_args(["backend"])

    command = launch.build_backend_command(args, reload=False)

    assert command[:4] == ["uv", "run", "--no-dev", "python"]


def test_custom_ports_propagate_to_vite_and_cors(monkeypatch) -> None:
    monkeypatch.delenv("COMPLEXUI_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("COMPLEXUI_BACKEND_URL", raising=False)
    args = launch.parse_args(
        ["dev", "--backend-port", "8010", "--frontend-port", "5200"]
    )

    backend_environment = launch.backend_environment(args, serve_dist=False)
    frontend_environment = launch.frontend_environment(args)

    assert backend_environment["COMPLEXUI_CORS_ORIGINS"] == (
        "http://localhost:5200,http://127.0.0.1:5200"
    )
    assert frontend_environment["COMPLEXUI_BACKEND_URL"] == (
        "http://127.0.0.1:8010"
    )


def test_explicit_cors_origins_are_preserved(monkeypatch) -> None:
    monkeypatch.setenv("COMPLEXUI_CORS_ORIGINS", "http://ui.local:5200")
    args = launch.parse_args(["dev", "--frontend-port", "5200"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_CORS_ORIGINS"] == "http://ui.local:5200"


def test_frontend_only_mode_keeps_explicit_backend_url(monkeypatch) -> None:
    monkeypatch.setenv("COMPLEXUI_BACKEND_URL", "http://gpu-box.local:9000")
    args = launch.parse_args(["frontend"])

    environment = launch.frontend_environment(args)

    assert environment["COMPLEXUI_BACKEND_URL"] == "http://gpu-box.local:9000"


def test_dev_mode_overrides_backend_url_with_flags(monkeypatch) -> None:
    monkeypatch.setenv("COMPLEXUI_BACKEND_URL", "http://stale.local:9000")
    args = launch.parse_args(["dev", "--backend-port", "8010"])

    environment = launch.frontend_environment(args)

    assert environment["COMPLEXUI_BACKEND_URL"] == "http://127.0.0.1:8010"


def test_node_version_support_matches_vite_engines() -> None:
    assert launch.node_version_supported((20, 19, 0)) is True
    assert launch.node_version_supported((20, 18, 0)) is False
    assert launch.node_version_supported((21, 7, 0)) is False
    assert launch.node_version_supported((22, 12, 0)) is True
    assert launch.node_version_supported((24, 1, 0)) is True
    assert launch.parse_node_version("v22.12.0") == (22, 12, 0)
    assert launch.parse_node_version("no version") is None


def test_performance_flags_propagate_to_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    cache_dir = tmp_path / "inductor"
    args = launch.parse_args(
        [
            "backend",
            "--attention-backend",
            "sdpa-math",
            "--memory-profile",
            "low",
            "--torch-compile",
            "default",
            "--compile-cache-dir",
            str(cache_dir),
            "--experimental-fp16-accumulation",
        ]
    )

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_ATTENTION"] == "sdpa-math"
    assert environment["COMPLEXUI_MEMORY_PROFILE"] == "low"
    assert environment["COMPLEXUI_TORCH_COMPILE"] == "default"
    assert environment["COMPLEXUI_COMPILE_CACHE_DIR"] == str(cache_dir)
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == str(cache_dir)
    assert environment["TRITON_CACHE_DIR"] == str(cache_dir / "triton")
    assert environment["COMPLEXUI_EXPERIMENTAL_FP16_ACCUM"] == "1"


def test_attention_benchmark_flag_propagates(monkeypatch) -> None:
    monkeypatch.delenv("COMPLEXUI_ATTENTION_BENCHMARK", raising=False)
    args = launch.parse_args(["backend", "--attention-benchmark"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_ATTENTION_BENCHMARK"] == "1"


def test_allocator_profiles_and_conflicts(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    args = launch.parse_args(
        ["backend", "--allocator", "cudaMallocAsync", "--no-expandable-segments"]
    )

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["COMPLEXUI_CUDA_ALLOCATOR"] == "cudaMallocAsync"
    assert environment["PYTORCH_ALLOC_CONF"] == "backend:cudaMallocAsync"

    with pytest.raises(SystemExit):
        launch.parse_args(
            [
                "backend",
                "--allocator",
                "cudaMallocAsync",
                "--expandable-segments",
            ]
        )


def test_expandable_segments_flag(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    args = launch.parse_args(["backend", "--expandable-segments"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_existing_allocator_environment_wins(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
    args = launch.parse_args(["backend", "--expandable-segments"])

    environment = launch.backend_environment(args, serve_dist=False)

    assert environment["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:128"
