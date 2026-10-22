"""Tests for the PEP 723 dependency manager (accelerator-aware torch routing)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.env_manager import (
    DependencyInstallError,
    build_install_command,
    install_dependencies,
    install_timeout_seconds,
)

_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
_PYPI_INDEX = "https://pypi.org/simple"


class _FakeProcess:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self) -> int:
        self.waited = True
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


def _fake_popen(
    process: _FakeProcess,
    *,
    captured: list[list[str]] | None = None,
    output: bytes = b"",
) -> Callable[..., _FakeProcess]:
    def popen(command: list[str], stdout: Any, **_kwargs: object) -> _FakeProcess:
        if captured is not None:
            captured.append(list(command))
        if output:
            stdout.write(output)
            stdout.flush()
        return process

    return popen


def test_plain_requirements_use_environment_indices() -> None:
    command = build_install_command(["scikit-learn>=1.5", "numpy>=1.26"])

    assert command == [
        "uv",
        "pip",
        "install",
        "scikit-learn>=1.5",
        "numpy>=1.26",
    ]


def test_torch_requirements_route_to_cpu_index() -> None:
    command = build_install_command(
        ["torch>=2.2"],
        accelerator="cpu",
        reinstall=False,
    )

    assert command[:5] == ["uv", "pip", "install", "--index", _CPU_INDEX]
    assert command[5:7] == ["--index", _PYPI_INDEX]
    assert command[-1] == "torch>=2.2"


def test_torch_requirements_route_to_cuda_index() -> None:
    command = build_install_command(
        ["torch>=2.2"],
        accelerator="cuda",
        reinstall=False,
    )

    assert command[:5] == ["uv", "pip", "install", "--index", _CUDA_INDEX]
    assert command[5:7] == ["--index", _PYPI_INDEX]
    assert command[-1] == "torch>=2.2"


def test_cuda_index_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPLEXUI_TORCH_CUDA_INDEX", "https://example.invalid/cu999")

    command = build_install_command(
        ["torch>=2.2"],
        accelerator="cuda",
        reinstall=False,
    )

    assert "https://example.invalid/cu999" in command


@pytest.mark.parametrize(("available", "expected"), [(True, _CUDA_INDEX), (False, _CPU_INDEX)])
def test_auto_accelerator_uses_nvidia_detection(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr("app.accelerator.nvidia_available", lambda: available)

    command = build_install_command(
        ["torch>=2.2"],
        accelerator="auto",
        reinstall=False,
    )

    assert expected in command


def test_cuda_install_reinstalls_cpu_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.env_manager._installed_version", lambda _name: "2.14.1+cpu")

    command = build_install_command(["torch>=2.2"], accelerator="cuda")

    assert command.count("--reinstall-package") == 1
    assert command[command.index("--reinstall-package") + 1] == "torch"


def test_cuda_install_keeps_matching_cuda_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.env_manager._installed_version", lambda _name: "2.14.1+cu130")

    command = build_install_command(["torch>=2.2"], accelerator="cuda")

    assert "--reinstall-package" not in command


def test_cpu_install_reinstalls_cuda_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.env_manager._installed_version", lambda _name: "2.14.1+cu130")

    command = build_install_command(["torch>=2.2"], accelerator="cpu")

    assert "--reinstall-package" in command


def test_torch_family_names_are_detected_case_insensitively() -> None:
    for raw in ("Torch", "torchvision==0.19.0", "torchaudio >= 2", "torch"):
        command = build_install_command(
            [raw],
            accelerator="cpu",
            reinstall=False,
        )
        assert "--index" in command, raw


def test_non_torch_family_names_are_not_routed() -> None:
    for raw in ("torch-stub", "torchvision-tools>=1.0", "not-torch==1.0"):
        command = build_install_command([raw])
        assert "--index" not in command, raw


def test_invalid_torch_requirement_still_routed() -> None:
    command = build_install_command(
        ["torch>="],
        accelerator="cpu",
        reinstall=False,
    )

    assert "--index" in command
    assert command[-1] == "torch>="


def test_darwin_platform_is_not_routed() -> None:
    command = build_install_command(
        ["torch>=2.2"],
        platform="darwin",
        accelerator="cuda",
        reinstall=False,
    )

    assert command == ["uv", "pip", "install", "torch>=2.2"]


def test_linux_platform_is_routed() -> None:
    command = build_install_command(
        ["torch>=2.2"],
        platform="linux",
        accelerator="cpu",
        reinstall=False,
    )

    assert command[3:7] == ["--index", _CPU_INDEX, "--index", _PYPI_INDEX]


def test_install_timeout_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPLEXUI_INSTALL_TIMEOUT_SECONDS", "42.5")
    assert install_timeout_seconds() == 42.5

    monkeypatch.setenv("COMPLEXUI_INSTALL_TIMEOUT_SECONDS", "nope")
    assert install_timeout_seconds() == 1200.0


def test_install_dependencies_uses_cpu_index_for_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setenv("COMPLEXUI_ACCELERATOR", "cpu")
    monkeypatch.setattr(
        "app.env_manager.subprocess.Popen",
        _fake_popen(_FakeProcess(0), captured=captured),
    )

    install_dependencies(["torch>=2.2"])

    assert len(captured) == 1
    assert captured[0][:5] == ["uv", "pip", "install", "--index", _CPU_INDEX]
    assert captured[0][-1] == "torch>=2.2"


def test_install_dependencies_rejects_option_like_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("subprocess.Popen must not be called")

    monkeypatch.setattr("app.env_manager.subprocess.Popen", fake_popen)

    with pytest.raises(DependencyInstallError, match="Invalid dependency"):
        install_dependencies(["--index-url=https://example.invalid"])

    with pytest.raises(DependencyInstallError, match="Invalid dependency"):
        install_dependencies(["not a requirement !!!"])


def test_install_dependencies_noop_for_empty_requirements(monkeypatch) -> None:
    def fake_popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("subprocess.Popen must not be called")

    monkeypatch.setattr("app.env_manager.subprocess.Popen", fake_popen)

    install_dependencies([])


def test_install_dependencies_reports_failure_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.env_manager.subprocess.Popen",
        _fake_popen(_FakeProcess(1), output=b"resolution failed\n"),
    )

    with pytest.raises(DependencyInstallError, match="resolution failed"):
        install_dependencies(["tiny-package>=0.1"])


def test_install_dependencies_times_out_and_kills_process(monkeypatch) -> None:
    process = _FakeProcess(None)
    monkeypatch.setattr("app.env_manager.subprocess.Popen", _fake_popen(process))

    with pytest.raises(DependencyInstallError, match="timed out"):
        install_dependencies(["tiny-package>=0.1"], timeout=0)

    assert process.killed is True
    assert process.waited is True


def test_install_dependencies_cancels_and_kills_process(monkeypatch) -> None:
    process = _FakeProcess(None)
    monkeypatch.setattr("app.env_manager.subprocess.Popen", _fake_popen(process))

    with pytest.raises(DependencyInstallError, match="cancelled"):
        install_dependencies(["tiny-package>=0.1"], should_stop=lambda: True)

    assert process.killed is True
    assert process.waited is True


def test_install_dependencies_wraps_spawn_failure(monkeypatch) -> None:
    def fake_popen(*_args: object, **_kwargs: object) -> object:
        raise OSError("uv is missing")

    monkeypatch.setattr("app.env_manager.subprocess.Popen", fake_popen)

    with pytest.raises(DependencyInstallError, match="could not start"):
        install_dependencies(["tiny-package>=0.1"])
