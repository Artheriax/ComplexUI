"""Tests for the PEP 723 dependency manager (CPU torch index routing)."""

from __future__ import annotations

from app.env_manager import build_install_command, install_dependencies

_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_PYPI_INDEX = "https://pypi.org/simple"


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
    command = build_install_command(["torch>=2.2"])

    assert command[:5] == ["uv", "pip", "install", "--index", _CPU_INDEX]
    assert command[5:7] == ["--index", _PYPI_INDEX]
    assert command[-1] == "torch>=2.2"


def test_torch_family_names_are_detected_case_insensitively() -> None:
    for raw in ("Torch", "torchvision==0.19.0", "torchaudio >= 2", "torch"):
        command = build_install_command([raw])
        assert "--index" in command, raw


def test_non_torch_family_names_are_not_routed() -> None:
    for raw in ("torch-stub", "torchvision-tools>=1.0", "not-torch==1.0"):
        command = build_install_command([raw])
        assert "--index" not in command, raw


def test_invalid_torch_requirement_still_routed() -> None:
    command = build_install_command(["torch>="])

    assert "--index" in command
    assert command[-1] == "torch>="


def test_darwin_platform_is_not_routed() -> None:
    command = build_install_command(["torch>=2.2"], platform="darwin")

    assert command == ["uv", "pip", "install", "torch>=2.2"]


def test_linux_platform_is_routed() -> None:
    command = build_install_command(["torch>=2.2"], platform="linux")

    assert command[3:7] == ["--index", _CPU_INDEX, "--index", _PYPI_INDEX]


def test_install_dependencies_uses_cpu_index_for_torch(monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        captured.append(list(command))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("app.env_manager.subprocess.run", fake_run)

    install_dependencies(["torch>=2.2"])

    assert len(captured) == 1
    assert captured[0][:5] == ["uv", "pip", "install", "--index", _CPU_INDEX]
    assert captured[0][-1] == "torch>=2.2"


def test_install_dependencies_noop_for_empty_requirements(monkeypatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("subprocess.run must not be called")

    monkeypatch.setattr("app.env_manager.subprocess.run", fake_run)

    install_dependencies([])
