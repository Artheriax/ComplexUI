"""PEP 723 dependency management for node code."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import subprocess
import sys
from collections.abc import Sequence

from packaging.requirements import InvalidRequirement, Requirement

logger = logging.getLogger("complexui.deps")

__all__ = [
    "DependencyInstallError",
    "build_install_command",
    "install_dependencies",
    "missing_dependencies",
]

_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_PYPI_INDEX = "https://pypi.org/simple"
_TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}


class DependencyInstallError(Exception):
    """Raised when a node's dependencies cannot be verified or installed."""


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def missing_dependencies(requirements: Sequence[str]) -> list[str]:
    """Return the requirements that are not satisfied by the environment."""
    missing: list[str] = []
    for raw in requirements:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            missing.append(raw)
            continue
        version = _installed_version(requirement.name)
        if version is None:
            missing.append(raw)
            continue
        if requirement.specifier and not requirement.specifier.contains(
            version, prereleases=True
        ):
            missing.append(raw)
    return missing


def _requirement_name(raw: str) -> str:
    try:
        return Requirement(raw).name.lower()
    except InvalidRequirement:
        name = raw.strip().lower()
        for separator in ("<", ">", "=", "!", "~", ";", "[", " ", "@"):
            name = name.split(separator, 1)[0]
        return name


def build_install_command(
    requirements: Sequence[str], *, platform: str | None = None
) -> list[str]:
    """Build ``uv pip install`` args, routing torch to the CPU wheel index.

    On Windows/Linux the default PyPI torch wheels bundle multi-GB CUDA
    runtimes; the PyTorch CPU index serves a lean CPU-only build. uv's
    first-index strategy resolves torch from the CPU index and everything else
    (sympy, filelock, ...) from PyPI.
    """
    target = sys.platform if platform is None else platform
    command = ["uv", "pip", "install"]
    if target.startswith(("win32", "linux")) and any(
        _requirement_name(raw) in _TORCH_PACKAGES for raw in requirements
    ):
        command.extend(["--index", _TORCH_CPU_INDEX, "--index", _PYPI_INDEX])
    command.extend(requirements)
    return command


def install_dependencies(requirements: Sequence[str]) -> None:
    """Install missing requirements with ``uv pip install``."""
    if not requirements:
        return
    environment = os.environ.copy()
    environment.setdefault("VIRTUAL_ENV", sys.prefix)
    command = build_install_command(requirements)
    logger.info("installing dependencies: %s", ", ".join(requirements))
    logger.debug("install command: %s", " ".join(command))
    try:
        subprocess.run(  # noqa: S603 - command is built from parsed requirements
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"exit code {exc.returncode}"
        logger.error("dependency install failed: %s", detail[:400])
        raise DependencyInstallError(
            f"`uv pip install {' '.join(requirements)}` failed: {detail}"
        ) from exc
    importlib.invalidate_caches()
