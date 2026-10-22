"""PEP 723 dependency management for node code."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence

from packaging.requirements import InvalidRequirement, Requirement

from .accelerator import (
    PYPI_INDEX,
    TORCH_CPU_INDEX,
    configured_accelerator,
    effective_accelerator,
    torch_index,
)

logger = logging.getLogger("complexui.deps")

__all__ = [
    "DependencyInstallError",
    "INSTALL_TIMEOUT_ENV",
    "INSTALL_TIMEOUT_SECONDS",
    "build_install_command",
    "install_dependencies",
    "missing_dependencies",
]

_PYPI_INDEX = PYPI_INDEX
_TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}
_POLL_INTERVAL_SECONDS = 0.2
INSTALL_TIMEOUT_ENV = "COMPLEXUI_INSTALL_TIMEOUT_SECONDS"
INSTALL_TIMEOUT_SECONDS = 1200.0


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
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
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


def install_timeout_seconds() -> float:
    """Return the configured dependency-install timeout."""
    raw = os.environ.get(INSTALL_TIMEOUT_ENV, "").strip()
    if not raw:
        return INSTALL_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return INSTALL_TIMEOUT_SECONDS
    return value if value >= 0 else INSTALL_TIMEOUT_SECONDS


def build_install_command(
    requirements: Sequence[str],
    *,
    platform: str | None = None,
    accelerator: str | None = None,
    reinstall: bool | None = None,
) -> list[str]:
    """Build ``uv pip install`` args with accelerator-aware torch routing.

    Torch-family packages use the CPU index by default, or the CUDA index
    selected by ``COMPLEXUI_ACCELERATOR``/``COMPLEXUI_TORCH_CUDA_INDEX``. When
    the installed torch build does not match the selected accelerator, the
    relevant packages are reinstalled so the GPU is actually usable.
    """
    target = sys.platform if platform is None else platform
    command = ["uv", "pip", "install"]
    torch_requirements = {
        _requirement_name(raw) for raw in requirements
    } & _TORCH_PACKAGES

    if target.startswith(("win32", "linux")) and torch_requirements:
        selected_index = torch_index(accelerator, platform=target)
        command.extend(["--index", selected_index, "--index", _PYPI_INDEX])
        installed = _installed_version("torch")
        if installed is not None and reinstall is not False:
            installed_cuda = "+cu" in installed.lower()
            wants_cuda = selected_index != TORCH_CPU_INDEX
            if reinstall is True or installed_cuda != wants_cuda:
                for package in sorted(torch_requirements):
                    command.extend(["--reinstall-package", package])
    command.extend(requirements)
    return command


def _validated_requirements(requirements: Sequence[str]) -> list[str]:
    validated: list[str] = []
    for raw in requirements:
        if raw.strip().startswith("-"):
            raise DependencyInstallError(f"Invalid dependency requirement: {raw!r}")
        try:
            Requirement(raw)
        except InvalidRequirement as exc:
            raise DependencyInstallError(f"Invalid dependency requirement: {raw!r}") from exc
        validated.append(raw)
    return validated


def install_dependencies(
    requirements: Sequence[str],
    *,
    timeout: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Install missing requirements with ``uv pip install``.

    The install process is polled so ``should_stop`` can terminate it and
    ``timeout`` caps the whole operation. Spawn failures, cancellations,
    timeouts and non-zero exits raise :class:`DependencyInstallError`.
    """
    if not requirements:
        return
    requirements = _validated_requirements(requirements)
    if timeout is None:
        timeout = install_timeout_seconds()
    environment = os.environ.copy()
    environment.setdefault("VIRTUAL_ENV", sys.prefix)
    command = build_install_command(requirements)
    logger.info(
        "installing dependencies for accelerator '%s' (effective '%s'): %s",
        configured_accelerator(),
        effective_accelerator(),
        ", ".join(requirements),
    )
    logger.debug("install command: %s", " ".join(command))
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+b") as output:
        try:
            process = subprocess.Popen(  # noqa: S603 - command is parsed deps
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        except OSError as exc:
            raise DependencyInstallError(f"could not start `uv pip install`: {exc}") from exc
        deadline = time.monotonic() + timeout
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            if should_stop is not None and should_stop():
                _terminate(process)
                logger.warning("dependency install cancelled")
                raise DependencyInstallError(
                    f"`uv pip install {' '.join(requirements)}` was cancelled"
                )
            if time.monotonic() >= deadline:
                _terminate(process)
                logger.error("dependency install timed out after %.0fs", timeout)
                raise DependencyInstallError(
                    f"`uv pip install {' '.join(requirements)}` timed out after {timeout:.0f}s"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
        output.seek(0)
        detail = output.read().decode("utf-8", "replace").strip()
    if returncode != 0:
        logger.error(
            "dependency install failed after %.1fs: %s",
            time.monotonic() - started,
            detail[:400],
        )
        raise DependencyInstallError(
            f"`uv pip install {' '.join(requirements)}` failed: "
            f"{detail or f'exit code {returncode}'}"
        )
    logger.info(
        "dependencies installed in %.1fs: %s",
        time.monotonic() - started,
        ", ".join(requirements),
    )
    importlib.invalidate_caches()


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        pass
    process.wait()
