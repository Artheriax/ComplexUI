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

logger = logging.getLogger("complexui.deps")

__all__ = [
    "DependencyInstallError",
    "INSTALL_TIMEOUT_SECONDS",
    "build_install_command",
    "install_dependencies",
    "missing_dependencies",
]

_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_PYPI_INDEX = "https://pypi.org/simple"
_TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}
_POLL_INTERVAL_SECONDS = 0.2
INSTALL_TIMEOUT_SECONDS = 600.0


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


def install_dependencies(
    requirements: Sequence[str],
    *,
    timeout: float = INSTALL_TIMEOUT_SECONDS,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Install missing requirements with ``uv pip install``.

    The install process is polled so ``should_stop`` can terminate it and
    ``timeout`` caps the whole operation. Spawn failures, cancellations,
    timeouts and non-zero exits raise :class:`DependencyInstallError`.
    """
    if not requirements:
        return
    environment = os.environ.copy()
    environment.setdefault("VIRTUAL_ENV", sys.prefix)
    command = build_install_command(requirements)
    logger.info("installing dependencies: %s", ", ".join(requirements))
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
            raise DependencyInstallError(
                f"could not start `uv pip install`: {exc}"
            ) from exc
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
                    f"`uv pip install {' '.join(requirements)}` timed out "
                    f"after {timeout:.0f}s"
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
