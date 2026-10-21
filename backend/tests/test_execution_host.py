"""Execution process limits and memory probing tests."""

from __future__ import annotations

import os

import pytest

from app.execution_host import (
    DEFAULT_MAX_MEMORY_MB,
    DEFAULT_NODE_TIMEOUT_SECONDS,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    ENV_MAX_MEMORY_MB,
    ENV_NODE_TIMEOUT,
    ENV_RUN_TIMEOUT,
    execution_limits,
    process_memory_bytes,
)


def test_execution_limits_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_NODE_TIMEOUT, raising=False)
    monkeypatch.delenv(ENV_RUN_TIMEOUT, raising=False)
    monkeypatch.delenv(ENV_MAX_MEMORY_MB, raising=False)

    defaults = execution_limits()
    assert defaults.node_timeout_seconds == DEFAULT_NODE_TIMEOUT_SECONDS
    assert defaults.run_timeout_seconds == DEFAULT_RUN_TIMEOUT_SECONDS
    assert defaults.max_memory_bytes == DEFAULT_MAX_MEMORY_MB * 1024 * 1024

    monkeypatch.setenv(ENV_NODE_TIMEOUT, "12.5")
    monkeypatch.setenv(ENV_RUN_TIMEOUT, "0")
    monkeypatch.setenv(ENV_MAX_MEMORY_MB, "256")

    configured = execution_limits()
    assert configured.node_timeout_seconds == 12.5
    assert configured.run_timeout_seconds == 0
    assert configured.max_memory_bytes == 256 * 1024 * 1024


def test_execution_limits_ignore_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_NODE_TIMEOUT, "not-a-number")
    monkeypatch.setenv(ENV_RUN_TIMEOUT, "-4")
    monkeypatch.setenv(ENV_MAX_MEMORY_MB, "many")

    limits = execution_limits()
    assert limits.node_timeout_seconds == DEFAULT_NODE_TIMEOUT_SECONDS
    assert limits.run_timeout_seconds == DEFAULT_RUN_TIMEOUT_SECONDS
    assert limits.max_memory_bytes == DEFAULT_MAX_MEMORY_MB * 1024 * 1024


def test_process_memory_bytes_reads_the_current_process() -> None:
    used = process_memory_bytes(os.getpid())
    assert used is not None
    assert used > 0
