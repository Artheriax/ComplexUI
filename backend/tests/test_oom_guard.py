"""OOM detection, cleanup, retry and telemetry contracts."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from app import executor
from app.executor import (
    NodeOOMError,
    clear_cache,
    execute_graph,
)
from app.schemas import EventType, GraphNode, WorkflowGraph
from app.security import SecurityRegistry, code_hash


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _graph(code: str) -> tuple[WorkflowGraph, SecurityRegistry]:
    node = GraphNode(id="oom", code=code)
    registry = SecurityRegistry()
    registry.approve(node.id, code_hash(node.code))
    return WorkflowGraph(nodes=[node]), registry


def test_is_oom_error_recognizes_unified_and_legacy_messages() -> None:
    oom_type = torch.OutOfMemoryError
    assert executor._is_oom_error(oom_type("out of memory"))
    assert executor._is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert executor._is_oom_error(RuntimeError("DefaultCPUAllocator: can't allocate memory"))
    assert executor._is_oom_error(MemoryError("out of memory"))
    assert not executor._is_oom_error(ValueError("bad input"))


def test_oom_cleanup_clears_intermediates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "_CACHE", OrderedDict({"cached": object()}))
    monkeypatch.setattr(executor, "_CACHE_BYTES", 10)
    monkeypatch.setattr(executor, "_memory_snapshot", lambda: {})

    retried, _snapshot = executor._oom_cleanup()

    assert retried is True
    assert executor._CACHE == OrderedDict()
    assert executor._CACHE_BYTES == 0


def test_oom_retries_once_when_cleanup_reclaims_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = """
import torch
attempts = globals().get("__oom_attempts", 0) + 1
globals()["__oom_attempts"] = attempts
if attempts == 1:
    raise RuntimeError("CUDA out of memory")
return attempts: int
"""
    graph, registry = _graph(code)
    events = []
    monkeypatch.setattr(
        executor,
        "_oom_cleanup",
        lambda: (True, {"cuda_free_bytes": 1024}),
    )
    monkeypatch.setattr(
        executor,
        "_collect_node_metrics",
        lambda: {"cuda_peak_allocated_bytes": 42},
    )

    result = execute_graph(graph, security=registry, emit=events.append)

    assert result.outputs_by_name["oom"]["attempts"] == 2
    completed = [
        event for event in events if event.type is EventType.NODE_COMPLETED
    ]
    assert completed[0].data["metrics"] == {"cuda_peak_allocated_bytes": 42}
    assert not any(event.type is EventType.NODE_ERROR for event in events)


def test_oom_fails_with_actionable_error_when_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = """
raise RuntimeError("CUDA out of memory")
"""
    graph, registry = _graph(code)
    events = []
    monkeypatch.setattr(
        executor,
        "_oom_cleanup",
        lambda: (False, {"cuda_allocated_bytes": 10, "cuda_reserved_bytes": 20}),
    )

    with pytest.raises(NodeOOMError) as caught:
        execute_graph(graph, security=registry, emit=events.append)

    assert caught.value.error_type == "NodeOOM"
    assert "Suggestions:" in caught.value.message
    error = next(event for event in events if event.type is EventType.NODE_ERROR)
    assert error.data["error_type"] == "NodeOOM"
    assert "reserved=" in error.data["message"]


def test_metrics_are_attached_to_completed_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    code = """
result = 7
return result: int
"""
    graph, registry = _graph(code)
    events = []
    monkeypatch.setattr(
        executor,
        "_collect_node_metrics",
        lambda: {"cpu_rss_bytes": 123, "cuda_peak_reserved_bytes": 456},
    )

    execute_graph(graph, security=registry, emit=events.append)

    completed = next(
        event for event in events if event.type is EventType.NODE_COMPLETED
    )
    assert completed.data["metrics"]["cpu_rss_bytes"] == 123
    assert completed.data["metrics"]["cuda_peak_reserved_bytes"] == 456
