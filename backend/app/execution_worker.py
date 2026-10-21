"""Isolated execution worker process.

The FastAPI parent process owns the HTTP/WebSocket surface and run history. Each
run executes in a disposable child process so runaway nodes can be terminated
without taking down the server, while the content-addressable cache is handed
across the process boundary before and after each run.
"""

from __future__ import annotations

import os
import sys
import traceback
from multiprocessing.connection import Connection
from typing import Any

from .executor import (
    ExecutionCancelled,
    GraphPlan,
    NodeExecutionError,
    cache_snapshot,
    execute_graph,
    restore_cache,
    transferable_cache,
)
from .schemas import ExecutionEvent, WorkflowGraph
from .security import SecurityRegistry, UntrustedNodeError


def failure_payload(exc: BaseException) -> dict[str, Any]:
    """Serialize execution failures into the terminal event payload shape."""
    if isinstance(exc, NodeExecutionError):
        return {
            "error_type": exc.error_type,
            "message": exc.message,
            "node_id": exc.node_id,
            "line": exc.line,
            "traceback": exc.traceback_text,
        }
    if isinstance(exc, UntrustedNodeError):
        return {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "node_id": exc.node_id,
            "line": None,
            "traceback": "",
        }
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "node_id": None,
        "line": None,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _apply_memory_limit(limit_bytes: int) -> None:
    if limit_bytes <= 0 or sys.platform == "win32":
        return
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (OSError, ValueError):
        pass


def run_worker(
    graph: WorkflowGraph,
    plan: GraphPlan,
    approvals: list[tuple[str, str]],
    cache: dict[str, Any],
    stop_event: Any,
    connection: Connection,
    memory_limit_bytes: int,
) -> None:
    """Execute one graph in a child process and stream messages to the parent."""
    _apply_memory_limit(memory_limit_bytes)
    restore_cache(cache)
    registry = SecurityRegistry()
    for node_id, hash_value in approvals:
        registry.approve(node_id, hash_value)

    def emit(event: ExecutionEvent) -> None:
        connection.send(
            {
                "kind": "event",
                "type": event.type.value,
                "node_id": event.node_id,
                "data": event.data,
            }
        )

    try:
        result = execute_graph(
            graph,
            security=registry,
            plan=plan,
            emit=emit,
            should_stop=stop_event.is_set,
        )
        connection.send(
            {
                "kind": "completed",
                "cached": result.cached,
                "previews": result.previews,
                "cache": transferable_cache(cache_snapshot()),
            }
        )
    except ExecutionCancelled:
        connection.send(
            {
                "kind": "cancelled",
                "cache": transferable_cache(cache_snapshot()),
            }
        )
    except BaseException as exc:
        connection.send(
            {
                "kind": "failed",
                "failure": failure_payload(exc),
                "cache": transferable_cache(cache_snapshot()),
            }
        )
    finally:
        try:
            connection.close()
        except OSError:
            pass
        os._exit(0)


__all__ = ["failure_payload", "run_worker"]
