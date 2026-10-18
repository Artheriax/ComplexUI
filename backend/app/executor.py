"""Topological DAG executor with content-addressable caching.

Executes a :class:`~app.schemas.WorkflowGraph` node by node in dependency
order, streams :class:`~app.schemas.ExecutionEvent` telemetry through an
optional sink, and caches each node's outputs keyed by

``sha256(node.code + json(widget_values) + "".join(upstream_hashes))``

so that re-running a graph only re-executes nodes whose code, widget values,
or transitive upstream state actually changed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .env_manager import (
    DependencyInstallError,
    install_dependencies,
    missing_dependencies,
)
from .schemas import (
    EventType,
    ExecutionEvent,
    GraphEdge,
    GraphNode,
    NodeSchema,
    SocketType,
    WorkflowGraph,
)
from .security import SecurityRegistry, UntrustedNodeError, code_hash, get_security_registry
from .transpiler import TranspileError, parse_node, transpile_node

__all__ = [
    "EventSink",
    "ExecutionCancelled",
    "ExecutionResult",
    "GraphCycleError",
    "GraphError",
    "GraphPlan",
    "GraphValidationError",
    "NodeExecutionError",
    "build_output_preview",
    "clear_cache",
    "execute_graph",
    "topological_sort",
    "validate_graph",
]

EventSink = Callable[[ExecutionEvent], None]

_CACHE: dict[str, tuple[str, tuple[Any, ...]]] = {}


def clear_cache() -> None:
    """Drop all cached node outputs."""
    _CACHE.clear()


class GraphError(Exception):
    """Base class for workflow graph problems."""


class GraphValidationError(GraphError):
    """Raised when the workflow graph is malformed (wiring, references)."""


class GraphCycleError(GraphError):
    """Raised when the workflow graph contains a cycle."""


class ExecutionCancelled(GraphError):
    """Raised when the caller requests cancellation between nodes."""


class NodeExecutionError(Exception):
    """Raised when a node's code fails at runtime."""

    def __init__(
        self,
        node_id: str,
        *,
        line: int | None,
        message: str,
        error_type: str,
        traceback_text: str,
    ) -> None:
        super().__init__(f"Node '{node_id}' failed: {message}")
        self.node_id = node_id
        self.line = line
        self.message = message
        self.error_type = error_type
        self.traceback_text = traceback_text


class _OutputContractError(Exception):
    """Raised when a node's return value does not match its output sockets."""


@dataclass(slots=True)
class ExecutionResult:
    outputs: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    outputs_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    previews: dict[str, dict[str, Any]] = field(default_factory=dict)
    cached: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class GraphPlan:
    order: list[str]
    schemas: dict[str, NodeSchema]
    untrusted: list[tuple[str, str]]


def validate_graph(
    graph: WorkflowGraph,
    *,
    security: SecurityRegistry | None = None,
    targets: Sequence[str] | None = None,
) -> GraphPlan:
    """Pre-flight a graph without executing it.

    Returns the topological order, per-node schemas, and the ``(node_id,
    code_hash)`` pairs of nodes whose code is neither an official template nor
    approved for this session. When ``targets`` is given, the plan is scoped to
    those nodes and their ancestors (run-to-node), including the trust check.
    """
    registry = security if security is not None else get_security_registry()
    nodes_by_id = _index_nodes(graph)
    order = _topological_order(nodes_by_id, graph.edges)
    allowed = _scope_to_targets(nodes_by_id, graph.edges, targets)
    if allowed is not None:
        order = [node_id for node_id in order if node_id in allowed]
    schemas = _plan(graph)
    untrusted = [
        (node.id, code_hash(node.code))
        for node in graph.nodes
        if (allowed is None or node.id in allowed)
        and not registry.is_trusted(node.id, node.code)
    ]
    return GraphPlan(order=order, schemas=schemas, untrusted=untrusted)


def _scope_to_targets(
    nodes_by_id: dict[str, GraphNode],
    edges: Sequence[GraphEdge],
    targets: Sequence[str] | None,
) -> set[str] | None:
    """Return the target nodes plus all their ancestors (None = whole graph)."""
    if targets is None:
        return None
    target_list = list(targets)
    if not target_list:
        raise GraphValidationError("targets must not be empty")
    unknown = [node_id for node_id in target_list if node_id not in nodes_by_id]
    if unknown:
        raise GraphValidationError(
            f"Unknown target node(s): {', '.join(sorted(unknown))}"
        )
    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}
    for edge in edges:
        incoming[edge.target_node_id].append(edge.source_node_id)
    allowed: set[str] = set(target_list)
    queue: deque[str] = deque(target_list)
    while queue:
        for source in incoming[queue.popleft()]:
            if source not in allowed:
                allowed.add(source)
                queue.append(source)
    return allowed


def execute_graph(
    graph: WorkflowGraph,
    *,
    security: SecurityRegistry | None = None,
    emit: EventSink | None = None,
    allow_installs: bool = True,
    should_stop: Callable[[], bool] | None = None,
    targets: Sequence[str] | None = None,
) -> ExecutionResult:
    """Execute a workflow graph and return per-node outputs and previews.

    ``should_stop`` is polled between nodes; when it returns ``True``,
    :class:`ExecutionCancelled` is raised before the next node starts (outputs
    of already-completed nodes remain cached and are reused on re-run).
    ``targets`` scopes execution to the given nodes and their ancestors.

    Raises:
        GraphValidationError: malformed graph (bad references, missing wiring).
        GraphCycleError: the graph contains a cycle.
        ExecutionCancelled: ``should_stop`` returned ``True`` between nodes.
        TranspileError: a node's code cannot be transpiled.
        UntrustedNodeError: a node's code is neither official nor approved.
        DependencyInstallError: required packages are missing and cannot
            be installed (or installs are disabled).
        NodeExecutionError: a node failed at runtime; a ``NODE_ERROR`` event
            with the raw source line was emitted first.
    """
    sink: EventSink = emit if emit is not None else _noop
    registry = security if security is not None else get_security_registry()
    nodes_by_id = _index_nodes(graph)
    order = _topological_order(nodes_by_id, graph.edges)
    allowed = _scope_to_targets(nodes_by_id, graph.edges, targets)
    if allowed is not None:
        order = [node_id for node_id in order if node_id in allowed]
    schemas = _plan(graph)
    incoming: dict[str, dict[str, GraphEdge]] = {
        node_id: {} for node_id in nodes_by_id
    }
    for edge in graph.edges:
        incoming[edge.target_node_id][edge.target_handle] = edge

    result = ExecutionResult()
    state_hashes: dict[str, str] = {}
    for node_id in order:
        if should_stop is not None and should_stop():
            raise ExecutionCancelled("Execution cancelled by user")
        node = nodes_by_id[node_id]
        schema = schemas[node_id]
        upstream_hashes = [
            state_hashes[incoming[node_id][handle].source_node_id]
            for handle in sorted(incoming[node_id])
        ]
        state_hash = _node_state_hash(node.code, node.widget_values, upstream_hashes)

        if not registry.is_trusted(node_id, node.code):
            raise UntrustedNodeError(node_id, code_hash(node.code))

        cached_entry = _CACHE.get(node_id)
        if cached_entry is not None and cached_entry[0] == state_hash:
            _record_outputs(result, node_id, schema, cached_entry[1], cached=True)
            state_hashes[node_id] = state_hash
            sink(
                ExecutionEvent(
                    type=EventType.NODE_COMPLETED,
                    node_id=node_id,
                    data={
                        "cached": True,
                        "outputs": result.previews[node_id],
                        "duration_ms": 0.0,
                    },
                )
            )
            continue

        missing = missing_dependencies(schema.dependencies)
        if missing:
            if not allow_installs:
                raise DependencyInstallError(
                    f"Node '{node_id}' requires missing dependencies: "
                    f"{', '.join(missing)}"
                )
            sink(
                ExecutionEvent(
                    type=EventType.INSTALLING_DEPS,
                    node_id=node_id,
                    data={"dependencies": missing},
                )
            )
            install_dependencies(missing)

        sink(ExecutionEvent(type=EventType.NODE_STARTED, node_id=node_id, data={}))
        node_started = time.perf_counter()
        transpiled = transpile_node(node.code, node_id)

        kwargs: dict[str, Any] = {
            widget.name: widget.default for widget in schema.widgets
        }
        kwargs.update(
            {
                key: value
                for key, value in node.widget_values.items()
                if key in kwargs
            }
        )
        for handle in sorted(incoming[node_id]):
            edge = incoming[node_id][handle]
            kwargs[handle] = result.outputs_by_name[edge.source_node_id][
                edge.source_handle
            ]

        try:
            value = transpiled.func(**kwargs)
            outputs = _normalize_outputs(value, len(schema.outputs), node_id)
        except Exception as exc:
            line, traceback_text = _localize_failure(
                exc, node_id, transpiled.line_offset
            )
            sink(
                ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=node_id,
                    data={
                        "line": line,
                        "message": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback_text,
                    },
                )
            )
            raise NodeExecutionError(
                node_id,
                line=line,
                message=str(exc),
                error_type=type(exc).__name__,
                traceback_text=traceback_text,
            ) from exc

        _CACHE[node_id] = (state_hash, outputs)
        state_hashes[node_id] = state_hash
        _record_outputs(result, node_id, schema, outputs, cached=False)
        sink(
            ExecutionEvent(
                type=EventType.NODE_COMPLETED,
                node_id=node_id,
                data={
                    "cached": False,
                    "outputs": result.previews[node_id],
                    "duration_ms": round(
                        (time.perf_counter() - node_started) * 1000, 1
                    ),
                },
            )
        )
    return result


def topological_sort(graph: WorkflowGraph) -> list[str]:
    """Return node ids in dependency order, raising on cycles."""
    nodes_by_id = _index_nodes(graph)
    return _topological_order(nodes_by_id, graph.edges)


def build_output_preview(data_type: SocketType, value: Any) -> Any:
    """Build the wire/node preview metadata described in the socket spec."""
    try:
        if data_type is SocketType.DATAFRAME:
            return _jsonable(
                {
                    "rows": value.height,
                    "cols": value.width,
                    "head": value.head(5).to_dicts(),
                }
            )
        if data_type is SocketType.DATASET:
            features, labels = value
            return _jsonable(
                {
                    "X_shape": tuple(features.shape),
                    "y_shape": tuple(labels.shape),
                    "dtype": str(features.dtype),
                }
            )
        if data_type is SocketType.ML_MODEL:
            if hasattr(value, "get_params"):
                return _jsonable(
                    {"class": type(value).__name__, "params": value.get_params()}
                )
            if hasattr(value, "state_dict"):
                state = value.state_dict()
                parameters = (
                    sum(int(param.numel()) for param in value.parameters())
                    if hasattr(value, "parameters")
                    else 0
                )
                return _jsonable(
                    {
                        "class": type(value).__name__,
                        "parameters": parameters,
                        "layers": len(state),
                    }
                )
            return repr(value)[:200]
        if data_type is SocketType.TENSOR:
            return _jsonable(
                {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(getattr(value, "device", "cpu")),
                    "sample": _tensor_sample(value),
                }
            )
        if data_type is SocketType.METRICS:
            return _jsonable(value)
    except Exception:
        return repr(value)[:200]
    return repr(value)[:200]


def _tensor_sample(value: Any, count: int = 5) -> list[float]:
    """Return the first ``count`` scalar values of a tensor-like object."""
    try:
        flat = (
            value.detach().reshape(-1)
            if hasattr(value, "detach")
            else value.reshape(-1)
        )
        sliced = flat[:count]
        items = sliced.tolist() if hasattr(sliced, "tolist") else list(sliced)
        return [float(item) for item in items]
    except Exception:
        return []


def _noop(event: ExecutionEvent) -> None:
    return None


def _index_nodes(graph: WorkflowGraph) -> dict[str, GraphNode]:
    nodes_by_id: dict[str, GraphNode] = {}
    for node in graph.nodes:
        if node.id in nodes_by_id:
            raise GraphValidationError(f"Duplicate node id '{node.id}'")
        nodes_by_id[node.id] = node
    return nodes_by_id


def _topological_order(
    nodes_by_id: dict[str, GraphNode], edges: list[GraphEdge]
) -> list[str]:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}
    indegree: dict[str, int] = {node_id: 0 for node_id in nodes_by_id}
    for edge in edges:
        if edge.source_node_id not in nodes_by_id:
            raise GraphValidationError(
                f"Edge references unknown source node '{edge.source_node_id}'"
            )
        if edge.target_node_id not in nodes_by_id:
            raise GraphValidationError(
                f"Edge references unknown target node '{edge.target_node_id}'"
            )
        adjacency[edge.source_node_id].append(edge.target_node_id)
        indegree[edge.target_node_id] += 1

    queue: deque[str] = deque(
        node_id for node_id in nodes_by_id if indegree[node_id] == 0
    )
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(order) != len(nodes_by_id):
        cyclic = sorted(set(nodes_by_id) - set(order))
        raise GraphCycleError(
            f"Workflow graph contains a cycle involving: {', '.join(cyclic)}"
        )
    return order


def _plan(graph: WorkflowGraph) -> dict[str, NodeSchema]:
    schemas: dict[str, NodeSchema] = {}
    for node in graph.nodes:
        schema = parse_node(node.code, node.id)
        if schema.syntax_errors:
            raise TranspileError(schema)
        schemas[node.id] = schema

    connected: set[tuple[str, str]] = set()
    for edge in graph.edges:
        source_schema = schemas[edge.source_node_id]
        if edge.source_handle not in {
            socket.name for socket in source_schema.outputs
        }:
            raise GraphValidationError(
                f"Node '{edge.source_node_id}' has no output socket "
                f"'{edge.source_handle}'"
            )
        target_schema = schemas[edge.target_node_id]
        target_handles = {
            socket.name for socket in target_schema.inputs
        } | {widget.name for widget in target_schema.widgets}
        if edge.target_handle not in target_handles:
            raise GraphValidationError(
                f"Node '{edge.target_node_id}' has no input socket or widget "
                f"'{edge.target_handle}'"
            )
        key = (edge.target_node_id, edge.target_handle)
        if key in connected:
            raise GraphValidationError(
                f"Input '{edge.target_handle}' of node "
                f"'{edge.target_node_id}' has multiple incoming connections"
            )
        connected.add(key)

    for node in graph.nodes:
        wired = {
            edge.target_handle
            for edge in graph.edges
            if edge.target_node_id == node.id
        }
        for socket in schemas[node.id].inputs:
            if socket.name not in wired:
                raise GraphValidationError(
                    f"Required input '{socket.name}' of node '{node.id}' "
                    f"is not connected"
                )
    return schemas


def _node_state_hash(
    code: str, widget_values: dict[str, Any], upstream_hashes: list[str]
) -> str:
    payload = (
        f"{code}{json.dumps(widget_values, sort_keys=True, default=str)}"
        f"{''.join(upstream_hashes)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_outputs(value: Any, expected: int, node_id: str) -> tuple[Any, ...]:
    if expected == 0:
        return ()
    if isinstance(value, tuple) and len(value) == expected:
        return value
    if isinstance(value, tuple):
        raise _OutputContractError(
            f"Node declared {expected} output socket(s) but returned "
            f"{len(value)} value(s)"
        )
    raise _OutputContractError(
        f"Node declared {expected} output socket(s) but returned a "
        f"non-tuple value of type {type(value).__name__}"
    )


def _record_outputs(
    result: ExecutionResult,
    node_id: str,
    schema: NodeSchema,
    outputs: tuple[Any, ...],
    *,
    cached: bool,
) -> None:
    result.outputs[node_id] = outputs
    result.outputs_by_name[node_id] = {
        socket.name: value for socket, value in zip(schema.outputs, outputs)
    }
    result.previews[node_id] = {
        socket.name: build_output_preview(socket.data_type, value)
        for socket, value in zip(schema.outputs, outputs)
    }
    result.cached[node_id] = cached


def _localize_failure(
    exc: Exception, node_id: str, line_offset: int
) -> tuple[int | None, str]:
    filename = f"<node:{node_id}>"
    raw_line: int | None = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == filename:
            raw_line = tb.tb_lineno - line_offset
        tb = tb.tb_next
    if raw_line is not None and raw_line < 1:
        raw_line = None
    text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    pattern = re.compile(rf'File "{re.escape(filename)}", line (\d+)')
    text = pattern.sub(
        lambda match: (
            f'File "{filename}", line {max(int(match.group(1)) - line_offset, 1)}'
        ),
        text,
    )
    return raw_line, text


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return repr(value)[:200]
