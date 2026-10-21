"""Topological DAG executor with content-addressable caching.

Executes a :class:`~app.schemas.WorkflowGraph` node by node in dependency
order, streams :class:`~app.schemas.ExecutionEvent` telemetry through an
optional sink, and caches each node's outputs keyed by

``sha256(node.code + json(widget_values) + "".join(upstream_hashes))``

so that re-running a graph only re-executes nodes whose code, widget values,
or transitive upstream state actually changed. ``MAX_CACHE_ENTRIES`` bounds the
number of retained intermediates, and ``MAX_CACHE_BYTES`` separately bounds
cached rich-output payloads such as base64 images and Plotly figures.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
import traceback
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .env_manager import (
    DependencyInstallError,
    install_dependencies,
    missing_dependencies,
)
from .output_capture import OutputCapture, rich_output_size
from .schemas import (
    EventType,
    ExecutionEvent,
    GraphEdge,
    GraphNode,
    NodeSchema,
    RichOutput,
    SocketType,
    WorkflowGraph,
    serialize_rich_outputs,
    socket_types_compatible,
    widget_choice_allowed,
    widget_socket_type,
)
from .security import SecurityRegistry, UntrustedNodeError, code_hash, get_security_registry
from .transpiler import TranspileError, parse_node, transpile_node
from .workspace import workspace_root

__all__ = [
    "EventSink",
    "ExecutionCancelled",
    "ExecutionResult",
    "FileAccessError",
    "GraphCycleError",
    "WidgetChoiceError",
    "GraphError",
    "GraphPlan",
    "GraphValidationError",
    "MAX_CACHE_BYTES",
    "MAX_CACHE_ENTRIES",
    "NodeExecutionError",
    "build_output_preview",
    "cache_snapshot",
    "clear_cache",
    "execute_graph",
    "restore_cache",
    "topological_sort",
    "transferable_cache",
    "validate_graph",
]

EventSink = Callable[[ExecutionEvent], None]

MAX_CACHE_ENTRIES = 64
MAX_CACHE_BYTES = 32 * 1024 * 1024


@dataclass(slots=True)
class _CacheEntry:
    state_hash: str
    outputs: tuple[Any, ...]
    rich_outputs: tuple[RichOutput, ...]
    size: int


_CACHE: OrderedDict[str, _CacheEntry] = OrderedDict()
_CACHE_BYTES = 0


def clear_cache() -> None:
    """Drop all cached node outputs."""
    global _CACHE_BYTES
    _CACHE.clear()
    _CACHE_BYTES = 0


def cache_snapshot() -> OrderedDict[str, _CacheEntry]:
    """Return a copy of the current cache for transfer to an executor process."""
    return OrderedDict(_CACHE.items())


def restore_cache(entries: dict[str, _CacheEntry] | None) -> None:
    """Replace the current cache with entries received from an executor process."""
    global _CACHE_BYTES
    _CACHE.clear()
    _CACHE_BYTES = 0
    for node_id, entry in (entries or {}).items():
        _CACHE[node_id] = entry
        _CACHE_BYTES += entry.size


def transferable_cache(
    entries: dict[str, _CacheEntry] | None = None,
) -> OrderedDict[str, _CacheEntry]:
    """Drop cache entries whose runtime outputs cannot cross a process boundary."""
    source = _CACHE if entries is None else entries
    transferable: OrderedDict[str, _CacheEntry] = OrderedDict()
    for node_id, entry in source.items():
        try:
            pickle.dumps(entry)
        except Exception:
            continue
        transferable[node_id] = entry
    return transferable


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


class FileAccessError(Exception):
    """Raised when a File widget points outside the workspace."""


class WidgetChoiceError(Exception):
    """Raised when a widget value is outside its declared Literal choices."""


@dataclass(slots=True)
class ExecutionResult:
    outputs: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    outputs_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    previews: dict[str, dict[str, Any]] = field(default_factory=dict)
    rich_outputs: dict[str, list[RichOutput]] = field(default_factory=dict)
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
    approved for this session. When ``targets`` is given, parsing, wiring,
    cycle detection and the trust check are all scoped to those nodes and their
    ancestors (run-to-node); unrelated nodes and edges are ignored.
    """
    registry = security if security is not None else get_security_registry()
    nodes_by_id = _index_nodes(graph)
    allowed = _scope_to_targets(nodes_by_id, graph.edges, targets)
    nodes, edges = _scoped_graph(graph, allowed)
    order = _topological_order({node.id: node for node in nodes}, edges)
    schemas = _plan(nodes, edges)
    untrusted = [
        (node.id, code_hash(node.code))
        for node in nodes
        if not registry.is_trusted(node.approval_id or node.id, node.code)
    ]
    return GraphPlan(order=order, schemas=schemas, untrusted=untrusted)


def _scoped_graph(
    graph: WorkflowGraph, allowed: set[str] | None
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Return the nodes and edges in scope (everything when ``allowed`` is None)."""
    if allowed is None:
        return list(graph.nodes), list(graph.edges)
    nodes = [node for node in graph.nodes if node.id in allowed]
    edges = [edge for edge in graph.edges if edge.target_node_id in allowed]
    return nodes, edges


def _scope_to_targets(
    nodes_by_id: dict[str, GraphNode],
    edges: Sequence[GraphEdge],
    targets: Sequence[str] | None,
) -> set[str] | None:
    """Return the target nodes plus all their ancestors (None = whole graph).

    Edges whose target is not a known node are skipped here; topological
    validation reports them once the relevant edge is in scope.
    """
    if targets is None:
        return None
    target_list = list(targets)
    if not target_list:
        raise GraphValidationError("targets must not be empty")
    unknown = [node_id for node_id in target_list if node_id not in nodes_by_id]
    if unknown:
        raise GraphValidationError(f"Unknown target node(s): {', '.join(sorted(unknown))}")
    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}
    for edge in edges:
        if edge.target_node_id in incoming:
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
    plan: GraphPlan | None = None,
) -> ExecutionResult:
    """Execute a workflow graph and return per-node outputs and previews.

    ``should_stop`` is polled between nodes; when it returns ``True``,
    :class:`ExecutionCancelled` is raised before the next node starts (outputs
    of already-completed nodes remain cached and are reused on re-run).
    ``targets`` scopes execution to the given nodes and their ancestors.

    ``plan`` may carry a pre-flight :class:`GraphPlan` from
    :func:`validate_graph`; when supplied it defines the run scope, avoids
    re-parsing every node, and ``targets`` is ignored.

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
    if plan is None:
        plan = validate_graph(graph, security=registry, targets=targets)
    nodes_by_id = _index_nodes(graph)
    order = plan.order
    schemas = plan.schemas
    incoming: dict[str, dict[str, GraphEdge]] = {node_id: {} for node_id in order}
    for edge in graph.edges:
        if edge.target_node_id in incoming:
            incoming[edge.target_node_id][edge.target_handle] = edge

    result = ExecutionResult()
    state_hashes: dict[str, str] = {}
    for node_id in order:
        if should_stop is not None and should_stop():
            raise ExecutionCancelled("Execution cancelled by user")
        node = nodes_by_id[node_id]
        schema = schemas[node_id]
        upstream_hashes = [
            f"{handle}:{state_hashes[incoming[node_id][handle].source_node_id]}"
            for handle in sorted(incoming[node_id])
        ]
        state_hash = _node_state_hash(node.code, node.widget_values, upstream_hashes)

        if not registry.is_trusted(node.approval_id or node_id, node.code):
            raise UntrustedNodeError(node_id, code_hash(node.code))

        cached_entry = _cache_lookup(node_id, state_hash)
        if cached_entry is not None:
            _record_outputs(
                result,
                node_id,
                schema,
                cached_entry.outputs,
                cached_entry.rich_outputs,
                cached=True,
            )
            state_hashes[node_id] = state_hash
            sink(
                ExecutionEvent(
                    type=EventType.NODE_COMPLETED,
                    node_id=node_id,
                    data={
                        "cached": True,
                        "outputs": result.previews[node_id],
                        "rich_outputs": serialize_rich_outputs(result.rich_outputs[node_id]),
                        "duration_ms": 0.0,
                    },
                )
            )
            continue

        missing = missing_dependencies(schema.dependencies)
        if missing:
            if not allow_installs:
                error = DependencyInstallError(
                    f"Node '{node_id}' requires missing dependencies: {', '.join(missing)}"
                )
                _emit_dependency_error(sink, node_id, error)
                raise error
            sink(
                ExecutionEvent(
                    type=EventType.INSTALLING_DEPS,
                    node_id=node_id,
                    data={"dependencies": missing},
                )
            )
            try:
                install_dependencies(missing, should_stop=should_stop)
            except DependencyInstallError as exc:
                if should_stop is not None and should_stop():
                    raise ExecutionCancelled(
                        "Execution cancelled during dependency install"
                    ) from exc
                _emit_dependency_error(sink, node_id, exc)
                raise
            if should_stop is not None and should_stop():
                raise ExecutionCancelled("Execution cancelled after dependency install")

        sink(ExecutionEvent(type=EventType.NODE_STARTED, node_id=node_id, data={}))
        node_started = time.perf_counter()
        transpiled = transpile_node(node.code, node_id)

        kwargs: dict[str, Any] = {widget.name: widget.default for widget in schema.widgets}
        kwargs.update({key: value for key, value in node.widget_values.items() if key in kwargs})
        for handle in sorted(incoming[node_id]):
            edge = incoming[node_id][handle]
            kwargs[handle] = result.outputs_by_name[edge.source_node_id][edge.source_handle]
        try:
            for widget in schema.widgets:
                if widget.data_type.strip().lower() == "file":
                    kwargs[widget.name] = _resolve_widget_file(kwargs.get(widget.name))
                if widget.options and not widget_choice_allowed(
                    kwargs.get(widget.name), widget.options
                ):
                    raise WidgetChoiceError(
                        f"Widget '{widget.name}' on node '{node_id}' must be "
                        f"one of {widget.options!r}"
                    )
        except (FileAccessError, WidgetChoiceError) as exc:
            sink(
                ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=node_id,
                    data={
                        "line": None,
                        "message": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": "",
                        "rich_outputs": [],
                    },
                )
            )
            raise NodeExecutionError(
                node_id,
                line=None,
                message=str(exc),
                error_type=type(exc).__name__,
                traceback_text="",
            ) from exc

        capture = OutputCapture()
        try:
            with capture.activate(transpiled.func):
                value = transpiled.func(**kwargs)
            outputs = _normalize_outputs(value, len(schema.outputs), node_id)
        except Exception as exc:
            line, traceback_text = _localize_failure(exc, node_id, transpiled.line_offset)
            sink(
                ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=node_id,
                    data={
                        "line": line,
                        "message": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback_text,
                        "rich_outputs": serialize_rich_outputs(capture.outputs),
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

        rich_outputs = tuple(capture.outputs)
        _cache_store(node_id, state_hash, outputs, rich_outputs)
        state_hashes[node_id] = state_hash
        _record_outputs(result, node_id, schema, outputs, rich_outputs, cached=False)
        sink(
            ExecutionEvent(
                type=EventType.NODE_COMPLETED,
                node_id=node_id,
                data={
                    "cached": False,
                    "outputs": result.previews[node_id],
                    "rich_outputs": serialize_rich_outputs(result.rich_outputs[node_id]),
                    "duration_ms": round((time.perf_counter() - node_started) * 1000, 1),
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
                return _jsonable({"class": type(value).__name__, "params": value.get_params()})
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
        if data_type in {
            SocketType.INT,
            SocketType.FLOAT,
            SocketType.BOOL,
            SocketType.STRING,
        }:
            return _jsonable(value)
    except Exception:
        return repr(value)[:200]
    return repr(value)[:200]


def _tensor_sample(value: Any, count: int = 5) -> list[float]:
    """Return the first ``count`` scalar values of a tensor-like object."""
    try:
        flat = value.detach().reshape(-1) if hasattr(value, "detach") else value.reshape(-1)
        sliced = flat[:count]
        items = sliced.tolist() if hasattr(sliced, "tolist") else list(sliced)
        return [float(item) for item in items]
    except Exception:
        return []


def _noop(event: ExecutionEvent) -> None:
    return None


def _emit_dependency_error(sink: EventSink, node_id: str, exc: Exception) -> None:
    sink(
        ExecutionEvent(
            type=EventType.NODE_ERROR,
            node_id=node_id,
            data={
                "line": None,
                "message": str(exc),
                "error_type": type(exc).__name__,
                "traceback": "",
            },
        )
    )


def _cache_lookup(node_id: str, state_hash: str) -> _CacheEntry | None:
    entry = _CACHE.get(node_id)
    if entry is None or entry.state_hash != state_hash:
        return None
    _CACHE.move_to_end(node_id)
    return entry


def _cache_store(
    node_id: str,
    state_hash: str,
    outputs: tuple[Any, ...],
    rich_outputs: tuple[RichOutput, ...],
) -> None:
    global _CACHE_BYTES
    previous = _CACHE.pop(node_id, None)
    if previous is not None:
        _CACHE_BYTES -= previous.size
    size = sum(rich_output_size(output) for output in rich_outputs)
    _CACHE[node_id] = _CacheEntry(
        state_hash=state_hash,
        outputs=outputs,
        rich_outputs=rich_outputs,
        size=size,
    )
    _CACHE_BYTES += size
    _CACHE.move_to_end(node_id)
    while _CACHE and (len(_CACHE) > MAX_CACHE_ENTRIES or _CACHE_BYTES > MAX_CACHE_BYTES):
        _, evicted = _CACHE.popitem(last=False)
        _CACHE_BYTES -= evicted.size


def _index_nodes(graph: WorkflowGraph) -> dict[str, GraphNode]:
    nodes_by_id: dict[str, GraphNode] = {}
    for node in graph.nodes:
        if node.id in nodes_by_id:
            raise GraphValidationError(f"Duplicate node id '{node.id}'")
        nodes_by_id[node.id] = node
    return nodes_by_id


def _topological_order(nodes_by_id: dict[str, GraphNode], edges: list[GraphEdge]) -> list[str]:
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

    queue: deque[str] = deque(node_id for node_id in nodes_by_id if indegree[node_id] == 0)
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
        raise GraphCycleError(f"Workflow graph contains a cycle involving: {', '.join(cyclic)}")
    return order


def _plan(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> dict[str, NodeSchema]:
    schemas: dict[str, NodeSchema] = {}
    for node in nodes:
        schema = parse_node(node.code, node.id)
        if schema.syntax_errors:
            raise TranspileError(schema)
        schemas[node.id] = schema

    connected: set[tuple[str, str]] = set()
    wired: dict[str, set[str]] = {}
    for edge in edges:
        source_schema = schemas[edge.source_node_id]
        source_socket = next(
            (
                socket
                for socket in source_schema.outputs
                if socket.name == edge.source_handle
            ),
            None,
        )
        if source_socket is None:
            raise GraphValidationError(
                f"Node '{edge.source_node_id}' has no output socket '{edge.source_handle}'"
            )
        target_schema = schemas[edge.target_node_id]
        target_socket = next(
            (
                socket
                for socket in target_schema.inputs
                if socket.name == edge.target_handle
            ),
            None,
        )
        target_widget = next(
            (
                widget
                for widget in target_schema.widgets
                if widget.name == edge.target_handle
            ),
            None,
        )
        if target_socket is not None:
            target_type = target_socket.data_type
        elif target_widget is not None:
            target_type = widget_socket_type(target_widget.data_type)
        else:
            raise GraphValidationError(
                f"Node '{edge.target_node_id}' has no input socket or widget '{edge.target_handle}'"
            )
        if not socket_types_compatible(source_socket.data_type, target_type):
            raise GraphValidationError(
                f"Type mismatch {source_socket.data_type.value} -> "
                f"{target_type.value} for {edge.source_node_id}."
                f"{edge.source_handle} -> {edge.target_node_id}.{edge.target_handle}"
            )
        key = (edge.target_node_id, edge.target_handle)
        if key in connected:
            raise GraphValidationError(
                f"Input '{edge.target_handle}' of node "
                f"'{edge.target_node_id}' has multiple incoming connections"
            )
        connected.add(key)
        wired.setdefault(edge.target_node_id, set()).add(edge.target_handle)

    for node in nodes:
        handles = wired.get(node.id, set())
        for socket in schemas[node.id].inputs:
            if socket.name not in handles:
                raise GraphValidationError(
                    f"Required input '{socket.name}' of node '{node.id}' is not connected"
                )
    return schemas


def _resolve_widget_file(value: Any) -> Any:
    """Resolve workspace-relative ``File`` widget paths inside the workspace."""
    if not isinstance(value, str) or not value.strip():
        return value
    root = workspace_root().expanduser().resolve()
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise FileAccessError("File widgets may only access files inside the workspace.")
    return str(resolved)


def _node_state_hash(code: str, widget_values: dict[str, Any], upstream_hashes: list[str]) -> str:
    payload = (
        f"{code}{json.dumps(widget_values, sort_keys=True, default=str)}{''.join(upstream_hashes)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_outputs(value: Any, expected: int, node_id: str) -> tuple[Any, ...]:
    if expected == 0:
        return ()
    if isinstance(value, tuple) and len(value) == expected:
        return value
    if isinstance(value, tuple):
        raise _OutputContractError(
            f"Node declared {expected} output socket(s) but returned {len(value)} value(s)"
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
    rich_outputs: Sequence[RichOutput],
    *,
    cached: bool,
) -> None:
    result.outputs[node_id] = outputs
    result.outputs_by_name[node_id] = {
        socket.name: value for socket, value in zip(schema.outputs, outputs, strict=False)
    }
    result.previews[node_id] = {
        socket.name: build_output_preview(socket.data_type, value)
        for socket, value in zip(schema.outputs, outputs, strict=False)
    }
    result.rich_outputs[node_id] = list(rich_outputs)
    result.cached[node_id] = cached


def _localize_failure(exc: Exception, node_id: str, line_offset: int) -> tuple[int | None, str]:
    filename = f"<node:{node_id}>"
    raw_line: int | None = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == filename:
            raw_line = tb.tb_lineno - line_offset
        tb = tb.tb_next
    if raw_line is not None and raw_line < 1:
        raw_line = None
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    pattern = re.compile(rf'File "{re.escape(filename)}", line (\d+)')
    text = pattern.sub(
        lambda match: f'File "{filename}", line {max(int(match.group(1)) - line_offset, 1)}',
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
