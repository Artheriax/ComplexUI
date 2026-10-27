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

import base64
import hashlib
import io
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
from urllib.parse import quote

import numpy as np

from .datasets import (
    IMAGE_FORMATS,
    VIDEO_FORMATS,
    DatasetNotFound,
    InvalidAssetPath,
    InvalidDataset,
    get_dataset_store,
    get_output_store,
)
from .env_manager import (
    DependencyInstallError,
    install_dependencies,
    missing_dependencies,
)
from .models import ModelError, get_model_store
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
    "ModelAccessError",
    "ModelSelectionError",
    "WidgetChoiceError",
    "GraphError",
    "GraphPlan",
    "GraphValidationError",
    "MAX_BATCH_ITEMS",
    "MAX_CACHE_BYTES",
    "MAX_CACHE_ENTRIES",
    "NON_CACHEABLE_SOCKET_TYPES",
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

MAX_BATCH_ITEMS = 256
MAX_FOLDER_FINGERPRINT_FILES = 512
MAX_ANY_PREVIEW_CHARS = 20_000
MAX_CACHE_ENTRIES = 64
MAX_CACHE_BYTES = 32 * 1024 * 1024
NON_CACHEABLE_SOCKET_TYPES = {
    SocketType.MODEL,
    SocketType.CLIP,
    SocketType.VAE,
    SocketType.GUIDER,
}


@dataclass(slots=True)
class _CacheEntry:
    state_hash: str
    outputs: tuple[Any, ...]
    rich_outputs: tuple[RichOutput, ...]
    size: int


@dataclass(slots=True)
class _TensorPayload:
    """Pickle-safe replacement for torch tensors in cross-process caches."""

    data: bytes


def _encode_cache_value(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        return value
    if isinstance(value, torch.Tensor):
        buffer = io.BytesIO()
        torch.save(value.detach().to("cpu").contiguous(), buffer)
        return _TensorPayload(buffer.getvalue())
    if isinstance(value, dict):
        return {key: _encode_cache_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_cache_value(item) for item in value)
    return value


def _contains_torch_module(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if isinstance(value, torch.nn.Module):
        return True
    if isinstance(value, dict):
        return any(_contains_torch_module(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_torch_module(item) for item in value)
    return False


def _decode_cache_value(value: Any) -> Any:
    if isinstance(value, _TensorPayload):
        import torch

        return torch.load(
            io.BytesIO(value.data),
            map_location="cpu",
            weights_only=True,
        )
    if isinstance(value, dict):
        return {key: _decode_cache_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_cache_value(item) for item in value)
    return value


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
        decoded = _CacheEntry(
            state_hash=entry.state_hash,
            outputs=_decode_cache_value(entry.outputs),
            rich_outputs=_decode_cache_value(entry.rich_outputs),
            size=entry.size,
        )
        _CACHE[node_id] = decoded
        _CACHE_BYTES += decoded.size


def transferable_cache(
    entries: dict[str, _CacheEntry] | None = None,
) -> OrderedDict[str, _CacheEntry]:
    """Drop cache entries whose runtime outputs cannot cross a process boundary."""
    source = _CACHE if entries is None else entries
    transferable: OrderedDict[str, _CacheEntry] = OrderedDict()
    for node_id, entry in source.items():
        encoded = _CacheEntry(
            state_hash=entry.state_hash,
            outputs=_encode_cache_value(entry.outputs),
            rich_outputs=_encode_cache_value(entry.rich_outputs),
            size=entry.size,
        )
        if _contains_torch_module(encoded):
            continue
        try:
            pickle.dumps(encoded)
        except Exception:
            continue
        transferable[node_id] = encoded
    return transferable


class GraphError(Exception):
    """Base class for workflow graph problems."""


class GraphValidationError(GraphError):
    """Raised when the workflow graph is malformed (wiring, references)."""


class ModelSelectionError(GraphValidationError):
    """Raised when a selected local model is stale or invalid."""

    def __init__(self, issues: list[dict[str, str]]) -> None:
        summary = "; ".join(
            f"{issue['node_id']}.{issue['widget']}: {issue['message']}"
            for issue in issues
        )
        super().__init__("Invalid local model selection(s): " + summary)
        self.issues = issues


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


class ModelAccessError(Exception):
    """Raised when a local model selection is missing or invalid."""


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
    _validate_model_selections(nodes, schemas)
    untrusted = [
        (node.id, code_hash(node.code))
        for node in nodes
        if not registry.is_trusted(node.approval_id or node.id, node.code)
    ]
    return GraphPlan(order=order, schemas=schemas, untrusted=untrusted)


def _validate_model_selections(
    nodes: Sequence[GraphNode],
    schemas: dict[str, NodeSchema],
) -> None:
    """Fail pre-flight when a selected local model is stale or invalid."""
    issues: list[dict[str, str]] = []
    for node in nodes:
        schema = schemas[node.id]
        for widget in schema.widgets:
            if widget.data_type.strip().lower() != "modelselector":
                continue
            selected = node.widget_values.get(widget.name, widget.default)
            if not isinstance(selected, str) or not selected.strip():
                continue
            try:
                get_model_store().resolve(widget.model_type or "", selected)
            except ModelError as exc:
                issues.append(
                    {
                        "node_id": node.id,
                        "widget": widget.name,
                        "model_type": widget.model_type or "",
                        "path": selected,
                        "message": str(exc),
                    }
                )
    if issues:
        raise ModelSelectionError(issues)


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
        widget_values = dict(node.widget_values)
        widget_names = {widget.name for widget in schema.widgets}
        for handle, edge in incoming[node_id].items():
            if handle in widget_names:
                widget_values[handle] = result.outputs_by_name[
                    edge.source_node_id
                ][edge.source_handle]
        state_hash = _node_state_hash(
            node.code,
            node.widget_values,
            upstream_hashes,
            _widget_fingerprints(node, schema, widget_values),
        )

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
        batch_size = 1
        current_index = 0
        progress_state = {"batch_size": 1, "index": 0}

        def report_progress(
            fraction: float,
            message: str | None = None,
            *,
            _node_id: str = node_id,
            _state: dict[str, int] = progress_state,
        ) -> None:
            try:
                value = float(fraction)
            except (TypeError, ValueError):
                return
            total = max(_state["batch_size"], 1)
            index = _state["index"]
            progress = min(1.0, max(0.0, (index + value) / total))
            data: dict[str, Any] = {"progress": progress}
            if message is not None:
                label = str(message)[:200]
                if total > 1:
                    label = f"{index + 1}/{total}: {label}"
                data["message"] = label
            sink(
                ExecutionEvent(
                    type=EventType.NODE_PROGRESS,
                    node_id=_node_id,
                    data=data,
                )
            )

        def raise_if_cancelled() -> None:
            if should_stop is not None and should_stop():
                raise ExecutionCancelled("Execution cancelled by user")

        transpiled.func.__globals__["report_progress"] = report_progress
        transpiled.func.__globals__["raise_if_cancelled"] = raise_if_cancelled
        transpiled.func.__globals__["outputs_root"] = (
            get_output_store().outputs_dir.resolve()
        )

        kwargs: dict[str, Any] = {widget.name: widget.default for widget in schema.widgets}
        kwargs.update({key: value for key, value in node.widget_values.items() if key in kwargs})
        for handle in sorted(incoming[node_id]):
            edge = incoming[node_id][handle]
            kwargs[handle] = result.outputs_by_name[edge.source_node_id][edge.source_handle]
        try:
            for widget in schema.widgets:
                widget_kind = widget.data_type.strip().lower()
                if widget_kind in {"file", "imagefile", "videofile"}:
                    kwargs[widget.name] = _resolve_widget_file(kwargs.get(widget.name))
                elif widget_kind in {"directory", "folder"}:
                    kwargs[widget.name] = _resolve_widget_directory(
                        kwargs.get(widget.name)
                    )
                elif widget_kind == "folderpath":
                    kwargs[widget.name] = _resolve_widget_image_folder(
                        kwargs.get(widget.name)
                    )
                elif widget_kind == "modelpath":
                    kwargs[widget.name] = _resolve_widget_model_path(
                        kwargs.get(widget.name)
                    )
                elif widget_kind == "outputfile":
                    kwargs[widget.name] = _resolve_output_file(kwargs.get(widget.name))
                elif widget_kind == "modelselector":
                    kwargs[widget.name] = _resolve_model_selection(
                        widget.model_type,
                        kwargs.get(widget.name),
                    )
                if widget.options and not widget_choice_allowed(
                    kwargs.get(widget.name), widget.options
                ):
                    raise WidgetChoiceError(
                        f"Widget '{widget.name}' on node '{node_id}' must be "
                        f"one of {widget.options!r}"
                    )
        except (FileAccessError, ModelAccessError, WidgetChoiceError) as exc:
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

        def fail_batch(
            message: str,
            *,
            _node_id: str = node_id,
        ) -> None:
            sink(
                ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=_node_id,
                    data={
                        "line": None,
                        "message": message,
                        "error_type": "BatchInputError",
                        "traceback": "",
                        "rich_outputs": [],
                    },
                )
            )
            raise NodeExecutionError(
                _node_id,
                line=None,
                message=message,
                error_type="BatchInputError",
                traceback_text="",
            )

        batch_inputs = [
            socket.name
            for socket in schema.inputs
            if socket.data_type is SocketType.IMAGE
            and isinstance(kwargs.get(socket.name), (list, tuple))
        ]
        batch_lengths = {name: len(kwargs[name]) for name in batch_inputs}
        if batch_lengths:
            if len(set(batch_lengths.values())) != 1:
                detail = ", ".join(
                    f"{name}={length}" for name, length in batch_lengths.items()
                )
                fail_batch(
                    "Image batch inputs must have matching lengths: " + detail
                )
            batch_size = next(iter(batch_lengths.values()))
            if batch_size > MAX_BATCH_ITEMS:
                fail_batch(
                    f"Image batch has {batch_size} items; the limit is "
                    f"{MAX_BATCH_ITEMS}."
                )
        progress_state["batch_size"] = batch_size
        transpiled.func.__globals__["batch_size"] = batch_size
        transpiled.func.__globals__["batch_index"] = 0

        capture = OutputCapture()
        try:
            with capture.activate(transpiled.func):
                if batch_size == 0:
                    outputs = tuple([] for _ in schema.outputs)
                elif batch_inputs:
                    buckets: list[list[Any]] = [
                        [] for _ in schema.outputs
                    ]
                    for index in range(batch_size):
                        if should_stop is not None and should_stop():
                            raise ExecutionCancelled(
                                "Execution cancelled by user"
                            )
                        current_index = index
                        progress_state["index"] = index
                        transpiled.func.__globals__["batch_index"] = index
                        item_kwargs = {
                            key: (
                                value[index]
                                if key in batch_inputs
                                else value
                            )
                            for key, value in kwargs.items()
                        }
                        value = transpiled.func(**item_kwargs)
                        item_outputs = _normalize_outputs(
                            value, len(schema.outputs), node_id
                        )
                        for bucket, item in zip(
                            buckets, item_outputs, strict=True
                        ):
                            bucket.append(item)
                    outputs = tuple(buckets)
                else:
                    value = transpiled.func(**kwargs)
                    outputs = _normalize_outputs(
                        value, len(schema.outputs), node_id
                    )
        except ExecutionCancelled:
            raise
        except Exception as exc:
            line, traceback_text = _localize_failure(exc, node_id, transpiled.line_offset)
            message = str(exc)
            if batch_inputs and batch_size > 1:
                message = (
                    f"Batch item {current_index + 1}/{batch_size}: {message}"
                )
            error_data: dict[str, Any] = {
                "line": line,
                "message": message,
                "error_type": type(exc).__name__,
                "traceback": traceback_text,
                "rich_outputs": serialize_rich_outputs(capture.outputs),
            }
            if batch_inputs:
                error_data["batch_index"] = current_index
                error_data["batch_size"] = batch_size
            sink(
                ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=node_id,
                    data=error_data,
                )
            )
            raise NodeExecutionError(
                node_id,
                line=line,
                message=message,
                error_type=type(exc).__name__,
                traceback_text=traceback_text,
            ) from exc

        for socket, output in zip(schema.outputs, outputs, strict=True):
            if (
                socket.data_type in {SocketType.IMAGE, SocketType.IMAGE_BATCH}
                and isinstance(output, (list, tuple))
                and len(output) > MAX_BATCH_ITEMS
            ):
                fail_batch(
                    f"Image batch has {len(output)} items; the limit is "
                    f"{MAX_BATCH_ITEMS}."
                )

        rich_outputs = tuple(capture.outputs)
        if all(
            socket.data_type not in NON_CACHEABLE_SOCKET_TYPES
            for socket in schema.outputs
        ):
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


_NO_BATCH_PREVIEW = object()


def _batch_value_preview(data_type: SocketType, value: Any) -> Any:
    if isinstance(value, list):
        if data_type in {
            SocketType.IMAGE,
            SocketType.IMAGE_BATCH,
            SocketType.MEDIA,
        }:
            return _image_batch_or_batch_preview(value)
        return _batch_preview(value)
    if isinstance(value, tuple) and data_type in {
        SocketType.IMAGE,
        SocketType.IMAGE_BATCH,
        SocketType.MEDIA,
    }:
        return _image_batch_or_batch_preview(value)
    return _NO_BATCH_PREVIEW


def build_output_preview(data_type: SocketType, value: Any) -> Any:
    """Build the wire/node preview metadata described in the socket spec."""
    try:
        if data_type is SocketType.ANY:
            return _any_preview(value)
        batch_preview = _batch_value_preview(data_type, value)
        if batch_preview is not _NO_BATCH_PREVIEW:
            return batch_preview
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
        if data_type is SocketType.IMAGE:
            return _image_preview(value)
        if data_type is SocketType.VIDEO:
            return _video_preview(value)
        if data_type is SocketType.MEDIA:
            return _media_preview(value)
        if data_type is SocketType.IMAGE_BATCH:
            return _image_batch_preview(value)
        if data_type is SocketType.MODEL:
            return _diffusion_model_preview(value)
        if data_type is SocketType.CLIP:
            return _clip_preview(value)
        if data_type is SocketType.VAE:
            return _vae_preview(value)
        if data_type is SocketType.CONDITIONING:
            return _conditioning_preview(value)
        if data_type is SocketType.LATENT:
            return _latent_preview(value)
        if data_type is SocketType.SIGMAS:
            return _sigmas_preview(value)
        if data_type is SocketType.SAMPLER:
            return _sampler_preview(value)
        if data_type is SocketType.NOISE:
            return _noise_preview(value)
        if data_type is SocketType.GUIDER:
            return _guider_preview(value)
        if data_type in {
            SocketType.DETECTION_MODEL,
            SocketType.SEGMENTATION_MODEL,
            SocketType.CLASSIFICATION_MODEL,
        }:
            return _vision_model_preview(value)
        if data_type is SocketType.DETECTIONS:
            return _detections_preview(value)
        if data_type is SocketType.MASKS:
            return _masks_preview(value)
        if data_type is SocketType.CLASSIFICATIONS:
            return _classifications_preview(value)
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


def _image_preview(value: Any, max_side: int = 512) -> dict[str, Any]:
    """Return image metadata plus a bounded JPEG data-URL thumbnail."""
    width = int(getattr(value, "width", 0))
    height = int(getattr(value, "height", 0))
    preview: dict[str, Any] = {
        "kind": "image",
        "width": width,
        "height": height,
        "mode": str(getattr(value, "mode", "")),
    }
    try:
        from PIL import Image

        if not isinstance(value, Image.Image):
            raise TypeError("Not a PIL image")
        thumbnail = value.convert("RGB")
        thumbnail.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        thumbnail.save(buffer, format="JPEG", quality=82)
        preview["thumbnail"] = (
            "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        )
    except Exception:
        preview["thumbnail"] = None
    return preview


def _video_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "video", "path": None, "error": "Unsupported video value"}
    path = value.get("path")
    return {
        "kind": "video",
        "path": path if isinstance(path, str) else None,
        "url": _media_url(path),
        "fps": _optional_number(value.get("fps")),
        "frames": _optional_number(value.get("frames")),
        "width": _optional_number(value.get("width")),
        "height": _optional_number(value.get("height")),
        "duration": _optional_number(value.get("duration")),
        "poster": value.get("poster") if isinstance(value.get("poster"), str) else None,
    }


def _media_preview(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and str(value.get("media_type", "")).lower() == "video":
        return _video_preview(value)
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            return _image_preview(value)
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, Image.Image) for item in value
        ):
            return _image_batch_preview(value)
    except ImportError:
        pass
    if isinstance(value, dict):
        return {"kind": "media", **_jsonable(value)}
    return {"kind": "media", "value": repr(value)[:200]}


def _image_batch_preview(value: Any) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {"kind": "image_batch", "count": 0, "thumbnail": None}
    images = (
        [item for item in value if isinstance(item, Image.Image)]
        if isinstance(value, (list, tuple))
        else []
    )
    first = images[0] if images else None
    preview: dict[str, Any] = {
        "kind": "image_batch",
        "count": len(value) if isinstance(value, (list, tuple)) else 0,
        "width": int(first.width) if first is not None else None,
        "height": int(first.height) if first is not None else None,
        "mode": str(first.mode) if first is not None else "",
        "thumbnail": None,
    }
    if not images:
        return preview
    try:
        cells = []
        for image in images[:4]:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((240, 240))
            cells.append(thumbnail)
        widths = [cell.width for cell in cells]
        heights = [cell.height for cell in cells]
        columns = 2 if len(cells) > 1 else 1
        rows = (len(cells) + columns - 1) // columns
        cell_width = max(widths)
        cell_height = max(heights)
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows * cell_height),
            (9, 9, 11),
        )
        for index, cell in enumerate(cells):
            x = (index % columns) * cell_width + (cell_width - cell.width) // 2
            y = (index // columns) * cell_height + (cell_height - cell.height) // 2
            sheet.paste(cell, (x, y))
        buffer = io.BytesIO()
        sheet.save(buffer, format="JPEG", quality=82)
        preview["thumbnail"] = (
            "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        )
    except Exception:
        preview["thumbnail"] = None
    return preview


def _diffusion_model_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "diffusion_model", "name": repr(value)[:200]}
    unet = value.get("unet")
    parameters = None
    if unet is not None and hasattr(unet, "parameters"):
        try:
            parameters = sum(int(param.numel()) for param in unet.parameters())
        except Exception:
            parameters = None
    return {
        "kind": "diffusion_model",
        "arch": str(value.get("arch", "")),
        "name": str(value.get("name", "")),
        "class": type(unet).__name__ if unet is not None else "UNet",
        "parameters": parameters,
        "device": value.get("device"),
        "dtype": str(value.get("dtype", "")),
        "source": str(value.get("source", "local")),
        "model_type": value.get("model_type"),
        "path": value.get("path"),
        "format": value.get("format"),
        "fingerprint": value.get("fingerprint"),
        "preset": value.get("preset"),
        "recommended": _jsonable(value.get("recommended") or {}),
    }


def _clip_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "clip", "arch": "", "encoders": 0}
    return {
        "kind": "clip",
        "arch": str(value.get("arch", "")),
        "encoders": 2 if value.get("text_encoder_2") is not None else 1,
        "device": value.get("device"),
        "dtype": str(value.get("dtype", "")),
    }


def _vae_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "vae", "arch": ""}
    return {
        "kind": "vae",
        "arch": str(value.get("arch", "")),
        "scaling_factor": _optional_number(value.get("scaling_factor")),
        "shift_factor": _optional_number(value.get("shift_factor")),
        "device": value.get("device"),
        "dtype": str(value.get("dtype", "")),
    }


def _conditioning_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "kind": "conditioning",
            "arch": "",
            "shape": [],
            "pooled_shape": None,
            "text": "",
        }
    tensor = value.get("cond")
    pooled = value.get("pooled")
    return {
        "kind": "conditioning",
        "arch": str(value.get("arch", "")),
        "shape": list(getattr(tensor, "shape", ())),
        "pooled_shape": (
            list(getattr(pooled, "shape", ())) if pooled is not None else None
        ),
        "text": str(value.get("text", ""))[:200],
    }


def _latent_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "latent", "shape": [], "dtype": "", "device": ""}
    tensor = value.get("samples")
    return {
        "kind": "latent",
        "shape": list(getattr(tensor, "shape", ())),
        "dtype": str(getattr(tensor, "dtype", "")),
        "device": str(getattr(tensor, "device", "")),
        "width": value.get("width"),
        "height": value.get("height"),
        "batch_size": value.get("batch_size"),
    }


def _sigmas_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "sigmas", "sigmas": [], "steps": 0, "scheduler": ""}
    tensor = value.get("sigmas")
    values: list[float] = []
    try:
        if isinstance(tensor, (list, tuple)):
            values = [float(item) for item in tensor]
        else:
            detach = getattr(tensor, "detach", None)
            if callable(detach):
                detached: Any = detach()
                raw = detached.cpu().flatten().tolist()
                values = [float(item) for item in raw]
    except Exception:
        values = []
    sampled = values
    if len(values) > 64:
        step = (len(values) - 1) / 63
        sampled = [values[round(index * step)] for index in range(64)]
    sigma_max = _optional_number(value.get("sigma_max"))
    sigma_min = _optional_number(value.get("sigma_min"))
    if sigma_max is None and values:
        sigma_max = values[0]
    if sigma_min is None and values:
        sigma_min = values[-1]
    return {
        "kind": "sigmas",
        "sigmas": sampled,
        "steps": int(value.get("steps", max(len(values) - 1, 0))),
        "scheduler": str(value.get("scheduler", "")),
        "denoise": _optional_number(value.get("denoise")) or 1.0,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
    }


def _sampler_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "sampler", "name": str(value)}
    return {
        "kind": "sampler",
        "name": str(value.get("name", "")),
        "custom": callable(value.get("sample")),
    }


def _noise_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "noise", "noise_type": str(value)}
    return {
        "kind": "noise",
        "noise_type": str(value.get("noise_type", "")),
        "noise_seed": value.get("noise_seed"),
        "alpha": value.get("alpha", 1.0),
        "k": value.get("k", 1.0),
        "scale": value.get("scale", 0.1),
    }


def _guider_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "guider", "cfg": None, "model": "", "arch": ""}
    model = value.get("model")
    positive = value.get("positive")
    negative = value.get("negative")
    return {
        "kind": "guider",
        "cfg": value.get("cfg"),
        "arch": str(model.get("arch", "")) if isinstance(model, dict) else "",
        "model": str(model.get("name", "")) if isinstance(model, dict) else "",
        "positive_text": str(positive.get("text", ""))[:200]
        if isinstance(positive, dict)
        else "",
        "negative_text": str(negative.get("text", ""))[:200]
        if isinstance(negative, dict)
        else "",
    }


def _media_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    output_relative = get_output_store().relative_path(value)
    if output_relative:
        output_parts = Path(output_relative).parts
        if len(output_parts) >= 2 and output_parts[0] == "outputs":
            encoded = "/".join(quote(part) for part in output_parts[1:])
            return f"/api/outputs/{encoded}/raw"
    raw = value.strip().replace("\\", "/")
    if raw.startswith("inputs/"):
        input_parts = Path(raw).parts
        if len(input_parts) == 2:
            return f"/api/uploads/{quote(input_parts[1])}/raw"
    inputs_root = get_dataset_store().uploads_dir.resolve()
    root = workspace_root().expanduser().resolve()
    path = Path(value).expanduser()
    try:
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        input_relative = resolved.relative_to(inputs_root)
    except (OSError, ValueError):
        input_relative = None
    if input_relative is not None and len(input_relative.parts) == 1:
        return f"/api/uploads/{quote(input_relative.parts[0])}/raw"
    try:
        relative = (
            path.resolve().relative_to(root)
            if path.is_absolute()
            else (root / path).resolve().relative_to(root)
        )
    except (OSError, ValueError):
        return None
    parts = relative.parts
    if len(parts) == 3 and parts[0] == "data" and parts[1] == "uploads":
        return f"/api/uploads/{quote(parts[2])}/raw"
    if path.is_absolute() and path.suffix.lower() in {
        *IMAGE_FORMATS,
        *VIDEO_FORMATS,
    }:
        return f"/api/artifacts/raw?path={quote(str(path.resolve()))}"
    return None


def _vision_model_preview(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "kind": "vision_model",
            "backend": str(value.get("backend", "")),
            "task": str(value.get("task", "")),
            "model": str(value.get("model", value.get("architecture", ""))),
            "weights": value.get("weights"),
            "device": value.get("device"),
            "classes": _jsonable(value.get("classes") or [])[:12],
            "source": str(value.get("source", "builtin")),
            "model_type": value.get("model_type"),
            "path": value.get("path"),
            "format": value.get("format"),
            "fingerprint": value.get("fingerprint"),
        }
    return {"kind": "vision_model", "model": type(value).__name__}


def _detections_preview(value: Any) -> dict[str, Any]:
    items = value.get("detections", value.get("items", [])) if isinstance(value, dict) else value
    if not isinstance(items, list):
        items = []
    return {
        "kind": "detections",
        "count": len(items),
        "items": _jsonable(items[:20]),
    }


def _masks_preview(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        masks = value.get("masks", [])
        return {
            "kind": "masks",
            "count": len(masks) if isinstance(masks, list) else 0,
            "thumbnail": value.get("thumbnail"),
        }
    return {"kind": "masks", "count": 0, "thumbnail": None}


def _classifications_preview(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.get("top_k", value.get("classifications", []))
    elif isinstance(value, list):
        items = value
    else:
        items = []
    if not isinstance(items, list):
        items = []
    return {
        "kind": "classifications",
        "items": _jsonable(items[:10]),
    }


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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


def _resolve_widget_model_path(value: Any) -> Any:
    """Resolve a file or directory under the repo or model-library root."""
    if not isinstance(value, str) or not value.strip():
        raise FileAccessError("ModelPath widgets require a path.")
    root = Path(__file__).resolve().parents[2]
    models_root = get_model_store().root.resolve()
    raw = value.strip().replace("\\", "/")
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    elif raw.startswith("models/"):
        resolved = (root / path).resolve()
    elif (models_root / path).exists():
        resolved = (models_root / path).resolve()
    else:
        resolved = (root / path).resolve()
    if not resolved.exists() or not (
        resolved.is_relative_to(root) or resolved.is_relative_to(models_root)
    ):
        raise FileAccessError(
            "ModelPath widgets may only access paths inside the repo or model library."
        )
    return str(resolved)


def _resolve_widget_directory(value: Any) -> Any:
    """Resolve a Folder widget under the repo or model-library root."""
    if not isinstance(value, str) or not value.strip():
        raise FileAccessError("Folder widgets require a directory path.")
    root = Path(__file__).resolve().parents[2]
    models_root = get_model_store().root.resolve()
    raw = value.strip().replace("\\", "/")
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    elif raw.startswith("models/"):
        resolved = (root / path).resolve()
    elif (models_root / path).exists():
        resolved = (models_root / path).resolve()
    else:
        resolved = (root / path).resolve()
    if not resolved.is_dir() or not (
        resolved.is_relative_to(root) or resolved.is_relative_to(models_root)
    ):
        raise FileAccessError(
            "Folder widgets may only access directories inside the repo or model library."
        )
    return str(resolved)


def _resolve_widget_image_folder(value: Any) -> str:
    """Resolve a read-only image folder chosen for a node."""
    if not isinstance(value, str) or not value.strip():
        raise FileAccessError("FolderPath widgets require a directory path.")
    root = Path(__file__).resolve().parents[2]
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_dir():
        raise FileAccessError(f"Image folder does not exist: {value}")
    return str(resolved)


def _resolve_widget_file(value: Any) -> Any:
    """Resolve uploads, generated outputs, and legacy workspace files."""
    if not isinstance(value, str) or not value.strip():
        return value
    root = workspace_root().expanduser().resolve()
    inputs_root = get_dataset_store().uploads_dir.resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.is_relative_to(root):
            return str(resolved)
        if resolved.is_relative_to(inputs_root) and resolved.is_file():
            return str(resolved)
        try:
            return str(get_output_store().resolve_path(str(resolved)))
        except (InvalidAssetPath, OSError) as exc:
            raise FileAccessError(
                "File widgets may only access files inside the workspace, inputs, "
                "or generated outputs."
            ) from exc
    raw = value.strip().replace("\\", "/")
    if raw.startswith("inputs/"):
        try:
            return str(get_dataset_store().resolve_path(raw))
        except (InvalidDataset, DatasetNotFound, OSError) as exc:
            raise FileAccessError(
                "File widgets may only access files inside the workspace, inputs, "
                "or generated outputs."
            ) from exc
    if raw.startswith("outputs/"):
        try:
            return str(get_output_store().resolve_path(raw))
        except (InvalidAssetPath, OSError) as exc:
            raise FileAccessError(
                "File widgets may only access files inside the workspace, inputs, "
                "or generated outputs."
            ) from exc
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise FileAccessError(
            "File widgets may only access files inside the workspace, inputs, "
            "or generated outputs."
        )
    return str(resolved)


def _resolve_output_file(value: Any) -> str:
    """Resolve an ``OutputFile`` widget under the generated-output directory."""
    if not isinstance(value, str) or not value.strip():
        raise FileAccessError("OutputFile widgets require a file name.")
    try:
        return str(get_output_store().resolve_path(value))
    except (InvalidAssetPath, OSError) as exc:
        raise FileAccessError(str(exc)) from exc


def _resolve_model_selection(model_type: str | None, value: Any) -> str:
    """Resolve an empty or repo-relative ``ModelSelector`` value."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        return str(get_model_store().resolve(model_type or "", value))
    except ModelError as exc:
        raise ModelAccessError(str(exc)) from exc


def _node_state_hash(
    code: str,
    widget_values: dict[str, Any],
    upstream_hashes: list[str],
    widget_fingerprints: dict[str, str] | None = None,
) -> str:
    payload = (
        f"{code}{json.dumps(widget_values, sort_keys=True, default=str)}"
        f"{json.dumps(widget_fingerprints or {}, sort_keys=True)}"
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


def _file_fingerprint(path: str) -> str:
    try:
        stat = Path(path).stat()
    except OSError:
        return "missing"
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _folder_fingerprint(path: str, *, media_only: bool = False) -> str:
    root = Path(path)
    try:
        stat = root.stat()
        entries = sorted(root.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return "missing"
    fingerprints: list[str] = []
    file_count = 0
    for entry in entries:
        if entry.is_symlink():
            continue
        if entry.is_file():
            if media_only and entry.suffix.lower() not in {
                *IMAGE_FORMATS,
                *VIDEO_FORMATS,
            }:
                continue
            file_count += 1
            if len(fingerprints) < MAX_FOLDER_FINGERPRINT_FILES:
                try:
                    item_stat = entry.stat()
                except OSError:
                    continue
                fingerprints.append(
                    f"{entry.name}:{item_stat.st_size}:{item_stat.st_mtime_ns}"
                )
        elif entry.is_dir() and len(fingerprints) < MAX_FOLDER_FINGERPRINT_FILES:
            fingerprints.append(f"{entry.name}/")
    return (
        f"{stat.st_mtime_ns}:{file_count}:"
        + "|".join(fingerprints[:MAX_FOLDER_FINGERPRINT_FILES])
    )


def _widget_fingerprints(
    node: GraphNode,
    schema: NodeSchema,
    values: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return file, folder and model fingerprints for cache invalidation."""
    fingerprints: dict[str, str] = {}
    resolved_values: dict[str, Any] = (
        node.widget_values if values is None else values
    )
    for widget in schema.widgets:
        kind = widget.data_type.strip().lower()
        value = resolved_values.get(widget.name, widget.default)
        if kind == "modelselector":
            if not isinstance(value, str) or not value.strip():
                fingerprints[widget.name] = "builtin"
                continue
            try:
                summary = get_model_store().describe(widget.model_type or "", value)
            except ModelError:
                fingerprints[widget.name] = f"missing:{value}"
            else:
                fingerprints[widget.name] = summary.fingerprint
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            if kind in {"file", "imagefile", "videofile"}:
                resolved = _resolve_widget_file(value)
                if isinstance(resolved, str):
                    fingerprints[widget.name] = _file_fingerprint(resolved)
            elif kind == "folderpath":
                resolved = _resolve_widget_image_folder(value)
                fingerprints[widget.name] = _folder_fingerprint(
                    resolved, media_only=True
                )
            elif kind in {"directory", "folder"}:
                resolved = _resolve_widget_directory(value)
                if isinstance(resolved, str):
                    fingerprints[widget.name] = _folder_fingerprint(resolved)
        except (FileAccessError, ModelAccessError):
            fingerprints[widget.name] = "missing"
    return fingerprints


def _batch_preview(value: Any) -> dict[str, Any]:
    count = len(value) if isinstance(value, (list, tuple)) else 1
    return {"kind": "batch", "count": count}


def _image_batch_or_batch_preview(value: Any) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return _batch_preview(value)
    if not value or all(isinstance(item, Image.Image) for item in value):
        return _image_batch_preview(value)
    return _batch_preview(value)


def _bounded_jsonable(value: Any) -> Any:
    try:
        converted = _jsonable(value)
        encoded = json.dumps(converted, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)[:200]
    if len(encoded) > MAX_ANY_PREVIEW_CHARS:
        return repr(value)[:200]
    return converted


def _any_preview(value: Any) -> Any:
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None and isinstance(value, Image.Image):
        return _image_preview(value)
    if Image is not None and isinstance(value, (list, tuple)) and value and all(
        isinstance(item, Image.Image) for item in value
    ):
        return _image_batch_preview(value)
    if isinstance(value, dict):
        if str(value.get("media_type", "")).lower() == "video":
            return _video_preview(value)
        return _bounded_jsonable(value)
    if isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and hasattr(value[0], "shape")
            and hasattr(value[1], "shape")
        ):
            features, labels = value
            return _jsonable(
                {
                    "X_shape": tuple(features.shape),
                    "y_shape": tuple(labels.shape),
                    "dtype": str(features.dtype),
                }
            )
        return _batch_preview(value)
    if hasattr(value, "to_dicts") and hasattr(value, "height"):
        return _jsonable(
            {
                "rows": value.height,
                "cols": value.width,
                "head": value.head(5).to_dicts(),
            }
        )
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _jsonable(
            {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "device": str(getattr(value, "device", "cpu")),
                "sample": _tensor_sample(value),
            }
        )
    return _bounded_jsonable(value)
