"""Export workflows as standalone Python scripts or Jupyter notebooks.

Exports reuse the same AST transpiler and topological planner as the executor,
so generated artifacts preserve node code, widget values, wiring and PEP 723
dependencies. Node imports are localized inside their generated functions so
two nodes that alias the same module name cannot clobber each other.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from .executor import GraphPlan, validate_graph
from .models import ModelError, get_model_store
from .schemas import GraphEdge, GraphNode, NodeSchema, WorkflowGraph
from .transpiler import transpile_node

__all__ = [
    "ExportError",
    "ExportedFile",
    "export_notebook",
    "export_python",
]

PYTHON_MEDIA_TYPE = "text/x-python"
NOTEBOOK_MEDIA_TYPE = "application/x-ipynb+json"

_PEP723_BLOCK = re.compile(
    r"^[ \t]*# /// script[ \t]*\n(?:[^\n]*\n)*?[ \t]*# ///[ \t]*(?:\n|$)",
    re.MULTILINE,
)


class ExportError(Exception):
    """Raised when a workflow cannot be converted into a portable artifact."""


@dataclass(frozen=True, slots=True)
class ExportedFile:
    """One downloadable export artifact."""

    filename: str
    content: str
    media_type: str


@dataclass(frozen=True, slots=True)
class _PreparedNode:
    node_id: str
    function_name: str
    function_source: str
    call_args: tuple[str, ...]
    outputs: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _PreparedExport:
    nodes: tuple[_PreparedNode, ...]
    dependencies: tuple[str, ...]
    future_imports: tuple[str, ...]


def export_python(
    graph: WorkflowGraph,
    *,
    targets: Sequence[str] | None = None,
    name: str = "workflow",
    generated_at: datetime | None = None,
) -> ExportedFile:
    """Render a runnable PEP 723 Python script for the selected graph scope."""
    prepared = _prepare(graph, targets)
    stamp = _timestamp(generated_at)
    lines = _python_header(
        prepared.dependencies, generated_at=generated_at
    )
    if prepared.future_imports:
        lines.extend(prepared.future_imports)
        lines.append("")
    for index, node in enumerate(prepared.nodes):
        if index > 0:
            lines.append("")
        lines.extend(node.function_source.splitlines())
    lines.extend(
        [
            "",
            "def main() -> dict[str, dict[str, object]]:",
            "    node_outputs: dict[str, dict[str, object]] = {}",
        ]
    )
    for node in prepared.nodes:
        lines.append("")
        lines.append(f"    # Node: {node.node_id}")
        lines.extend(_call_lines(node, "    "))
    lines.extend(
        [
            "",
            (
                "    print("
                + repr(f"ComplexUI workflow completed: {len(prepared.nodes)} node(s).")
                + ")"
            ),
            "    return node_outputs",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
    )
    return ExportedFile(
        filename=_filename(name, "py", stamp),
        content="\n".join(lines) + "\n",
        media_type=PYTHON_MEDIA_TYPE,
    )


def export_notebook(
    graph: WorkflowGraph,
    *,
    targets: Sequence[str] | None = None,
    name: str = "workflow",
    generated_at: datetime | None = None,
) -> ExportedFile:
    """Render a nbformat 4.5 notebook with one cell per workflow node."""
    prepared = _prepare(graph, targets)
    stamp = _timestamp(generated_at)
    cells: list[dict[str, object]] = [
        _markdown_cell(
            "intro",
            _notebook_intro(prepared, generated_at=generated_at),
        ),
        _code_cell("setup", _notebook_setup(prepared)),
    ]
    for index, node in enumerate(prepared.nodes):
        cells.append(
            _code_cell(
                f"node-{index}",
                "\n".join(
                    [f"# Node: {node.node_id}", *_call_lines(node, "")]
                ),
            )
        )
    cells.append(
        _code_cell(
            "outputs",
            "# Every node's outputs are available in this dictionary.\n"
            "node_outputs",
        )
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return ExportedFile(
        filename=_filename(name, "ipynb", stamp),
        content=json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        media_type=NOTEBOOK_MEDIA_TYPE,
    )


def _prepare(
    graph: WorkflowGraph, targets: Sequence[str] | None
) -> _PreparedExport:
    if not graph.nodes:
        raise ExportError("Workflow has no nodes to export")
    plan = validate_graph(graph, targets=targets)
    if not plan.order:
        raise ExportError("Workflow has no nodes to export")

    nodes_by_id = {node.id: node for node in graph.nodes}
    incoming = _incoming_edges(graph.edges, plan)
    dependencies: list[str] = []
    seen_dependencies: set[str] = set()
    future_imports: list[str] = []
    seen_future_imports: set[str] = set()
    function_names: set[str] = set()
    prepared_nodes: list[_PreparedNode] = []

    for node_id in plan.order:
        node = nodes_by_id[node_id]
        schema = plan.schemas[node_id]
        transpiled = transpile_node(node.code, node_id)
        if transpiled.func_name in function_names:
            raise ExportError(
                f"Node ids '{node_id}' and another node generate the same "
                f"function name '{transpiled.func_name}'"
            )
        function_names.add(transpiled.func_name)
        node_future, function_source = _localize_imports(
            transpiled.source, transpiled.func_name
        )
        for line in node_future:
            if line not in seen_future_imports:
                seen_future_imports.add(line)
                future_imports.append(line)
        for dependency in schema.dependencies:
            if dependency not in seen_dependencies:
                seen_dependencies.add(dependency)
                dependencies.append(dependency)
        prepared_nodes.append(
            _PreparedNode(
                node_id=node_id,
                function_name=transpiled.func_name,
                function_source=function_source,
                call_args=_call_args(node, schema, incoming.get(node_id, {})),
                outputs=tuple(
                    (socket.name, index)
                    for index, socket in enumerate(schema.outputs)
                ),
            )
        )

    return _PreparedExport(
        nodes=tuple(prepared_nodes),
        dependencies=tuple(sorted(dependencies)),
        future_imports=tuple(future_imports),
    )


def _incoming_edges(
    edges: Sequence[GraphEdge], plan: GraphPlan
) -> dict[str, dict[str, GraphEdge]]:
    incoming: dict[str, dict[str, GraphEdge]] = {
        node_id: {} for node_id in plan.order
    }
    for edge in edges:
        bucket = incoming.get(edge.target_node_id)
        if bucket is not None:
            bucket[edge.target_handle] = edge
    return incoming


def _call_args(
    node: GraphNode,
    schema: NodeSchema,
    incoming: dict[str, GraphEdge],
) -> tuple[str, ...]:
    args: list[str] = []
    for socket in schema.inputs:
        edge = incoming[socket.name]
        args.append(f"{socket.name}={_source_expression(edge)}")
    for widget in schema.widgets:
        edge = incoming.get(widget.name)
        if edge is not None:
            args.append(f"{widget.name}={_source_expression(edge)}")
            continue
        value = node.widget_values.get(widget.name, widget.default)
        if widget.data_type.strip().lower() == "modelselector":
            value = _export_model_value(widget.model_type, value)
        args.append(f"{widget.name}={_literal(value)}")
    return tuple(args)


def _export_model_value(model_type: str | None, value: object) -> object:
    """Export local models as absolute paths, since the root is outside the workspace."""
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        return str(get_model_store().resolve(model_type or "", value))
    except ModelError as exc:
        raise ExportError(str(exc)) from exc


def _source_expression(edge: GraphEdge) -> str:
    return (
        f"node_outputs[{edge.source_node_id!r}]"
        f"[{edge.source_handle!r}]"
    )


def _literal(value: object) -> str:
    return repr(value)


def _localize_imports(
    source: str, function_name: str
) -> tuple[tuple[str, ...], str]:
    """Move non-future imports inside the generated function.

    Import aliases at module scope would be shared by every node; localizing
    them preserves the executor's per-node namespace semantics. Future imports
    must stay at module scope, so they are returned separately.
    """
    lines = _PEP723_BLOCK.sub("", source).split("\n")
    marker = f"def {function_name}("
    try:
        def_index = next(
            index for index, line in enumerate(lines) if line.startswith(marker)
        )
    except StopIteration as exc:
        raise ExportError(
            f"Transpiled node function '{function_name}' could not be located"
        ) from exc

    future: list[str] = []
    local_imports: list[str] = []
    for line in lines[:def_index]:
        if line.lstrip().startswith("from __future__ import"):
            future.append(line.strip())
        else:
            local_imports.append(line)

    body = lines[def_index + 1 :]
    while body and not body[0].strip():
        body.pop(0)
    function_lines = [lines[def_index]]
    if local_imports:
        function_lines.extend(
            f"    {line}" if line.strip() else "" for line in local_imports
        )
        if body:
            function_lines.append("")
    function_lines.extend(body)
    return tuple(future), "\n".join(function_lines)


def _python_header(
    dependencies: tuple[str, ...], *, generated_at: datetime | None
) -> list[str]:
    lines = [
        "# /// script",
        '# requires-python = ">=3.12"',
    ]
    if dependencies:
        lines.append("# dependencies = [")
        for dependency in dependencies:
            lines.append(f"#   {json.dumps(dependency)},")
        lines.append("# ]")
    lines.extend(
        [
            "# ///",
            "",
            f"# Generated by ComplexUI on {_human_timestamp(generated_at)}",
            f"# {len(dependencies)} declared dependency package(s).",
            "# Run with: uv run <this-file>.py",
            "",
        ]
    )
    return lines


def _notebook_intro(
    prepared: _PreparedExport, *, generated_at: datetime | None
) -> str:
    lines = [
        "# ComplexUI workflow",
        "",
        f"Generated on {_human_timestamp(generated_at)} with "
        f"{len(prepared.nodes)} node(s).",
        "",
    ]
    if prepared.dependencies:
        lines.append("Install dependencies with `uv add`:")
        lines.append("")
        lines.extend(f"- `{dependency}`" for dependency in prepared.dependencies)
    else:
        lines.append("This workflow declares no third-party dependencies.")
    lines.extend(
        [
            "",
            "Run the cells in order. Every node's outputs are collected in "
            "the `node_outputs` dictionary.",
        ]
    )
    return "\n".join(lines)


def _notebook_setup(prepared: _PreparedExport) -> str:
    lines = ["# Generated node functions and imports."]
    if prepared.future_imports:
        lines.extend(prepared.future_imports)
        lines.append("")
    for index, node in enumerate(prepared.nodes):
        if index > 0:
            lines.append("")
        lines.extend(node.function_source.splitlines())
    lines.extend(["", "node_outputs = {}"])
    return "\n".join(lines)


def _call_lines(node: _PreparedNode, indent: str) -> list[str]:
    if node.call_args:
        lines = [f"{indent}__node_result = {node.function_name}("]
        lines.extend(f"{indent}    {argument}," for argument in node.call_args)
        lines.append(f"{indent})")
    else:
        lines = [f"{indent}__node_result = {node.function_name}()"]
    if node.outputs:
        lines.append(f"{indent}node_outputs[{node.node_id!r}] = {{")
        lines.extend(
            f"{indent}    {name!r}: __node_result[{index}],"
            for name, index in node.outputs
        )
        lines.append(f"{indent}}}")
    return lines


def _markdown_cell(cell_id: str, text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _source_lines(text),
    }


def _code_cell(cell_id: str, text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
    }


def _source_lines(text: str) -> list[str]:
    if not text:
        return []
    if text.endswith("\n"):
        return text.splitlines(keepends=True)
    lines = text.splitlines(keepends=True)
    lines[-1] = f"{lines[-1]}\n"
    return lines


def _timestamp(moment: datetime | None) -> str:
    return _utc_moment(moment).strftime("%Y%m%d-%H%M%S")


def _human_timestamp(moment: datetime | None) -> str:
    return _utc_moment(moment).strftime("%Y-%m-%d %H:%M:%S UTC")


def _utc_moment(moment: datetime | None) -> datetime:
    value = moment if moment is not None else datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _filename(name: str, extension: str, stamp: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_-]+", "-", name).strip("-").lower()
    return f"{slug or 'complexui-workflow'}-{stamp}.{extension}"
