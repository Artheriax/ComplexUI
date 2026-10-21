"""Example workflow catalog: JSON definitions, validation and materialisation.

Each workflow references official templates by name and stores widget values
plus node positions. ``validate_workflow`` enforces socket-type compatibility
and DAG shape at load time; ``materialize_workflow`` turns a definition into an
executable :class:`WorkflowGraph`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .schemas import (
    GraphEdge,
    GraphNode,
    NodeSchema,
    WorkflowGraph,
    socket_types_compatible,
    widget_choice_allowed,
    widget_socket_type,
)
from .security import get_security_registry
from .transpiler import parse_node

__all__ = [
    "WORKFLOW_CATEGORIES",
    "WorkflowCatalogError",
    "WorkflowEdgeSpec",
    "WorkflowNodeSpec",
    "WorkflowTemplateInfo",
    "get_workflows",
    "load_workflows",
    "materialize_workflow",
    "validate_workflow",
]

WORKFLOW_CATEGORIES: tuple[str, ...] = (
    "Getting Started",
    "Classical ML",
    "Data Quality & EDA",
    "PyTorch",
    "Computer Vision",
)

_WORKFLOW_DIR = Path(__file__).resolve().parent / "workflows"

_WIDGET_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "int": (int,),
    "float": (int, float),
    "str": (str,),
    "bool": (bool,),
}


class WorkflowCatalogError(RuntimeError):
    """Raised when a workflow definition is structurally invalid."""


class WorkflowNodeSpec(BaseModel):
    id: str
    template: str
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    widget_values: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdgeSpec(BaseModel):
    source_node_id: str
    source_handle: str
    target_node_id: str
    target_handle: str


class WorkflowTemplateInfo(BaseModel):
    name: str
    title: str
    category: str
    description: str
    nodes: list[WorkflowNodeSpec]
    edges: list[WorkflowEdgeSpec]


def load_workflows(directory: Path | None = None) -> dict[str, WorkflowTemplateInfo]:
    """Load and validate every ``*.json`` file in ``directory``."""
    base = directory or _WORKFLOW_DIR
    if not base.is_dir():
        raise WorkflowCatalogError(f"Workflow directory not found: {base}")

    workflows: dict[str, WorkflowTemplateInfo] = {}
    for path in sorted(base.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkflowCatalogError(f"{path.name}: invalid JSON ({exc})") from exc
        try:
            info = WorkflowTemplateInfo.model_validate(payload)
        except ValidationError as exc:
            raise WorkflowCatalogError(f"{path.name}: {exc}") from exc
        if info.name != path.stem:
            raise WorkflowCatalogError(
                f"{path.name}: name '{info.name}' does not match file stem"
            )
        if info.name in workflows:
            raise WorkflowCatalogError(f"{path.name}: duplicate workflow name")
        validate_workflow(info)
        workflows[info.name] = info
    return workflows


@lru_cache(maxsize=1)
def get_workflows() -> dict[str, WorkflowTemplateInfo]:
    """Return the bundled workflow catalog (validated, cached)."""
    return load_workflows()


def validate_workflow(info: WorkflowTemplateInfo) -> None:
    """Raise :class:`WorkflowCatalogError` when the definition is invalid."""
    label = f"workflow '{info.name}'"
    if info.category not in WORKFLOW_CATEGORIES:
        raise WorkflowCatalogError(f"{label}: unknown category '{info.category}'")
    if not info.nodes:
        raise WorkflowCatalogError(f"{label}: no nodes")

    registry = get_security_registry()
    known = {record.name for record in registry.templates()}
    schemas: dict[str, NodeSchema] = {}
    seen: set[str] = set()
    for node in info.nodes:
        if node.id in seen:
            raise WorkflowCatalogError(f"{label}: duplicate node id '{node.id}'")
        seen.add(node.id)
        if node.template not in known:
            raise WorkflowCatalogError(
                f"{label}: unknown template '{node.template}' on node '{node.id}'"
            )
        schema = parse_node(registry.get_template(node.template).code, node.id)
        schemas[node.id] = schema
        widget_types = {widget.name: widget.data_type for widget in schema.widgets}
        widget_options = {
            widget.name: widget.options for widget in schema.widgets
        }
        for key, value in node.widget_values.items():
            if key not in widget_types:
                raise WorkflowCatalogError(
                    f"{label}: node '{node.id}' sets unknown widget '{key}'"
                )
            allowed = _WIDGET_PYTHON_TYPES.get(widget_types[key])
            if allowed is not None:
                if isinstance(value, bool) and widget_types[key] != "bool":
                    raise WorkflowCatalogError(
                        f"{label}: widget '{key}' on '{node.id}' expects "
                        f"{widget_types[key]}, got bool"
                    )
                if not isinstance(value, allowed):
                    raise WorkflowCatalogError(
                        f"{label}: widget '{key}' on '{node.id}' expects "
                        f"{widget_types[key]}, got {type(value).__name__}"
                    )
            options = widget_options[key]
            if options and not widget_choice_allowed(value, options):
                raise WorkflowCatalogError(
                    f"{label}: widget '{key}' on '{node.id}' must be one of "
                    f"{options!r}"
                )

    occupancy: set[tuple[str, str]] = set()
    for edge in info.edges:
        for node_id in (edge.source_node_id, edge.target_node_id):
            if node_id not in schemas:
                raise WorkflowCatalogError(
                    f"{label}: edge references unknown node '{node_id}'"
                )
        outputs = {socket.name: socket.data_type for socket in schemas[edge.source_node_id].outputs}
        inputs = {socket.name: socket.data_type for socket in schemas[edge.target_node_id].inputs}
        widgets = {
            widget.name: widget_socket_type(widget.data_type)
            for widget in schemas[edge.target_node_id].widgets
        }
        if edge.source_handle not in outputs:
            raise WorkflowCatalogError(
                f"{label}: unknown output '{edge.source_handle}' on "
                f"'{edge.source_node_id}'"
            )
        if edge.target_handle in inputs:
            target_type = inputs[edge.target_handle]
        elif edge.target_handle in widgets:
            target_type = widgets[edge.target_handle]
        else:
            raise WorkflowCatalogError(
                f"{label}: unknown input or widget '{edge.target_handle}' on "
                f"'{edge.target_node_id}'"
            )
        source_type = outputs[edge.source_handle]
        if not socket_types_compatible(source_type, target_type):
            raise WorkflowCatalogError(
                f"{label}: type mismatch {source_type.value} -> "
                f"{target_type.value} for {edge.source_node_id}.{edge.source_handle}"
            )
        key = (edge.target_node_id, edge.target_handle)
        if key in occupancy:
            raise WorkflowCatalogError(
                f"{label}: parameter '{edge.target_handle}' of "
                f"'{edge.target_node_id}' is wired twice"
            )
        occupancy.add(key)

    _assert_acyclic(label, info)


def _assert_acyclic(label: str, info: WorkflowTemplateInfo) -> None:
    incoming: dict[str, int] = {node.id: 0 for node in info.nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in info.nodes}
    for edge in info.edges:
        incoming[edge.target_node_id] += 1
        outgoing[edge.source_node_id].append(edge.target_node_id)
    queue = [node_id for node_id, count in incoming.items() if count == 0]
    visited = 0
    while queue:
        node_id = queue.pop()
        visited += 1
        for target in outgoing[node_id]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    if visited != len(incoming):
        raise WorkflowCatalogError(f"{label}: graph contains a cycle")


def materialize_workflow(
    workflow: WorkflowTemplateInfo,
    templates: Mapping[str, str] | None = None,
) -> WorkflowGraph:
    """Build an executable :class:`WorkflowGraph` from a definition."""
    if templates is None:
        templates = {
            record.name: record.code for record in get_security_registry().templates()
        }
    nodes = [
        GraphNode(
            id=node.id,
            code=templates[node.template],
            widget_values=dict(node.widget_values),
        )
        for node in workflow.nodes
    ]
    edges = [
        GraphEdge(
            source_node_id=edge.source_node_id,
            source_handle=edge.source_handle,
            target_node_id=edge.target_node_id,
            target_handle=edge.target_handle,
        )
        for edge in workflow.edges
    ]
    return WorkflowGraph(nodes=nodes, edges=edges)
