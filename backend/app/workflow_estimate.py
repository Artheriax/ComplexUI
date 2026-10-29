"""Workflow-level model memory estimates.

Model references are discovered from the typed ``ModelSelector`` widgets of
each graph node, so the estimator does not execute node code or load weights.
Invalid selections are reported per reference instead of failing the whole
estimate.
"""

from __future__ import annotations

from .memory_estimate import estimate_model_summary_memory
from .models import ModelError, ModelStore, get_model_store
from .schemas import (
    MAX_GRAPH_NODES,
    WorkflowEstimateRequest,
    WorkflowGraph,
    WorkflowMemoryEstimate,
    WorkflowModelEstimate,
)
from .transpiler import parse_node

__all__ = [
    "estimate_workflow",
    "estimate_workflow_request",
]

_MAX_WARNINGS = 16
_MAX_ERROR_CHARS = 240


def _error_text(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:_MAX_ERROR_CHARS]


def estimate_workflow_request(
    payload: WorkflowEstimateRequest,
    store: ModelStore | None = None,
) -> WorkflowMemoryEstimate:
    """Estimate every model reference in a workflow request."""
    return estimate_workflow(payload.workflow, store)


def estimate_workflow(
    graph: WorkflowGraph,
    store: ModelStore | None = None,
) -> WorkflowMemoryEstimate:
    """Estimate distinct model weights and resident/offloaded memory."""
    store = store or get_model_store()
    models: list[WorkflowModelEstimate] = []
    warnings: list[str] = []
    seen: set[str] = set()
    weights = 0
    resident = 0
    offload = 0
    unknown = False
    for node in graph.nodes:
        schema = parse_node(node.code, node.id)
        if schema.syntax_errors:
            if len(warnings) < _MAX_WARNINGS:
                warnings.append(
                    f"{node.id}: node code has syntax errors; model selection skipped"
                )
            continue
        for widget in schema.widgets:
            if widget.data_type != "ModelSelector":
                continue
            raw = node.widget_values.get(widget.name)
            value = raw.strip() if isinstance(raw, str) else ""
            if not value:
                if widget.model_required and len(warnings) < _MAX_WARNINGS:
                    warnings.append(
                        f"{node.id}: required model '{widget.name}' is not selected"
                    )
                continue
            model_type = widget.model_type or ""
            try:
                summary = store.describe(
                    model_type,
                    value,
                    select=widget.model_select,
                    include=widget.model_include,
                )
            except (ModelError, OSError, ValueError) as exc:
                unknown = True
                models.append(
                    WorkflowModelEstimate(
                        node_id=node.id,
                        widget=widget.name,
                        value=value,
                        model_type=model_type or None,
                        error=_error_text(exc),
                    )
                )
                continue
            memory = summary.memory or estimate_model_summary_memory(summary)
            models.append(
                WorkflowModelEstimate(
                    node_id=node.id,
                    widget=widget.name,
                    value=value,
                    model_id=summary.id,
                    model_type=summary.model_type,
                    memory=memory,
                )
            )
            if summary.id in seen:
                continue
            seen.add(summary.id)
            weights += memory.weights_bytes
            resident += memory.resident_estimate_bytes
            offload += memory.offload_ram_estimate_bytes
            unknown = unknown or memory.unknown
    if len(models) > MAX_GRAPH_NODES:
        models = models[:MAX_GRAPH_NODES]
        if len(warnings) < _MAX_WARNINGS:
            warnings.append("model list truncated to the workflow node limit")
    return WorkflowMemoryEstimate(
        models=models,
        weights_bytes=weights,
        resident_estimate_bytes=resident,
        offload_ram_estimate_bytes=offload,
        unknown=unknown,
        warnings=warnings[:_MAX_WARNINGS],
    )
