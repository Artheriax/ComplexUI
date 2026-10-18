"""Pydantic v2 models for node schemas and workflow graphs."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class SocketType(str, Enum):
    """First-class socket data types (classical ML path + tensor path)."""

    DATAFRAME = "DATAFRAME"
    DATASET = "DATASET"
    ML_MODEL = "ML_MODEL"
    METRICS = "METRICS"
    TENSOR = "TENSOR"
    ANY = "ANY"


def normalize_socket_type(raw: str) -> SocketType:
    """Map a raw annotation string onto a known SocketType, defaulting to ANY."""
    cleaned = raw.strip().strip("'\"").upper()
    try:
        return SocketType(cleaned)
    except ValueError:
        return SocketType.ANY


class InputSocket(BaseModel):
    name: str
    data_type: SocketType


class OutputSocket(BaseModel):
    name: str
    data_type: SocketType


class WidgetParam(BaseModel):
    name: str
    data_type: str
    default: Any = None


class SyntaxErrorInfo(BaseModel):
    line: int = 1
    column: int = 0
    message: str


class NodeSchema(BaseModel):
    node_id: str = ""
    inputs: list[InputSocket] = Field(default_factory=list)
    widgets: list[WidgetParam] = Field(default_factory=list)
    outputs: list[OutputSocket] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    syntax_errors: list[SyntaxErrorInfo] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    code: str
    widget_values: dict[str, Any] = Field(default_factory=dict)
    is_trusted: bool = True


class GraphEdge(BaseModel):
    source_node_id: str
    source_handle: str
    target_node_id: str
    target_handle: str


class WorkflowGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class EventType(str, Enum):
    """Telemetry events streamed during workflow execution."""

    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    NODE_STARTED = "NODE_STARTED"
    NODE_COMPLETED = "NODE_COMPLETED"
    NODE_ERROR = "NODE_ERROR"
    INSTALLING_DEPS = "INSTALLING_DEPS"


class ExecutionEvent(BaseModel):
    type: EventType
    node_id: str
    data: dict[str, Any] = Field(default_factory=dict)


class ParseRequest(BaseModel):
    code: str
    node_id: str = ""


class ExecuteRequest(BaseModel):
    graph: WorkflowGraph
    targets: list[str] | None = None


class ExecuteResponse(BaseModel):
    status: str
    run_id: str


class ApproveRequest(BaseModel):
    node_id: str
    code_hash: str

    @field_validator("code_hash")
    @classmethod
    def _validate_code_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SHA256_HEX.fullmatch(normalized):
            raise ValueError(
                "code_hash must be a 64-character hex sha256 digest"
            )
        return normalized


class ApproveResponse(BaseModel):
    status: str
    node_id: str
    code_hash: str


class TemplateInfo(BaseModel):
    name: str
    title: str
    code: str
    code_hash: str
    node_schema: NodeSchema
