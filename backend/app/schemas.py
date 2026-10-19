"""Pydantic v2 models for node schemas and workflow graphs."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

MAX_NODE_CODE_CHARS = 200_000
MAX_NODE_IDENTIFIER_CHARS = 200
MAX_GRAPH_NODES = 200
MAX_GRAPH_EDGES = 1_000
MAX_EXECUTE_TARGETS = 200


class SocketType(StrEnum):
    """First-class socket data types (classical ML path + tensor path)."""

    DATAFRAME = "DATAFRAME"
    DATASET = "DATASET"
    ML_MODEL = "ML_MODEL"
    METRICS = "METRICS"
    TENSOR = "TENSOR"
    ANY = "ANY"


class TextOutput(BaseModel):
    """Console text or a text representation emitted by a node."""

    kind: Literal["text"] = "text"
    source: str = "display"
    text: str
    truncated: bool = False


class TableOutput(BaseModel):
    """A dataframe-like value rendered as a table preview."""

    kind: Literal["table"] = "table"
    source: str = "display"
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    total_rows: int = 0
    total_cols: int = 0
    truncated: bool = False


class ImageOutput(BaseModel):
    """A base64-encoded bitmap or SVG image."""

    kind: Literal["image"] = "image"
    source: str = "display"
    mime: str = "image/png"
    data: str
    alt: str = "Plot output"


class HtmlOutput(BaseModel):
    """HTML from a Jupyter-style rich repr, sanitized by the client before rendering."""

    kind: Literal["html"] = "html"
    source: str = "display"
    html: str
    truncated: bool = False


class PlotlyOutput(BaseModel):
    """A serialized Plotly figure (data/layout/frames)."""

    kind: Literal["plotly"] = "plotly"
    source: str = "display"
    figure: dict[str, Any] = Field(default_factory=dict)


RichOutput = Annotated[
    TextOutput | TableOutput | ImageOutput | HtmlOutput | PlotlyOutput,
    Field(discriminator="kind"),
]


def serialize_rich_outputs(outputs: list[RichOutput]) -> list[dict[str, Any]]:
    """Return JSON-ready dictionaries for execution telemetry."""
    return [output.model_dump(mode="json") for output in outputs]



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
    id: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
    code: str = Field(max_length=MAX_NODE_CODE_CHARS)
    widget_values: dict[str, Any] = Field(default_factory=dict)
    is_trusted: bool = True


class GraphEdge(BaseModel):
    source_node_id: str = Field(
        min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS
    )
    source_handle: str = Field(
        min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS
    )
    target_node_id: str = Field(
        min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS
    )
    target_handle: str = Field(
        min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS
    )


class WorkflowGraph(BaseModel):
    nodes: list[GraphNode] = Field(
        default_factory=list, max_length=MAX_GRAPH_NODES
    )
    edges: list[GraphEdge] = Field(
        default_factory=list, max_length=MAX_GRAPH_EDGES
    )


class EventType(StrEnum):
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
    code: str = Field(max_length=MAX_NODE_CODE_CHARS)
    node_id: str = Field(default="", max_length=MAX_NODE_IDENTIFIER_CHARS)


class ExecuteRequest(BaseModel):
    graph: WorkflowGraph
    targets: list[str] | None = Field(
        default=None, max_length=MAX_EXECUTE_TARGETS
    )


class ExecuteResponse(BaseModel):
    status: str
    run_id: str


class ExecutionStatusResponse(BaseModel):
    status: str
    run_id: str | None = None


class ApproveRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
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
    category: str
    description: str
    hidden: bool = False
    code: str
    code_hash: str
    node_schema: NodeSchema
