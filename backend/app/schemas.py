"""Pydantic v2 models for node schemas and workflow graphs."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

MAX_NODE_CODE_CHARS = 200_000
MAX_NODE_IDENTIFIER_CHARS = 200
MAX_GRAPH_NODES = 200
MAX_GRAPH_EDGES = 1_000
MAX_SUBGRAPHS = 50
MAX_EXECUTE_TARGETS = 200
SUBGRAPH_INPUT_NODE_ID = "__subgraph_input__"
SUBGRAPH_OUTPUT_NODE_ID = "__subgraph_output__"
MAX_PROJECT_NAME_CHARS = 80
MAX_PROJECT_SLUG_CHARS = 80
MAX_FRAME_DIMENSION = 100_000
PROJECT_SCHEMA_VERSION = 1
PROJECT_VERSION_RETENTION = 30
MAX_DATASET_NAME_CHARS = 160
MAX_DATASET_BYTES = 100 * 1024 * 1024
MAX_MEDIA_BYTES = 500 * 1024 * 1024
MAX_DATASET_PREVIEW_ROWS = 100
DEFAULT_DATASET_PREVIEW_ROWS = 20
RUN_RECORD_SCHEMA_VERSION = 1
RUN_HISTORY_RETENTION = 50
MAX_RUN_RICH_OUTPUT_BYTES = 5 * 1024 * 1024


class SocketType(StrEnum):
    """First-class socket data types (classical ML, tensor and vision paths)."""

    DATAFRAME = "DATAFRAME"
    DATASET = "DATASET"
    ML_MODEL = "ML_MODEL"
    METRICS = "METRICS"
    TENSOR = "TENSOR"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    MEDIA = "MEDIA"
    DETECTION_MODEL = "DETECTION_MODEL"
    SEGMENTATION_MODEL = "SEGMENTATION_MODEL"
    CLASSIFICATION_MODEL = "CLASSIFICATION_MODEL"
    DETECTIONS = "DETECTIONS"
    MASKS = "MASKS"
    CLASSIFICATIONS = "CLASSIFICATIONS"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    STRING = "STRING"
    ANY = "ANY"


class AssetKind(StrEnum):
    """Kinds of workspace assets stored under ``data/uploads``."""

    TABLE = "table"
    IMAGE = "image"
    VIDEO = "video"


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


def widget_socket_type(raw: str) -> SocketType:
    """Map a widget's Python-ish type label onto its connectable socket type."""
    cleaned = raw.strip().strip("'\"").lower()
    if cleaned == "int":
        return SocketType.INT
    if cleaned == "float":
        return SocketType.FLOAT
    if cleaned == "bool":
        return SocketType.BOOL
    if cleaned in {
        "str",
        "string",
        "file",
        "path",
        "imagefile",
        "videofile",
        "outputfile",
        "modelselector",
    }:
        return SocketType.STRING
    return normalize_socket_type(raw)


_MEDIA_TARGETS = {SocketType.IMAGE, SocketType.VIDEO, SocketType.MEDIA}


def socket_types_compatible(source: SocketType, target: SocketType) -> bool:
    """Return whether a source output can feed a target socket/widget."""
    return (
        source is target
        or source is SocketType.ANY
        or target is SocketType.ANY
        or (source is SocketType.INT and target is SocketType.FLOAT)
        or (
            source in _MEDIA_TARGETS
            and target is SocketType.MEDIA
            and source is not SocketType.MEDIA
        )
    )


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
    options: list[Any] = Field(default_factory=list)
    model_type: str | None = None


def widget_choice_allowed(value: Any, options: list[Any]) -> bool:
    """Return whether a value is one of a widget's explicit choices."""
    for option in options:
        if isinstance(value, bool) or isinstance(option, bool):
            if isinstance(value, bool) and isinstance(option, bool):
                if value is option:
                    return True
        elif isinstance(value, (int, float)) and isinstance(option, (int, float)):
            if value == option:
                return True
        elif type(value) is type(option) and value == option:
            return True
    return False


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
    approval_id: str | None = Field(default=None, max_length=MAX_NODE_IDENTIFIER_CHARS)


class GraphEdge(BaseModel):
    source_node_id: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
    source_handle: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
    target_node_id: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
    target_handle: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)


class WorkflowGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list, max_length=MAX_GRAPH_NODES)
    edges: list[GraphEdge] = Field(default_factory=list, max_length=MAX_GRAPH_EDGES)


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
    targets: list[str] | None = Field(default=None, max_length=MAX_EXECUTE_TARGETS)
    node_labels: dict[str, str] | None = Field(default=None, max_length=MAX_GRAPH_NODES)

    @field_validator("node_labels")
    @classmethod
    def _validate_node_labels(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        for node_id, label in value.items():
            if not node_id or len(node_id) > MAX_NODE_IDENTIFIER_CHARS:
                raise ValueError("Node-label keys must be valid node ids")
            if len(label) > MAX_PROJECT_NAME_CHARS:
                raise ValueError(f"Node labels must be at most {MAX_PROJECT_NAME_CHARS} characters")
        return value


class ExecuteResponse(BaseModel):
    status: str
    run_id: str


class RunStatus(StrEnum):
    """Terminal state of a persisted execution run."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunNodeStatus(StrEnum):
    """Per-node state captured in a run record."""

    PENDING = "pending"
    COMPLETED = "completed"
    ERROR = "error"


class RunErrorRecord(BaseModel):
    """Failure details retained with a run or one of its nodes."""

    message: str
    error_type: str
    node_id: str | None = None
    line: int | None = None
    traceback: str = ""


class RunNodeRecord(BaseModel):
    """One node's status and bounded outputs in a persisted run."""

    node_id: str
    label: str
    status: RunNodeStatus = RunNodeStatus.PENDING
    duration_ms: float | None = Field(default=None, ge=0)
    cached: bool = False
    previews: dict[str, Any] = Field(default_factory=dict)
    rich_outputs: list[RichOutput] = Field(default_factory=list)
    rich_outputs_truncated: bool = False
    error: RunErrorRecord | None = None


class RunSummary(BaseModel):
    """List-view metadata for one terminal execution run."""

    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    status: RunStatus
    started_at: str
    finished_at: str
    duration_ms: float = Field(ge=0)
    node_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    cached_count: int = Field(ge=0)
    error_type: str | None = None


class RunDetail(RunSummary):
    """Full run record returned by the detail endpoint."""

    order: list[str] = Field(default_factory=list, max_length=MAX_GRAPH_NODES)
    targets: list[str] | None = Field(default=None, max_length=MAX_EXECUTE_TARGETS)
    node_labels: dict[str, str] = Field(default_factory=dict)
    nodes: list[RunNodeRecord] = Field(default_factory=list, max_length=MAX_GRAPH_NODES)
    error: RunErrorRecord | None = None


class RunRecord(RunDetail):
    """On-disk run document."""

    schema_version: int = RUN_RECORD_SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != RUN_RECORD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported run record version: {value}")
        return value


class RunDeleteResponse(BaseModel):
    status: Literal["deleted"]
    run_id: str


class ExportFormat(StrEnum):
    """Downloadable workflow source formats."""

    PYTHON = "python"
    NOTEBOOK = "notebook"


class ExportRequest(BaseModel):
    graph: WorkflowGraph
    format: ExportFormat = ExportFormat.PYTHON
    targets: list[str] | None = Field(default=None, max_length=MAX_EXECUTE_TARGETS)


class ExportResponse(BaseModel):
    filename: str
    media_type: str
    content: str


class ModelSummary(BaseModel):
    """Metadata for one file in the repo-root local model library."""

    id: str = Field(min_length=1, max_length=MAX_DATASET_NAME_CHARS)
    name: str = Field(min_length=1, max_length=MAX_DATASET_NAME_CHARS)
    path: str
    model_type: str = Field(min_length=1, max_length=64)
    format: str
    size_bytes: int = Field(ge=0)
    modified_at: str
    mtime_ns: int = Field(ge=0)
    fingerprint: str
    executable: bool = False


class ModelTypeSummary(BaseModel):
    """One model type folder under the repo model library."""

    name: str = Field(min_length=1, max_length=64)
    path: str
    count: int = Field(ge=0)
    default: bool = False


class DatasetSummary(BaseModel):
    """Metadata for one asset stored in the workspace upload directory."""

    id: str = Field(min_length=1, max_length=MAX_DATASET_NAME_CHARS)
    name: str = Field(min_length=1, max_length=MAX_DATASET_NAME_CHARS)
    path: str
    format: str
    kind: AssetKind = AssetKind.TABLE
    size_bytes: int = Field(ge=0)
    modified_at: str


class DatasetPreview(BaseModel):
    """A bounded table preview of an uploaded dataset."""

    dataset: DatasetSummary
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    truncated: bool = False
    row_limit: int = Field(ge=1, le=MAX_DATASET_PREVIEW_ROWS)


class DatasetUploadResponse(BaseModel):
    status: Literal["uploaded"]
    dataset: DatasetSummary


class DatasetDeleteResponse(BaseModel):
    status: Literal["deleted"]
    dataset: DatasetSummary


class SavedNodePosition(BaseModel):
    """Canvas position persisted with a workflow node."""

    model_config = ConfigDict(allow_inf_nan=False)

    x: float
    y: float


class SavedNode(BaseModel):
    """Node payload stored in a ``.cuigraph.json`` project document."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
    label: str = Field(default="Node", max_length=MAX_PROJECT_NAME_CHARS)
    position: SavedNodePosition
    code: str = Field(max_length=MAX_NODE_CODE_CHARS)
    widget_values: dict[str, Any] = Field(default_factory=dict, alias="widgetValues")
    template_name: str | None = Field(
        default=None, max_length=MAX_NODE_IDENTIFIER_CHARS, alias="templateName"
    )
    color: str = Field(default="default", max_length=32)
    is_frame: bool = Field(default=False, alias="isFrame")
    width: float | None = Field(default=None, gt=0, le=MAX_FRAME_DIMENSION, allow_inf_nan=False)
    height: float | None = Field(default=None, gt=0, le=MAX_FRAME_DIMENSION, allow_inf_nan=False)
    parent_id: str | None = Field(
        default=None, max_length=MAX_NODE_IDENTIFIER_CHARS, alias="parentId"
    )
    is_subgraph: bool = Field(default=False, alias="isSubgraph")
    subgraph_id: str | None = Field(
        default=None, max_length=MAX_NODE_IDENTIFIER_CHARS, alias="subgraphId"
    )


class SavedEdge(BaseModel):
    """Edge payload stored in a ``.cuigraph.json`` project document."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(max_length=400)
    source: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
    source_handle: str = Field(
        min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS, alias="sourceHandle"
    )
    target: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
    target_handle: str = Field(
        min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS, alias="targetHandle"
    )


class SavedSubgraph(BaseModel):
    """Reusable nested graph with a shared input/output port contract."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1, max_length=MAX_NODE_IDENTIFIER_CHARS)
    name: str = Field(default="Subgraph", max_length=MAX_PROJECT_NAME_CHARS)
    inputs: list[InputSocket] = Field(default_factory=list, max_length=MAX_NODE_IDENTIFIER_CHARS)
    outputs: list[OutputSocket] = Field(default_factory=list, max_length=MAX_NODE_IDENTIFIER_CHARS)
    nodes: list[SavedNode] = Field(default_factory=list, max_length=MAX_GRAPH_NODES)
    edges: list[SavedEdge] = Field(default_factory=list, max_length=MAX_GRAPH_EDGES)
    input_position: SavedNodePosition = Field(
        default_factory=lambda: SavedNodePosition(x=-260, y=0),
        alias="inputPosition",
    )
    output_position: SavedNodePosition = Field(
        default_factory=lambda: SavedNodePosition(x=300, y=0),
        alias="outputPosition",
    )

    @field_validator("inputs", "outputs")
    @classmethod
    def _validate_ports(
        cls, value: list[InputSocket] | list[OutputSocket]
    ) -> list[InputSocket] | list[OutputSocket]:
        names = [socket.name for socket in value]
        if len(set(names)) != len(names):
            raise ValueError("Subgraph port names must be unique")
        return value


def _validate_saved_graph(
    nodes: list[SavedNode],
    edges: list[SavedEdge],
    subgraphs: dict[str, SavedSubgraph],
    definition: SavedSubgraph | None,
) -> None:
    scope = f"Subgraph '{definition.id}'" if definition else "Workflow"
    nodes_by_id = {node.id: node for node in nodes}
    if len(nodes_by_id) != len(nodes):
        raise ValueError(f"{scope} contains duplicate node ids")
    allowed = set(nodes_by_id)
    if definition is not None:
        allowed.update({SUBGRAPH_INPUT_NODE_ID, SUBGRAPH_OUTPUT_NODE_ID})
    input_names = {socket.name for socket in definition.inputs} if definition else set()
    output_names = {socket.name for socket in definition.outputs} if definition else set()

    for node in nodes:
        if node.is_frame and node.is_subgraph:
            raise ValueError(f"{scope} node '{node.id}' has conflicting types")
        if node.is_subgraph:
            if not node.subgraph_id or node.subgraph_id not in subgraphs:
                raise ValueError(f"{scope} node '{node.id}' references a missing subgraph")
        if node.parent_id is None:
            continue
        if node.parent_id == node.id:
            raise ValueError(f"Node '{node.id}' cannot be its own parent")
        parent = nodes_by_id.get(node.parent_id)
        if parent is None:
            raise ValueError(f"Node '{node.id}' references missing parent '{node.parent_id}'")
        if not parent.is_frame:
            raise ValueError(f"Node '{node.id}' must be parented to a frame node")

    edge_ids: set[str] = set()
    for edge in edges:
        if edge.id:
            if edge.id in edge_ids:
                raise ValueError(f"{scope} contains duplicate edge ids")
            edge_ids.add(edge.id)
        if edge.source not in allowed or edge.target not in allowed:
            raise ValueError(f"{scope} edge '{edge.id}' references unknown nodes")
        source_node = nodes_by_id.get(edge.source)
        target_node = nodes_by_id.get(edge.target)
        if source_node and source_node.is_frame:
            raise ValueError("Edges cannot reference frame nodes")
        if target_node and target_node.is_frame:
            raise ValueError("Edges cannot reference frame nodes")
        if source_node and source_node.is_subgraph:
            ports = subgraphs[source_node.subgraph_id or ""].outputs
            if not any(socket.name == edge.source_handle for socket in ports):
                raise ValueError(
                    f"{scope} edge references unknown subgraph output '{edge.source_handle}'"
                )
        if target_node and target_node.is_subgraph:
            ports = subgraphs[target_node.subgraph_id or ""].inputs
            if not any(socket.name == edge.target_handle for socket in ports):
                raise ValueError(
                    f"{scope} edge references unknown subgraph input '{edge.target_handle}'"
                )
        if definition is None:
            continue
        if edge.target == SUBGRAPH_INPUT_NODE_ID:
            raise ValueError("Subgraph input managers cannot receive edges")
        if edge.source == SUBGRAPH_OUTPUT_NODE_ID:
            raise ValueError("Subgraph output managers cannot emit edges")
        if edge.source == SUBGRAPH_INPUT_NODE_ID:
            if edge.source_handle not in input_names:
                raise ValueError(
                    f"Subgraph '{definition.id}' edge references unknown input "
                    f"'{edge.source_handle}'"
                )
            if edge.target == SUBGRAPH_OUTPUT_NODE_ID:
                raise ValueError("Subgraph inputs cannot connect directly to outputs")
        if edge.target == SUBGRAPH_OUTPUT_NODE_ID:
            if edge.target_handle not in output_names:
                raise ValueError(
                    f"Subgraph '{definition.id}' edge references unknown output "
                    f"'{edge.target_handle}'"
                )


def _validate_subgraph_cycles(
    root_nodes: list[SavedNode],
    subgraphs: dict[str, SavedSubgraph],
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(definition_id: str) -> None:
        if definition_id in visited:
            return
        if definition_id in visiting:
            raise ValueError(f"Workflow contains a subgraph cycle involving '{definition_id}'")
        visiting.add(definition_id)
        for node in subgraphs[definition_id].nodes:
            if node.is_subgraph and node.subgraph_id:
                if node.subgraph_id not in subgraphs:
                    raise ValueError(
                        f"Subgraph '{definition_id}' references missing subgraph "
                        f"'{node.subgraph_id}'"
                    )
                visit(node.subgraph_id)
        visiting.remove(definition_id)
        visited.add(definition_id)

    for definition_id in subgraphs:
        visit(definition_id)
    for node in root_nodes:
        if node.is_subgraph and (not node.subgraph_id or node.subgraph_id not in subgraphs):
            raise ValueError(f"Workflow node '{node.id}' references missing subgraph")


class SavedWorkflowDocument(BaseModel):
    """Full editable workflow document, including UI metadata."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(default=None, max_length=MAX_PROJECT_NAME_CHARS)
    version: int = PROJECT_SCHEMA_VERSION
    saved_at: str = Field(default="", max_length=64, alias="savedAt")
    nodes: list[SavedNode] = Field(default_factory=list, max_length=MAX_GRAPH_NODES)
    edges: list[SavedEdge] = Field(default_factory=list, max_length=MAX_GRAPH_EDGES)
    subgraphs: dict[str, SavedSubgraph] = Field(default_factory=dict, max_length=MAX_SUBGRAPHS)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value != PROJECT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported workflow version: {value}")
        return value

    @model_validator(mode="after")
    def _validate_graph(self) -> SavedWorkflowDocument:
        for key, definition in self.subgraphs.items():
            if key != definition.id:
                raise ValueError(f"Subgraph key '{key}' does not match id '{definition.id}'")
        _validate_saved_graph(self.nodes, self.edges, self.subgraphs, None)
        for definition in self.subgraphs.values():
            _validate_saved_graph(
                definition.nodes,
                definition.edges,
                self.subgraphs,
                definition,
            )
        _validate_subgraph_cycles(self.nodes, self.subgraphs)
        return self


class ProjectSummary(BaseModel):
    slug: str
    name: str
    saved_at: str
    node_count: int
    edge_count: int
    version_count: int = 0


class ProjectDetail(ProjectSummary):
    workflow: SavedWorkflowDocument


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_PROJECT_NAME_CHARS)
    workflow: SavedWorkflowDocument
    overwrite: bool = False

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Project name must not be blank")
        return cleaned


class ProjectUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_PROJECT_NAME_CHARS)
    workflow: SavedWorkflowDocument

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Project name must not be blank")
        return cleaned


class ProjectSaveResponse(BaseModel):
    status: Literal["created", "updated"]
    project: ProjectSummary


class ProjectDeleteResponse(BaseModel):
    status: Literal["deleted"]
    slug: str


class ProjectVersionInfo(BaseModel):
    version: str
    saved_at: str
    node_count: int
    edge_count: int
    size_bytes: int


class ProjectVersionDetail(ProjectVersionInfo):
    workflow: SavedWorkflowDocument


class WorkspaceSessionRequest(BaseModel):
    workflow: SavedWorkflowDocument
    project_slug: str | None = Field(default=None, max_length=MAX_PROJECT_SLUG_CHARS)
    project_name: str | None = Field(default=None, max_length=MAX_PROJECT_NAME_CHARS)


class WorkspaceSession(BaseModel):
    saved_at: str
    project_slug: str | None = None
    project_name: str | None = None
    workflow: SavedWorkflowDocument


class WorkspaceSessionResponse(BaseModel):
    status: Literal["saved"]
    saved_at: str


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
            raise ValueError("code_hash must be a 64-character hex sha256 digest")
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
