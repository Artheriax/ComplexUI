"""FastAPI entry point: REST endpoints and WebSocket execution streaming.

``POST /api/execute`` accepts a workflow graph, runs a synchronous pre-flight
(parse + wiring + trust validation) and then executes it in an isolated child
process while the event loop stays responsive. Telemetry flows through
:class:`ConnectionManager` to every client connected to ``/ws/execution``.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import mimetypes
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .accelerator import configured_accelerator, effective_accelerator
from .datasets import (
    MAX_UPLOAD_REQUEST_BYTES,
    DatasetNotFound,
    DatasetPreviewError,
    DatasetTooLarge,
    InvalidAssetPath,
    InvalidDataset,
    get_dataset_store,
    get_output_store,
)
from .execution_host import (
    WorkerHandle,
    execution_limits,
    process_memory_bytes,
    start_worker,
    stop_worker,
)
from .executor import (
    GraphError,
    GraphPlan,
    ModelSelectionError,
    restore_cache,
    validate_graph,
)
from .exporter import (
    ExportError,
    export_notebook,
    export_python,
)
from .logging_setup import (
    FileFormatter,
    configure_logging,
    resolve_log_dir,
)
from .models import InvalidModelPath, get_model_store
from .network_security import (
    RequestSizeLimitMiddleware,
    configured_allowed_hosts,
    origin_is_allowed,
)
from .nvml import nvidia_available
from .run_history import (
    InvalidRunId,
    RunHistoryError,
    RunNotFound,
    RunRecorder,
    get_run_history_store,
)
from .schemas import (
    DEFAULT_DATASET_PREVIEW_ROWS,
    MAX_DATASET_PREVIEW_ROWS,
    RUN_HISTORY_RETENTION,
    ApproveRequest,
    ApproveResponse,
    DatasetDeleteResponse,
    DatasetPreview,
    DatasetSummary,
    DatasetUploadResponse,
    EventType,
    ExecuteRequest,
    ExecuteResponse,
    ExecutionEvent,
    ExecutionStatusResponse,
    ExportFormat,
    ExportRequest,
    ExportResponse,
    ModelSummary,
    ModelTypeSummary,
    NodeSchema,
    OutputDeleteResponse,
    OutputSummary,
    ParseRequest,
    ProjectCreateRequest,
    ProjectDeleteResponse,
    ProjectDetail,
    ProjectSaveResponse,
    ProjectSummary,
    ProjectUpdateRequest,
    ProjectVersionDetail,
    ProjectVersionInfo,
    RunDeleteResponse,
    RunDetail,
    RunErrorRecord,
    RunStatus,
    RunSummary,
    SystemMetricsResponse,
    TemplateCreateRequest,
    TemplateInfo,
    WorkflowGraph,
    WorkspaceSession,
    WorkspaceSessionRequest,
    WorkspaceSessionResponse,
)
from .security import code_hash, get_security_registry
from .system_metrics import collect_system_metrics
from .template_catalog import template_meta
from .transpiler import TranspileError, parse_node
from .user_templates import (
    InvalidUserTemplate,
    UserTemplateConflict,
    UserTemplateError,
    get_user_template_store,
)
from .workflow_catalog import (
    WORKFLOW_CATEGORIES,
    WorkflowTemplateInfo,
    get_workflows,
)
from .workspace import (
    InvalidProjectDocument,
    InvalidProjectSlug,
    ProjectAlreadyExists,
    ProjectNotFound,
    WorkspaceError,
    get_project_store,
)

__all__ = ["app", "coordinator", "manager"]

_LOG_DIR = resolve_log_dir()
configure_logging(
    os.environ.get("COMPLEXUI_LOG_LEVEL", "INFO"),
    log_dir=_LOG_DIR,
)
logger = logging.getLogger("complexui.server")

_SEND_TIMEOUT_SECONDS = 2.0


class ConnectionManager:
    """Thread-safe bridge between executor telemetry and WebSocket clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._queue: asyncio.Queue[ExecutionEvent] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drainer: asyncio.Task[None] | None = None

    def ensure_started(self) -> None:
        """Bind the manager to the running loop (safe to call repeatedly)."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._queue = asyncio.Queue()
            self._connections.clear()
            self._drainer = None
        if self._drainer is None or self._drainer.done():
            self._drainer = loop.create_task(self._drain())

    async def _drain(self) -> None:
        queue = self._queue
        if queue is None:
            return
        while True:
            event = await queue.get()
            _log_event(event)
            await self._broadcast(event)

    async def _broadcast(self, event: ExecutionEvent) -> None:
        payload = event.model_dump(mode="json")
        dead: list[WebSocket] = []
        for connection in list(self._connections):
            try:
                await asyncio.wait_for(
                    connection.send_json(payload),
                    timeout=_SEND_TIMEOUT_SECONDS,
                )
            except Exception:
                dead.append(connection)
        for connection in dead:
            self._connections.discard(connection)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.ensure_started()
        self._connections.add(websocket)

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    def emit(self, event: ExecutionEvent) -> None:
        """Enqueue an event from the event loop thread."""
        if self._queue is not None:
            self._queue.put_nowait(event)

    def emit_from_thread(self, event: ExecutionEvent) -> None:
        """Enqueue an event from a worker thread (executor callback)."""
        loop, queue = self._loop, self._queue
        if loop is None or queue is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:
            return


class ExecutionCoordinator:
    """Allows at most one workflow execution at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_id: str | None = None
        self._stop_event: threading.Event | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._run_id is not None

    @property
    def run_id(self) -> str | None:
        with self._lock:
            return self._run_id

    def try_begin(self) -> tuple[str, threading.Event] | None:
        with self._lock:
            if self._run_id is not None:
                return None
            self._run_id = uuid4().hex
            self._stop_event = threading.Event()
            return self._run_id, self._stop_event

    def cancel(self) -> bool:
        """Request cancellation of the running execution (False if idle)."""
        with self._lock:
            if self._stop_event is None:
                return False
            self._stop_event.set()
            return True

    def finish(self) -> None:
        with self._lock:
            self._run_id = None
            self._stop_event = None


def _log_event(event: ExecutionEvent, *, target: logging.Logger | None = None) -> None:
    """Narrate every execution telemetry event on the console or a run log."""
    log = target if target is not None else logger
    data = event.data
    run_id = str(data.get("run_id", ""))
    short_run = run_id[:12]
    extra: dict[str, Any] = {
        "run_id": run_id or None,
        "node_id": event.node_id or None,
    }
    if event.type is EventType.EXECUTION_STARTED:
        order = data.get("order")
        node_count = len(order) if isinstance(order, list) else 0
        log.info("run %s started: %d node(s)", short_run, node_count, extra=extra)
        log.debug("run %s order: %s", short_run, order, extra=extra)
    elif event.type is EventType.NODE_STARTED:
        log.info("  [start] %s", event.node_id, extra=extra)
    elif event.type is EventType.INSTALLING_DEPS:
        dependencies = data.get("dependencies") or []
        extra["dependencies"] = dependencies
        log.info(
            "  [deps ] %s installing: %s",
            event.node_id,
            ", ".join(str(dep) for dep in dependencies),
            extra=extra,
        )
    elif event.type is EventType.NODE_PROGRESS:
        log.debug(
            "  [prog ] %s %s",
            event.node_id,
            data.get("message", ""),
            extra=extra,
        )
    elif event.type is EventType.NODE_COMPLETED:
        cached = bool(data.get("cached"))
        extra["cached"] = cached
        if cached:
            log.info("  [done ] %s (cached)", event.node_id, extra=extra)
        else:
            duration = data.get("duration_ms")
            if isinstance(duration, (int, float)):
                extra["duration_ms"] = duration
            log.info("  [done ] %s in %sms", event.node_id, duration, extra=extra)
    elif event.type is EventType.NODE_ERROR:
        extra["error_type"] = data.get("error_type")
        extra["line"] = data.get("line")
        log.error(
            "  [error] %s: %s (line %s): %s",
            event.node_id,
            data.get("error_type"),
            data.get("line"),
            data.get("message", ""),
            extra=extra,
        )
    elif event.type is EventType.EXECUTION_COMPLETED:
        cached_map = data.get("cached")
        cached_map = cached_map if isinstance(cached_map, dict) else {}
        cached_count = sum(1 for value in cached_map.values() if value)
        duration = data.get("duration_ms")
        extra["cached"] = cached_count
        if isinstance(duration, (int, float)):
            extra["duration_ms"] = duration
            log.info(
                "run %s completed in %sms: %d node(s), %d cached",
                short_run,
                duration,
                len(cached_map),
                cached_count,
                extra=extra,
            )
        else:
            log.info(
                "run %s completed: %d node(s), %d cached",
                short_run,
                len(cached_map),
                cached_count,
                extra=extra,
            )
    elif event.type is EventType.EXECUTION_FAILED:
        failed_node = data.get("node_id")
        extra["error_type"] = data.get("error_type")
        extra["line"] = data.get("line")
        duration = data.get("duration_ms")
        if isinstance(duration, (int, float)):
            extra["duration_ms"] = duration
        log.error(
            "run %s failed%s: %s",
            short_run,
            f" at {failed_node}" if failed_node else "",
            data.get("message"),
            extra=extra,
        )
        traceback_text = data.get("traceback")
        if isinstance(traceback_text, str) and traceback_text:
            log.debug("run %s traceback:\n%s", short_run, traceback_text, extra=extra)
    elif event.type is EventType.EXECUTION_CANCELLED:
        duration = data.get("duration_ms")
        if isinstance(duration, (int, float)):
            extra["duration_ms"] = duration
        log.warning("run %s cancelled", short_run, extra=extra)
    else:
        log.debug("event %s node=%s", event.type.value, event.node_id, extra=extra)


_ALLOWED_HOSTS = configured_allowed_hosts()

app = FastAPI(title="ComplexUI Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_ALLOWED_HOSTS,
    www_redirect=False,
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    overrides={"/api/uploads": MAX_UPLOAD_REQUEST_BYTES},
)

manager = ConnectionManager()
coordinator = ExecutionCoordinator()
_background_tasks: set[asyncio.Task[None]] = set()

logger.info(
    "ComplexUI backend ready: %d official template(s), allowed hosts: %s",
    len(get_security_registry().templates()),
    ", ".join(_ALLOWED_HOSTS),
)
if _LOG_DIR is not None:
    logger.info("daily logs: %s", _LOG_DIR)
logger.info("workspace: %s", get_project_store().root)
logger.info("models: %s", get_model_store().root)
if "*" in _ALLOWED_HOSTS:
    logger.warning(
        "network mode is enabled: Host validation accepts any hostname and "
        "the API has no authentication"
    )


@app.post("/api/parse", response_model=NodeSchema)
def parse_endpoint(payload: ParseRequest) -> NodeSchema:
    """Statically parse node code into a NodeSchema (never executes code)."""
    schema = parse_node(payload.code, payload.node_id)
    logger.debug(
        "parse: node=%s errors=%d inputs=%d outputs=%d widgets=%d",
        payload.node_id or "<anon>",
        len(schema.syntax_errors),
        len(schema.inputs),
        len(schema.outputs),
        len(schema.widgets),
    )
    return schema


@app.get("/api/templates", response_model=list[TemplateInfo])
def list_templates() -> list[TemplateInfo]:
    """Return bundled and user templates with code, hashes and schemas."""
    registry = get_security_registry()
    templates: list[TemplateInfo] = []
    for record in registry.templates():
        if registry.is_user_template(record.name):
            user = get_user_template_store().get(record.name)
            title, category, description = (
                user.title,
                user.category,
                user.description,
            )
            source: str = "user"
            hidden = False
        else:
            meta = template_meta(record.name)
            title, category = meta.title, meta.category
            description, hidden = meta.description, meta.hidden
            source = "builtin"
        templates.append(
            TemplateInfo(
                name=record.name,
                title=title,
                category=category,
                description=description,
                hidden=hidden,
                source=source,
                code=record.code,
                code_hash=record.code_hash,
                node_schema=parse_node(record.code, record.name),
            )
        )
    return templates


@app.post("/api/templates", status_code=201, response_model=TemplateInfo)
def create_template(payload: TemplateCreateRequest) -> TemplateInfo:
    """Save a scratch node as a reusable, locally trusted user template."""
    registry = get_security_registry()
    if not registry.is_user_template(payload.name):
        try:
            registry.get_template(payload.name)
        except KeyError:
            pass
        else:
            raise HTTPException(
                status_code=409,
                detail=f"'{payload.name}' is a bundled template name",
            )
    schema = parse_node(payload.code, payload.name)
    if schema.syntax_errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Template code has syntax errors",
                "syntax_errors": [
                    error.model_dump(mode="json") for error in schema.syntax_errors
                ],
            },
        )
    try:
        record = get_user_template_store().save(
            name=payload.name,
            title=payload.title,
            category=payload.category,
            description=payload.description,
            code=payload.code,
            overwrite=payload.overwrite,
        )
    except UserTemplateConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidUserTemplate as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UserTemplateError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    registry.reload_templates()
    logger.info("user template saved: %s", record.name)
    return TemplateInfo(
        name=record.name,
        title=record.title,
        category=record.category,
        description=record.description,
        hidden=False,
        source="user",
        code=record.code,
        code_hash=code_hash(record.code),
        node_schema=parse_node(record.code, record.name),
    )


@app.delete("/api/templates/{name}", status_code=200)
def delete_template(name: str) -> dict[str, str]:
    """Delete a user template; bundled templates cannot be removed."""
    registry = get_security_registry()
    if not registry.is_user_template(name):
        raise HTTPException(
            status_code=403,
            detail="Only user-created templates can be deleted",
        )
    try:
        get_user_template_store().delete(name)
    except UserTemplateError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    registry.reload_templates()
    logger.info("user template deleted: %s", name)
    return {"status": "deleted", "name": name}


@app.get("/api/workflows", response_model=list[WorkflowTemplateInfo])
def list_workflows() -> list[WorkflowTemplateInfo]:
    """Return the bundled example workflows, ordered by category then title."""
    workflows = get_workflows().values()
    return sorted(
        workflows,
        key=lambda item: (
            WORKFLOW_CATEGORIES.index(item.category),
            item.title,
        ),
    )


@app.get("/api/health")
def health_endpoint() -> dict[str, object]:
    """Liveness probe used by the launcher and external tools."""
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    limits = execution_limits()
    return {
        "status": "ok",
        "version": app.version,
        "templates": len(get_security_registry().templates()),
        "workflows": len(get_workflows()),
        "models_root": str(get_model_store().root),
        "accelerator": configured_accelerator(),
        "effective_accelerator": effective_accelerator(),
        "torch": torch_version,
        "torch_cuda": bool(torch_version and "+cu" in torch_version.lower()),
        "nvidia": nvidia_available(),
        "limits": {
            "node_timeout_seconds": limits.node_timeout_seconds,
            "run_timeout_seconds": limits.run_timeout_seconds,
            "max_memory_mb": limits.max_memory_bytes // (1024 * 1024),
        },
    }


@app.get("/api/system/metrics", response_model=SystemMetricsResponse)
def system_metrics_endpoint() -> SystemMetricsResponse:
    """Return a live CPU, memory and NVIDIA GPU snapshot."""
    return collect_system_metrics()


@app.get("/api/models/types", response_model=list[ModelTypeSummary])
def list_model_types_endpoint() -> list[ModelTypeSummary]:
    """List default and custom folders in the repo model library."""
    try:
        return get_model_store().list_types()
    except (InvalidModelPath, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models", response_model=list[ModelSummary])
def list_models_endpoint(
    model_type: str | None = Query(default=None, alias="type"),
) -> list[ModelSummary]:
    """List metadata-only model files, optionally scoped to one type."""
    try:
        return get_model_store().list_models(model_type)
    except InvalidModelPath as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/status", response_model=ExecutionStatusResponse)
def status_endpoint() -> ExecutionStatusResponse:
    """Report whether an execution is running (used to resync after reconnects)."""
    run_id = coordinator.run_id
    return ExecutionStatusResponse(
        status="running" if run_id is not None else "idle",
        run_id=run_id,
    )


@app.get("/api/runs", response_model=list[RunSummary])
def list_runs_endpoint(
    limit: Annotated[int, Query(ge=1, le=200)] = RUN_HISTORY_RETENTION,
) -> list[RunSummary]:
    """Return retained terminal runs, newest first."""
    try:
        return get_run_history_store().list_runs(limit)
    except RunHistoryError as exc:
        logger.error("run history list failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}", response_model=RunDetail)
def get_run_endpoint(run_id: str) -> RunDetail:
    """Load one retained run with per-node outputs and errors."""
    try:
        return get_run_history_store().get(run_id)
    except InvalidRunId as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunHistoryError as exc:
        logger.error("run history read failed for %s: %s", run_id[:12], exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/runs/{run_id}", response_model=RunDeleteResponse)
def delete_run_endpoint(run_id: str) -> RunDeleteResponse:
    """Delete one retained run record."""
    try:
        get_run_history_store().delete(run_id)
    except InvalidRunId as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunHistoryError as exc:
        logger.error("run history delete failed for %s: %s", run_id[:12], exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RunDeleteResponse(status="deleted", run_id=run_id)


@app.post("/api/approve", response_model=ApproveResponse)
def approve_endpoint(payload: ApproveRequest) -> ApproveResponse:
    """Approve custom/modified node code for the current session."""
    get_security_registry().approve(payload.node_id, payload.code_hash)
    logger.info("approved node '%s' (hash %s…)", payload.node_id, payload.code_hash[:12])
    return ApproveResponse(
        status="approved",
        node_id=payload.node_id,
        code_hash=payload.code_hash,
    )


@app.post("/api/export", response_model=ExportResponse)
def export_endpoint(payload: ExportRequest) -> ExportResponse:
    """Render the workflow as a standalone Python script or notebook."""
    try:
        if payload.format is ExportFormat.NOTEBOOK:
            exported = export_notebook(payload.graph, targets=payload.targets)
        else:
            exported = export_python(payload.graph, targets=payload.targets)
    except TranspileError as exc:
        logger.error(
            "export rejected: syntax error in node '%s' (line %d)",
            exc.schema.node_id,
            exc.schema.syntax_errors[0].line if exc.schema.syntax_errors else 0,
        )
        raise HTTPException(status_code=422, detail=exc.schema.model_dump(mode="json")) from exc
    except (GraphError, ExportError) as exc:
        logger.error("export rejected: %s", exc)
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc

    logger.info(
        "export: format=%s nodes=%d filename=%s bytes=%d",
        payload.format.value,
        len(payload.graph.nodes),
        exported.filename,
        len(exported.content.encode("utf-8")),
    )
    return ExportResponse(
        filename=exported.filename,
        media_type=exported.media_type,
        content=exported.content,
    )


def _dataset_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DatasetNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, DatasetTooLarge):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, (InvalidDataset, InvalidAssetPath, DatasetPreviewError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _media_response(path: Path) -> FileResponse:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(path.name)}",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/uploads", response_model=list[DatasetSummary])
def list_uploads_endpoint() -> list[DatasetSummary]:
    """List assets stored in the workspace upload directory."""
    try:
        return get_dataset_store().list_datasets()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/uploads", response_model=DatasetUploadResponse)
async def upload_dataset_endpoint(
    file: Annotated[UploadFile, File()],
) -> DatasetUploadResponse:
    """Stream one dataset into the workspace with a size cap."""
    if not file.filename:
        raise HTTPException(status_code=422, detail="Upload requires a filename")
    store = get_dataset_store()
    try:
        dataset = await run_in_threadpool(store.save, file.filename, file.file)
    except (DatasetNotFound, DatasetTooLarge, InvalidDataset) as exc:
        raise _dataset_http_error(exc) from exc
    except OSError as exc:
        logger.error("dataset upload failed: %s", exc)
        raise _dataset_http_error(exc) from exc
    finally:
        await file.close()
    logger.info(
        "dataset uploaded: '%s' (%d bytes, path=%s)",
        dataset.name,
        dataset.size_bytes,
        dataset.path,
    )
    return DatasetUploadResponse(status="uploaded", dataset=dataset)


@app.get("/api/uploads/{file_id}", response_model=DatasetSummary)
def get_upload_endpoint(file_id: str) -> DatasetSummary:
    """Return metadata for one uploaded asset."""
    try:
        return get_dataset_store().get_dataset(file_id)
    except (DatasetNotFound, InvalidDataset, OSError) as exc:
        raise _dataset_http_error(exc) from exc


@app.get("/api/uploads/{file_id}/raw")
def get_upload_raw_endpoint(file_id: str) -> FileResponse:
    """Serve an uploaded image or video inline with HTTP range support."""
    try:
        return _media_response(get_dataset_store().raw_path(file_id))
    except (DatasetNotFound, InvalidDataset, OSError) as exc:
        raise _dataset_http_error(exc) from exc


@app.get("/api/outputs", response_model=list[OutputSummary])
def list_outputs_endpoint(
    limit: int = Query(default=200, ge=1, le=500),
) -> list[OutputSummary]:
    """List generated images and videos, newest first."""
    try:
        return get_output_store().list_outputs(limit)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/outputs/{file_id}/raw")
def get_output_raw_endpoint(file_id: str) -> FileResponse:
    """Serve a generated image or video inline with HTTP range support."""
    try:
        return _media_response(get_output_store().media_path(file_id))
    except (DatasetNotFound, InvalidAssetPath, OSError) as exc:
        raise _dataset_http_error(exc) from exc


@app.delete("/api/outputs/{file_id}", response_model=OutputDeleteResponse)
def delete_output_endpoint(file_id: str) -> OutputDeleteResponse:
    """Delete one generated image or video."""
    try:
        output = get_output_store().delete_output(file_id)
    except (DatasetNotFound, InvalidAssetPath, OSError) as exc:
        raise _dataset_http_error(exc) from exc
    logger.info("generated output deleted: %s", file_id)
    return OutputDeleteResponse(status="deleted", output=output)


@app.get("/api/uploads/{file_id}/preview", response_model=DatasetPreview)
def preview_upload_endpoint(
    file_id: str,
    limit: int = Query(
        default=DEFAULT_DATASET_PREVIEW_ROWS,
        ge=1,
        le=MAX_DATASET_PREVIEW_ROWS,
    ),
) -> DatasetPreview:
    """Return a bounded table preview for one uploaded dataset."""
    try:
        return get_dataset_store().preview(file_id, limit=limit)
    except (DatasetNotFound, DatasetPreviewError, InvalidDataset, OSError) as exc:
        raise _dataset_http_error(exc) from exc


@app.delete("/api/uploads/{file_id}", response_model=DatasetDeleteResponse)
def delete_upload_endpoint(file_id: str) -> DatasetDeleteResponse:
    """Delete one uploaded dataset."""
    try:
        dataset = get_dataset_store().delete(file_id)
    except (DatasetNotFound, InvalidDataset, OSError) as exc:
        raise _dataset_http_error(exc) from exc
    logger.info("dataset deleted: %s", file_id)
    return DatasetDeleteResponse(status="deleted", dataset=dataset)


@app.get("/api/projects", response_model=list[ProjectSummary])
def list_projects_endpoint() -> list[ProjectSummary]:
    """List named projects stored in the workspace."""
    return get_project_store().list_projects()


@app.post("/api/projects", response_model=ProjectSaveResponse)
def create_project_endpoint(payload: ProjectCreateRequest) -> ProjectSaveResponse:
    """Create a named project, optionally replacing an existing one."""
    try:
        project, created = get_project_store().create_project(
            payload.name, payload.workflow, overwrite=payload.overwrite
        )
    except ProjectAlreadyExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidProjectSlug as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkspaceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info(
        "project %s: '%s' (%d node(s), slug=%s)",
        "created" if created else "overwritten",
        payload.name,
        len(payload.workflow.nodes),
        project.slug,
    )
    return ProjectSaveResponse(status="created" if created else "updated", project=project)


@app.get("/api/projects/{slug}", response_model=ProjectDetail)
def get_project_endpoint(slug: str) -> ProjectDetail:
    """Load a project document by slug."""
    try:
        return get_project_store().get_project(slug)
    except ProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidProjectSlug as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidProjectDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkspaceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/projects/{slug}", response_model=ProjectSaveResponse)
def update_project_endpoint(slug: str, payload: ProjectUpdateRequest) -> ProjectSaveResponse:
    """Overwrite a project and retain its previous contents as a snapshot."""
    try:
        project = get_project_store().update_project(slug, payload.name, payload.workflow)
    except ProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidProjectSlug as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkspaceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info(
        "project saved: '%s' (%d node(s), slug=%s, %d version(s))",
        project.name,
        project.node_count,
        project.slug,
        project.version_count,
    )
    return ProjectSaveResponse(status="updated", project=project)


@app.delete("/api/projects/{slug}", response_model=ProjectDeleteResponse)
def delete_project_endpoint(slug: str) -> ProjectDeleteResponse:
    """Move a project and its snapshots to the workspace trash."""
    try:
        get_project_store().delete_project(slug)
    except ProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidProjectSlug as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkspaceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info("project deleted: slug=%s", slug)
    return ProjectDeleteResponse(status="deleted", slug=slug)


@app.get("/api/projects/{slug}/versions", response_model=list[ProjectVersionInfo])
def list_project_versions_endpoint(slug: str) -> list[ProjectVersionInfo]:
    """List retained version snapshots for a project."""
    try:
        return get_project_store().list_versions(slug)
    except ProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidProjectSlug as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkspaceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/api/projects/{slug}/versions/{version}",
    response_model=ProjectVersionDetail,
)
def get_project_version_endpoint(slug: str, version: str) -> ProjectVersionDetail:
    """Load one retained snapshot."""
    try:
        return get_project_store().get_version(slug, version)
    except ProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidProjectSlug as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidProjectDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkspaceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/session", response_model=WorkspaceSession | None)
def get_session_endpoint() -> WorkspaceSession | None:
    """Return the latest close/visibility autosave for recovery."""
    return get_project_store().load_session()


@app.put("/api/session", response_model=WorkspaceSessionResponse)
def save_session_endpoint(
    payload: WorkspaceSessionRequest,
) -> WorkspaceSessionResponse:
    """Persist the canvas when the page is hidden or closed."""
    try:
        session = get_project_store().save_session(payload)
    except WorkspaceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.debug(
        "session autosave: %d node(s), project=%s",
        len(payload.workflow.nodes),
        payload.project_slug or "<unsaved>",
    )
    return WorkspaceSessionResponse(status="saved", saved_at=session.saved_at)


@app.post("/api/execute", status_code=202, response_model=ExecuteResponse)
async def execute_endpoint(payload: ExecuteRequest) -> ExecuteResponse:
    """Pre-flight a workflow graph and start async execution + WS streaming."""
    if coordinator.running:
        logger.warning("execute rejected: a run is already in progress")
        raise HTTPException(status_code=409, detail="An execution is already running")

    try:
        plan = validate_graph(payload.graph, targets=payload.targets)
    except TranspileError as exc:
        logger.error(
            "execute rejected: syntax error in node '%s' (line %d)",
            exc.schema.node_id,
            exc.schema.syntax_errors[0].line if exc.schema.syntax_errors else 0,
        )
        raise HTTPException(status_code=422, detail=exc.schema.model_dump(mode="json")) from exc
    except ModelSelectionError as exc:
        logger.error("execute rejected: %s", exc)
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "type": "ModelSelection", "models": exc.issues},
        ) from exc
    except GraphError as exc:
        logger.error("execute rejected: %s", exc)
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc

    if plan.untrusted:
        logger.warning(
            "execute blocked: %d untrusted node(s): %s",
            len(plan.untrusted),
            ", ".join(node_id for node_id, _hash in plan.untrusted),
        )
        raise HTTPException(
            status_code=403,
            detail={
                "message": (
                    "Workflow contains untrusted code; approve the listed nodes before execution"
                ),
                "untrusted_nodes": [
                    {"node_id": node_id, "code_hash": hash_value}
                    for node_id, hash_value in plan.untrusted
                ],
            },
        )

    run = coordinator.try_begin()
    if run is None:  # unreachable: no await since the check above
        raise HTTPException(status_code=409, detail="An execution is already running")
    run_id, stop_event = run
    targets = payload.targets
    logger.info(
        "execute accepted: %d node(s), run=%s, targets=%s",
        len(payload.graph.nodes),
        run_id[:12],
        targets if targets else "all",
    )

    manager.ensure_started()
    task = asyncio.create_task(
        _run_execution(
            payload.graph,
            plan,
            run_id,
            stop_event,
            payload.targets,
            payload.node_labels,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return ExecuteResponse(status="started", run_id=run_id)


@app.post("/api/cancel", status_code=202)
def cancel_endpoint() -> dict[str, str]:
    """Request cancellation of the running execution (checked between nodes)."""
    if not coordinator.cancel():
        logger.debug("cancel ignored: no execution is running")
        raise HTTPException(status_code=409, detail="No execution is running")
    logger.info("cancel requested: stopping after the current node")
    return {"status": "cancelling"}


@app.websocket("/ws/execution")
async def execution_socket(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if not origin_is_allowed(origin, _ALLOWED_HOSTS):
        logger.warning("websocket rejected: disallowed origin %s", origin)
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    await manager.connect(websocket)
    logger.debug("telemetry client connected (%d total)", manager.connection_count)
    try:
        while True:
            await websocket.receive()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        manager.disconnect(websocket)
        logger.debug("telemetry client disconnected (%d left)", manager.connection_count)


def _open_run_log(
    run_id: str,
) -> tuple[logging.Logger | None, logging.FileHandler | None]:
    """Create a per-run plain-text log when file logging is enabled."""
    log_dir = resolve_log_dir()
    if log_dir is None:
        return None, None
    path = log_dir / "runs" / f"{run_id}.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        logger.warning("run log disabled for %s: %s", run_id[:12], exc)
        return None, None
    handler.setFormatter(FileFormatter())
    run_logger = logging.getLogger(f"complexui.run.{run_id}")
    run_logger.setLevel(logging.DEBUG)
    run_logger.propagate = False
    run_logger.addHandler(handler)
    logger.debug("run %s log: %s", run_id[:12], path)
    return run_logger, handler


def _close_run_log(run_logger: logging.Logger | None, handler: logging.FileHandler | None) -> None:
    if run_logger is None or handler is None:
        return
    run_logger.removeHandler(handler)
    handler.close()


def _receive_message(connection: Any, timeout: float) -> dict[str, Any] | None:
    if not connection.poll(timeout):
        return None
    message = connection.recv()
    return message if isinstance(message, dict) else None


async def _run_execution(
    graph: WorkflowGraph,
    plan: GraphPlan,
    run_id: str,
    stop_event: threading.Event,
    targets: list[str] | None,
    node_labels: dict[str, str] | None,
) -> None:
    started = time.perf_counter()
    run_logger, run_handler = _open_run_log(run_id)
    recorder = RunRecorder(run_id, plan, node_labels, targets)
    limits = execution_limits()
    handle: WorkerHandle | None = None
    terminal: dict[str, Any] | None = None
    abnormal: tuple[str, str, str | None] | None = None
    current_node_id: str | None = None
    node_started_at: float | None = None

    def emit_event(event: ExecutionEvent) -> None:
        event.data.setdefault("run_id", run_id)
        try:
            recorder.observe(event)
        except Exception as exc:
            logger.warning("run history telemetry ignored for %s: %s", run_id[:12], exc)
        if run_logger is not None:
            _log_event(event, target=run_logger)
        manager.emit(event)

    async def persist_run(
        status: RunStatus,
        *,
        duration_ms: float,
        error: RunErrorRecord | None = None,
    ) -> None:
        try:
            record = recorder.finish(status, duration_ms=duration_ms, error=error)
            await asyncio.to_thread(get_run_history_store().save, record)
        except Exception as exc:
            logger.warning("run history write failed for %s: %s", run_id[:12], exc)

    def apply_cache(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        try:
            restore_cache(payload)
        except Exception as exc:
            logger.warning("cache transfer failed for %s: %s", run_id[:12], exc)

    def failure_from(payload: dict[str, Any]) -> RunErrorRecord:
        node_id = payload.get("node_id")
        line = payload.get("line")
        return RunErrorRecord(
            message=str(payload.get("message", "Execution failed")),
            error_type=str(payload.get("error_type", "Error")),
            node_id=node_id if isinstance(node_id, str) else None,
            line=line if isinstance(line, int) else None,
            traceback=str(payload.get("traceback", "")),
        )

    try:
        emit_event(
            ExecutionEvent(
                type=EventType.EXECUTION_STARTED,
                node_id="",
                data={"run_id": run_id, "order": plan.order},
            )
        )
        worker_started = time.perf_counter()
        try:
            handle = await asyncio.to_thread(
                start_worker,
                graph,
                plan,
                limits.max_memory_bytes,
            )
        except Exception as exc:
            terminal = {
                "kind": "failed",
                "failure": {
                    "error_type": type(exc).__name__,
                    "message": f"Could not start execution process: {exc}",
                    "node_id": None,
                    "line": None,
                    "traceback": "",
                },
            }

        while handle is not None and terminal is None:
            if stop_event.is_set():
                handle.stop_event.set()
            try:
                message = await asyncio.to_thread(_receive_message, handle.connection, 0.1)
            except EOFError:
                abnormal = (
                    "WorkerCrashed",
                    "The execution process exited unexpectedly.",
                    current_node_id,
                )
                break
            if message is None:
                now = time.perf_counter()
                if (
                    limits.run_timeout_seconds > 0
                    and now - worker_started >= limits.run_timeout_seconds
                ):
                    abnormal = (
                        "RunTimeout",
                        f"Execution exceeded the {limits.run_timeout_seconds:.0f}s run timeout.",
                        current_node_id,
                    )
                    break
                if (
                    limits.node_timeout_seconds > 0
                    and current_node_id is not None
                    and node_started_at is not None
                    and now - node_started_at >= limits.node_timeout_seconds
                ):
                    abnormal = (
                        "NodeTimeout",
                        f"Node '{current_node_id}' exceeded the "
                        f"{limits.node_timeout_seconds:.0f}s node timeout.",
                        current_node_id,
                    )
                    break
                if limits.max_memory_bytes > 0:
                    used = process_memory_bytes(handle.process.pid)
                    if used is not None and used > limits.max_memory_bytes:
                        abnormal = (
                            "MemoryLimit",
                            "Execution exceeded the "
                            f"{limits.max_memory_bytes // (1024 * 1024)} MiB "
                            "memory limit.",
                            current_node_id,
                        )
                        break
                if not handle.process.is_alive():
                    abnormal = (
                        "WorkerCrashed",
                        "The execution process exited unexpectedly.",
                        current_node_id,
                    )
                    break
                continue

            if not isinstance(message, dict):
                continue
            if message.get("kind") == "event":
                try:
                    event = ExecutionEvent(
                        type=EventType(str(message.get("type", ""))),
                        node_id=str(message.get("node_id", "")),
                        data=message.get("data") if isinstance(message.get("data"), dict) else {},
                    )
                except ValueError:
                    continue
                emit_event(event)
                if event.type == EventType.NODE_STARTED:
                    current_node_id = event.node_id
                    node_started_at = time.perf_counter()
                elif event.type in {
                    EventType.NODE_COMPLETED,
                    EventType.NODE_ERROR,
                }:
                    current_node_id = None
                    node_started_at = None
                continue
            terminal = message

        duration_ms = _elapsed_ms(started)
        if terminal is None:
            error_type, message, node_id = abnormal or (
                "WorkerCrashed",
                "The execution process exited unexpectedly.",
                None,
            )
            terminal = {
                "kind": "failed",
                "failure": {
                    "error_type": error_type,
                    "message": message,
                    "node_id": node_id,
                    "line": None,
                    "traceback": "",
                },
            }

        kind = terminal.get("kind")
        if kind == "completed":
            apply_cache(terminal.get("cache"))
            await persist_run(RunStatus.COMPLETED, duration_ms=duration_ms)
            coordinator.finish()
            emit_event(
                ExecutionEvent(
                    type=EventType.EXECUTION_COMPLETED,
                    node_id="",
                    data={
                        "run_id": run_id,
                        "cached": terminal.get("cached", {}),
                        "previews": terminal.get("previews", {}),
                        "duration_ms": duration_ms,
                    },
                )
            )
        elif kind == "cancelled":
            apply_cache(terminal.get("cache"))
            await persist_run(RunStatus.CANCELLED, duration_ms=duration_ms)
            coordinator.finish()
            emit_event(
                ExecutionEvent(
                    type=EventType.EXECUTION_CANCELLED,
                    node_id="",
                    data={"run_id": run_id, "duration_ms": duration_ms},
                )
            )
        else:
            apply_cache(terminal.get("cache"))
            failure = terminal.get("failure")
            payload = failure if isinstance(failure, dict) else {}
            error = failure_from(payload)
            await persist_run(
                RunStatus.FAILED,
                duration_ms=duration_ms,
                error=error,
            )
            coordinator.finish()
            emit_event(
                ExecutionEvent(
                    type=EventType.EXECUTION_FAILED,
                    node_id=error.node_id or "",
                    data={
                        "run_id": run_id,
                        "error_type": error.error_type,
                        "message": error.message,
                        "node_id": error.node_id,
                        "line": error.line,
                        "traceback": error.traceback,
                        "duration_ms": duration_ms,
                    },
                )
            )
    finally:
        if handle is not None:
            await asyncio.to_thread(
                stop_worker,
                handle,
                terminate=abnormal is not None,
            )
        _close_run_log(run_logger, run_handler)
        coordinator.finish()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def mount_static_frontend(target: FastAPI, dist_dir: Path) -> None:
    """Serve a built frontend (SPA) from ``dist_dir`` on ``target``.

    ``/assets/*`` is served from the hashed asset folder; every other
    non-API path falls back to ``index.html`` so client-side routing works.
    """
    resolved = dist_dir.expanduser().resolve()
    index = resolved / "index.html"
    if not index.is_file():
        raise RuntimeError(
            f"Static frontend requested but {index} does not exist — "
            "run `npm run build` in frontend/ first"
        )
    assets_dir = resolved / "assets"
    if assets_dir.is_dir():
        target.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @target.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        if full_path.startswith(("api/", "ws/")):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (resolved / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(resolved):
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    logger.info("serving built frontend from %s", resolved)


_static_dist = os.environ.get("COMPLEXUI_SERVE_DIST")
if _static_dist:
    mount_static_frontend(app, Path(_static_dist))
