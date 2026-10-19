"""FastAPI entry point: REST endpoints and WebSocket execution streaming.

``POST /api/execute`` accepts a workflow graph, runs a synchronous pre-flight
(parse + wiring + trust validation) and then starts the execution in a worker
thread so the event loop stays responsive. Telemetry flows through
:class:`ConnectionManager` to every client connected to ``/ws/execution``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .executor import (
    ExecutionCancelled,
    ExecutionResult,
    GraphError,
    GraphPlan,
    NodeExecutionError,
    execute_graph,
    validate_graph,
)
from .logging_setup import configure_logging
from .network_security import (
    RequestSizeLimitMiddleware,
    configured_allowed_hosts,
    origin_is_allowed,
)
from .schemas import (
    ApproveRequest,
    ApproveResponse,
    EventType,
    ExecuteRequest,
    ExecuteResponse,
    ExecutionEvent,
    ExecutionStatusResponse,
    NodeSchema,
    ParseRequest,
    TemplateInfo,
    WorkflowGraph,
)
from .security import UntrustedNodeError, get_security_registry
from .template_catalog import template_meta
from .transpiler import TranspileError, parse_node
from .workflow_catalog import (
    WORKFLOW_CATEGORIES,
    WorkflowTemplateInfo,
    get_workflows,
)

__all__ = ["app", "coordinator", "manager"]

configure_logging(os.environ.get("COMPLEXUI_LOG_LEVEL", "INFO"))
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


def _log_event(event: ExecutionEvent) -> None:
    """Narrate every execution telemetry event on the console."""
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
        logger.info(
            "run %s started: %d node(s)", short_run, node_count, extra=extra
        )
        logger.debug("run %s order: %s", short_run, order, extra=extra)
    elif event.type is EventType.NODE_STARTED:
        logger.info("  [start] %s", event.node_id, extra=extra)
    elif event.type is EventType.INSTALLING_DEPS:
        dependencies = data.get("dependencies") or []
        extra["dependencies"] = dependencies
        logger.info(
            "  [deps ] %s installing: %s",
            event.node_id,
            ", ".join(str(dep) for dep in dependencies),
            extra=extra,
        )
    elif event.type is EventType.NODE_COMPLETED:
        cached = bool(data.get("cached"))
        extra["cached"] = cached
        if cached:
            logger.info("  [done ] %s (cached)", event.node_id, extra=extra)
        else:
            duration = data.get("duration_ms")
            if isinstance(duration, (int, float)):
                extra["duration_ms"] = duration
            logger.info(
                "  [done ] %s in %sms", event.node_id, duration, extra=extra
            )
    elif event.type is EventType.NODE_ERROR:
        extra["error_type"] = data.get("error_type")
        extra["line"] = data.get("line")
        logger.error(
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
            logger.info(
                "run %s completed in %sms: %d node(s), %d cached",
                short_run,
                duration,
                len(cached_map),
                cached_count,
                extra=extra,
            )
        else:
            logger.info(
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
        logger.error(
            "run %s failed%s: %s",
            short_run,
            f" at {failed_node}" if failed_node else "",
            data.get("message"),
            extra=extra,
        )
        traceback_text = data.get("traceback")
        if isinstance(traceback_text, str) and traceback_text:
            logger.debug(
                "run %s traceback:\n%s", short_run, traceback_text, extra=extra
            )
    elif event.type is EventType.EXECUTION_CANCELLED:
        duration = data.get("duration_ms")
        if isinstance(duration, (int, float)):
            extra["duration_ms"] = duration
        logger.warning("run %s cancelled", short_run, extra=extra)
    else:
        logger.debug(
            "event %s node=%s", event.type.value, event.node_id, extra=extra
        )


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
app.add_middleware(RequestSizeLimitMiddleware)

manager = ConnectionManager()
coordinator = ExecutionCoordinator()
_background_tasks: set[asyncio.Task[None]] = set()

logger.info(
    "ComplexUI backend ready: %d official template(s), allowed hosts: %s",
    len(get_security_registry().templates()),
    ", ".join(_ALLOWED_HOSTS),
)
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
    """Return the official templates with code, hashes and socket schemas."""
    registry = get_security_registry()
    templates: list[TemplateInfo] = []
    for record in registry.templates():
        meta = template_meta(record.name)
        templates.append(
            TemplateInfo(
                name=record.name,
                title=meta.title,
                category=meta.category,
                description=meta.description,
                hidden=meta.hidden,
                code=record.code,
                code_hash=record.code_hash,
                node_schema=parse_node(record.code, record.name),
            )
        )
    return templates


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
    return {
        "status": "ok",
        "version": app.version,
        "templates": len(get_security_registry().templates()),
        "workflows": len(get_workflows()),
    }


@app.get("/api/status", response_model=ExecutionStatusResponse)
def status_endpoint() -> ExecutionStatusResponse:
    """Report whether an execution is running (used to resync after reconnects)."""
    run_id = coordinator.run_id
    return ExecutionStatusResponse(
        status="running" if run_id is not None else "idle",
        run_id=run_id,
    )


@app.post("/api/approve", response_model=ApproveResponse)
def approve_endpoint(payload: ApproveRequest) -> ApproveResponse:
    """Approve custom/modified node code for the current session."""
    get_security_registry().approve(payload.node_id, payload.code_hash)
    logger.info(
        "approved node '%s' (hash %s…)", payload.node_id, payload.code_hash[:12]
    )
    return ApproveResponse(
        status="approved",
        node_id=payload.node_id,
        code_hash=payload.code_hash,
    )


@app.post("/api/execute", status_code=202, response_model=ExecuteResponse)
async def execute_endpoint(payload: ExecuteRequest) -> ExecuteResponse:
    """Pre-flight a workflow graph and start async execution + WS streaming."""
    if coordinator.running:
        logger.warning("execute rejected: a run is already in progress")
        raise HTTPException(
            status_code=409, detail="An execution is already running"
        )

    try:
        plan = validate_graph(payload.graph, targets=payload.targets)
    except TranspileError as exc:
        logger.error(
            "execute rejected: syntax error in node '%s' (line %d)",
            exc.schema.node_id,
            exc.schema.syntax_errors[0].line
            if exc.schema.syntax_errors
            else 0,
        )
        raise HTTPException(
            status_code=422, detail=exc.schema.model_dump(mode="json")
        ) from exc
    except GraphError as exc:
        logger.error("execute rejected: %s", exc)
        raise HTTPException(
            status_code=422, detail={"message": str(exc)}
        ) from exc

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
                    "Workflow contains untrusted code; approve the listed "
                    "nodes before execution"
                ),
                "untrusted_nodes": [
                    {"node_id": node_id, "code_hash": hash_value}
                    for node_id, hash_value in plan.untrusted
                ],
            },
        )

    run = coordinator.try_begin()
    if run is None:  # unreachable: no await since the check above
        raise HTTPException(
            status_code=409, detail="An execution is already running"
        )
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
        _run_execution(payload.graph, plan, run_id, stop_event)
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
        logger.debug(
            "telemetry client disconnected (%d left)", manager.connection_count
        )


async def _run_execution(
    graph: WorkflowGraph,
    plan: GraphPlan,
    run_id: str,
    stop_event: threading.Event,
) -> None:
    started = time.perf_counter()

    def emit(event: ExecutionEvent) -> None:
        event.data.setdefault("run_id", run_id)
        manager.emit_from_thread(event)

    try:
        manager.emit(
            ExecutionEvent(
                type=EventType.EXECUTION_STARTED,
                node_id="",
                data={"run_id": run_id, "order": plan.order},
            )
        )
        result: ExecutionResult = await asyncio.to_thread(
            execute_graph,
            graph,
            plan=plan,
            emit=emit,
            should_stop=stop_event.is_set,
        )
        manager.emit(
            ExecutionEvent(
                type=EventType.EXECUTION_COMPLETED,
                node_id="",
                data={
                    "run_id": run_id,
                    "cached": result.cached,
                    "previews": result.previews,
                    "duration_ms": _elapsed_ms(started),
                },
            )
        )
    except ExecutionCancelled:
        manager.emit(
            ExecutionEvent(
                type=EventType.EXECUTION_CANCELLED,
                node_id="",
                data={"run_id": run_id, "duration_ms": _elapsed_ms(started)},
            )
        )
    except Exception as exc:
        manager.emit(
            ExecutionEvent(
                type=EventType.EXECUTION_FAILED,
                node_id=_failed_node_id(exc),
                data={
                    "run_id": run_id,
                    **_failure_payload(exc),
                    "duration_ms": _elapsed_ms(started),
                },
            )
        )
    finally:
        coordinator.finish()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _failed_node_id(exc: Exception) -> str:
    if isinstance(exc, (NodeExecutionError, UntrustedNodeError)):
        return exc.node_id
    return ""


def _failure_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, NodeExecutionError):
        return {
            "error_type": exc.error_type,
            "message": exc.message,
            "node_id": exc.node_id,
            "line": exc.line,
            "traceback": exc.traceback_text,
        }
    if isinstance(exc, UntrustedNodeError):
        return {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "node_id": exc.node_id,
            "line": None,
            "traceback": "",
        }
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "node_id": None,
        "line": None,
        "traceback": "",
    }


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
