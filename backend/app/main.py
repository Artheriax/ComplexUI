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
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
from .schemas import (
    ApproveRequest,
    ApproveResponse,
    EventType,
    ExecuteRequest,
    ExecuteResponse,
    ExecutionEvent,
    NodeSchema,
    ParseRequest,
    TemplateInfo,
    WorkflowGraph,
)
from .security import UntrustedNodeError, get_security_registry
from .transpiler import TranspileError, parse_node

__all__ = ["app", "coordinator", "manager"]

configure_logging(os.environ.get("COMPLEXUI_LOG_LEVEL", "INFO"))
logger = logging.getLogger("complexui.server")

_SEND_TIMEOUT_SECONDS = 2.0

_TEMPLATE_TITLES = {
    "load_iris": "Load Iris Dataset",
    "train_test_split": "Train / Test Split",
    "knn_train": "KNN Classifier",
    "evaluate_metrics": "Evaluate Classifier",
    "tensor_dataset": "Synthetic Tensor Dataset",
    "mlp_train": "MLP Trainer (Torch)",
    "evaluate_torch": "Evaluate Torch Model",
}


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
        self._run_id: str | None = None
        self._stop_event: threading.Event | None = None

    @property
    def running(self) -> bool:
        return self._run_id is not None

    def try_begin(self) -> tuple[str, threading.Event] | None:
        if self._run_id is not None:
            return None
        self._run_id = uuid4().hex
        self._stop_event = threading.Event()
        return self._run_id, self._stop_event

    def cancel(self) -> bool:
        """Request cancellation of the running execution (False if idle)."""
        if self._stop_event is None:
            return False
        self._stop_event.set()
        return True

    def finish(self) -> None:
        self._run_id = None
        self._stop_event = None


def _log_event(event: ExecutionEvent) -> None:
    """Narrate every execution telemetry event on the console."""
    data = event.data
    short_run = str(data.get("run_id", ""))[:12]
    if event.type is EventType.EXECUTION_STARTED:
        logger.info("run %s started: order=%s", short_run, data.get("order", []))
    elif event.type is EventType.NODE_STARTED:
        logger.info("  [start] %s", event.node_id)
    elif event.type is EventType.INSTALLING_DEPS:
        dependencies = data.get("dependencies") or []
        logger.info(
            "  [deps ] %s installing: %s",
            event.node_id,
            ", ".join(str(dep) for dep in dependencies),
        )
    elif event.type is EventType.NODE_COMPLETED:
        if data.get("cached"):
            logger.info("  [done ] %s (cached)", event.node_id)
        else:
            logger.info(
                "  [done ] %s in %sms", event.node_id, data.get("duration_ms")
            )
    elif event.type is EventType.NODE_ERROR:
        logger.error(
            "  [error] %s: %s (line %s)",
            event.node_id,
            data.get("error_type"),
            data.get("line"),
        )
    elif event.type is EventType.EXECUTION_COMPLETED:
        cached_map = data.get("cached") or {}
        cached_count = sum(1 for value in cached_map.values() if value)
        logger.info(
            "run %s completed: %d node(s), %d cached",
            short_run,
            len(cached_map),
            cached_count,
        )
    elif event.type is EventType.EXECUTION_FAILED:
        logger.error("run %s failed: %s", short_run, data.get("message"))
    elif event.type is EventType.EXECUTION_CANCELLED:
        logger.warning("run %s cancelled", short_run)
    else:
        logger.debug("event %s node=%s", event.type.value, event.node_id)


app = FastAPI(title="ComplexUI Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()
coordinator = ExecutionCoordinator()
_background_tasks: set[asyncio.Task[None]] = set()

logger.info(
    "ComplexUI backend ready: %d official template(s), CORS for localhost:5173",
    len(get_security_registry().templates()),
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
    return [
        TemplateInfo(
            name=record.name,
            title=_TEMPLATE_TITLES.get(
                record.name, record.name.replace("_", " ").title()
            ),
            code=record.code,
            code_hash=record.code_hash,
            node_schema=parse_node(record.code, record.name),
        )
        for record in registry.templates()
    ]


@app.get("/api/health")
def health_endpoint() -> dict[str, object]:
    """Liveness probe used by the launcher and external tools."""
    return {
        "status": "ok",
        "version": app.version,
        "templates": len(get_security_registry().templates()),
    }


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
        _run_execution(payload.graph, plan, run_id, stop_event, targets)
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
    targets: list[str] | None = None,
) -> None:
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
            emit=manager.emit_from_thread,
            should_stop=stop_event.is_set,
            targets=targets,
        )
        manager.emit(
            ExecutionEvent(
                type=EventType.EXECUTION_COMPLETED,
                node_id="",
                data={
                    "run_id": run_id,
                    "cached": result.cached,
                    "previews": result.previews,
                },
            )
        )
    except ExecutionCancelled:
        manager.emit(
            ExecutionEvent(
                type=EventType.EXECUTION_CANCELLED,
                node_id="",
                data={"run_id": run_id},
            )
        )
    except Exception as exc:
        manager.emit(
            ExecutionEvent(
                type=EventType.EXECUTION_FAILED,
                node_id=_failed_node_id(exc),
                data={"run_id": run_id, **_failure_payload(exc)},
            )
        )
    finally:
        coordinator.finish()


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
