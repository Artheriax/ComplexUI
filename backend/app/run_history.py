"""Durable, read-only history for terminal workflow executions."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .executor import GraphPlan
from .schemas import (
    MAX_RUN_RICH_OUTPUT_BYTES,
    RUN_HISTORY_RETENTION,
    EventType,
    ExecutionEvent,
    RichOutput,
    RunDetail,
    RunErrorRecord,
    RunNodeRecord,
    RunNodeStatus,
    RunRecord,
    RunStatus,
    RunSummary,
)
from .storage import utc_timestamp, write_json_atomic
from .workspace import workspace_root

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_RECORD_SUFFIX = ".json"
_RICH_OUTPUTS_ADAPTER = TypeAdapter(list[RichOutput])
_SUMMARY_EXCLUDES = {
    "nodes",
    "order",
    "targets",
    "node_labels",
    "error",
    "schema_version",
}


def _summary(record: RunRecord) -> RunSummary:
    return RunSummary(**record.model_dump(exclude=_SUMMARY_EXCLUDES))


class RunHistoryError(Exception):
    """Base class for run-history storage failures."""


class RunNotFound(RunHistoryError):
    """The requested terminal run does not exist."""


class InvalidRunId(RunHistoryError):
    """A run id is malformed or could escape the run directory."""


class InvalidRunRecord(RunHistoryError):
    """A stored run document is not valid."""


def _timestamp() -> str:
    return utc_timestamp(milliseconds=True)


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise InvalidRunId(f"Invalid run id: {run_id!r}")
    return run_id


class RunHistoryStore:
    """Thread-safe JSON storage for retained terminal run records."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = ((root or workspace_root()) / "runs").expanduser()
        self._lock = threading.Lock()

    def _path(self, run_id: str) -> Path:
        return self.root / f"{_validate_run_id(run_id)}{_RECORD_SUFFIX}"

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        write_json_atomic(path, payload)

    def _read_record(self, path: Path) -> RunRecord:
        if not path.is_file():
            raise RunNotFound(f"Run record not found: {path.name}")
        try:
            return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            message = exc.errors()[0].get("msg", "validation failed")
            raise InvalidRunRecord(f"Invalid run record {path.name}: {message}") from exc
        except OSError as exc:
            raise RunHistoryError(f"Could not read run record {path.name}: {exc}") from exc

    def save(self, record: RunRecord) -> RunSummary:
        """Persist a terminal run, then prune records beyond the retention cap."""
        path = self._path(record.run_id)
        with self._lock:
            self._write_json(path, record.model_dump(mode="json"))
            self._prune()
        return _summary(record)

    def _prune(self) -> None:
        if not self.root.is_dir():
            return
        paths = sorted(
            self.root.glob(f"*{_RECORD_SUFFIX}"),
            key=lambda candidate: candidate.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in paths[RUN_HISTORY_RETENTION:]:
            stale.unlink(missing_ok=True)

    def list_runs(self, limit: int = RUN_HISTORY_RETENTION) -> list[RunSummary]:
        """Return retained run summaries, newest first."""
        if limit <= 0 or not self.root.is_dir():
            return []
        summaries: list[RunSummary] = []
        for path in self.root.glob(f"*{_RECORD_SUFFIX}"):
            try:
                record = self._read_record(path)
            except (InvalidRunRecord, RunNotFound, RunHistoryError, OSError):
                continue
            summaries.append(_summary(record))
        summaries.sort(key=lambda item: (item.started_at, item.run_id), reverse=True)
        return summaries[:limit]

    def get(self, run_id: str) -> RunDetail:
        """Load one retained run record."""
        return RunDetail(
            **self._read_record(self._path(run_id)).model_dump(exclude={"schema_version"})
        )

    def delete(self, run_id: str) -> None:
        """Delete one retained run record."""
        path = self._path(run_id)
        with self._lock:
            if not path.is_file():
                raise RunNotFound(f"Run record not found: {run_id}")
            path.unlink(missing_ok=True)

    def saved_paths(self) -> set[str]:
        """Return saved file paths recorded by retained runs."""
        paths: set[str] = set()
        if not self.root.is_dir():
            return paths
        for path in self.root.glob(f"*{_RECORD_SUFFIX}"):
            try:
                record = self._read_record(path)
            except (InvalidRunRecord, RunNotFound, RunHistoryError, OSError):
                continue
            for node in record.nodes:
                value = node.previews.get("saved_path")
                if isinstance(value, str) and value.strip():
                    paths.add(value)
        return paths

    def count(self) -> int:
        if not self.root.is_dir():
            return 0
        return len(list(self.root.glob(f"*{_RECORD_SUFFIX}")))


class RunRecorder:
    """Aggregate execution telemetry into a durable run record."""

    def __init__(
        self,
        run_id: str,
        plan: GraphPlan,
        node_labels: dict[str, str] | None,
        targets: Sequence[str] | None,
    ) -> None:
        self.run_id = run_id
        self.started_at = _timestamp()
        self._order = list(plan.order)
        self._targets = list(targets) if targets else None
        self._labels = {
            node_id: (node_labels or {}).get(node_id, node_id) for node_id in self._order
        }
        self._nodes: dict[str, RunNodeRecord] = {
            node_id: RunNodeRecord(
                node_id=node_id,
                label=self._labels[node_id],
            )
            for node_id in self._order
        }
        self._node_started: dict[str, float] = {}
        self._rich_output_bytes = 0

    def observe(self, event: ExecutionEvent) -> None:
        """Collect per-node status and bounded outputs from a live event."""
        if event.type not in {
            EventType.NODE_STARTED,
            EventType.NODE_COMPLETED,
            EventType.NODE_ERROR,
        }:
            return
        node = self._nodes.get(event.node_id)
        if node is None:
            return
        if event.type == EventType.NODE_STARTED:
            self._node_started[event.node_id] = time.perf_counter()
            return
        duration = event.data.get("duration_ms")
        if not isinstance(duration, (int, float)):
            started = self._node_started.get(event.node_id)
            duration = (
                round((time.perf_counter() - started) * 1000, 1) if started is not None else None
            )
        if event.type == EventType.NODE_COMPLETED:
            previews = event.data.get("outputs")
            node.status = RunNodeStatus.COMPLETED
            node.cached = event.data.get("cached") is True
            node.duration_ms = max(0.0, float(duration or 0.0))
            node.previews = previews if isinstance(previews, dict) else {}
            rich, truncated = self._bounded_rich_outputs(event.data.get("rich_outputs"))
            node.rich_outputs = rich
            node.rich_outputs_truncated = truncated
            metrics = event.data.get("metrics")
            node.metrics = metrics if isinstance(metrics, dict) else None
            return
        line = event.data.get("line")
        node.status = RunNodeStatus.ERROR
        node.duration_ms = max(0.0, float(duration)) if duration is not None else None
        node.error = RunErrorRecord(
            message=str(event.data.get("message", "Node failed")),
            error_type=str(event.data.get("error_type", "Error")),
            node_id=event.node_id,
            line=line if isinstance(line, int) else None,
            traceback=str(event.data.get("traceback", "")),
        )
        rich, truncated = self._bounded_rich_outputs(event.data.get("rich_outputs"))
        node.rich_outputs = rich
        node.rich_outputs_truncated = truncated
        metrics = event.data.get("metrics")
        node.metrics = metrics if isinstance(metrics, dict) else None

    def _bounded_rich_outputs(self, payload: Any) -> tuple[list[RichOutput], bool]:
        if not isinstance(payload, list) or not payload:
            return [], False
        try:
            outputs = _RICH_OUTPUTS_ADAPTER.validate_python(payload)
            size = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            return [], True
        if self._rich_output_bytes + size > MAX_RUN_RICH_OUTPUT_BYTES:
            return [], True
        self._rich_output_bytes += size
        return outputs, False

    def finish(
        self,
        status: RunStatus,
        *,
        duration_ms: float,
        error: RunErrorRecord | None,
    ) -> RunRecord:
        """Build the final record after a terminal execution event."""
        return RunRecord(
            run_id=self.run_id,
            status=status,
            started_at=self.started_at,
            finished_at=_timestamp(),
            duration_ms=max(0.0, duration_ms),
            node_count=len(self._nodes),
            completed_count=sum(
                node.status == RunNodeStatus.COMPLETED for node in self._nodes.values()
            ),
            cached_count=sum(node.cached for node in self._nodes.values()),
            error_type=error.error_type if error is not None else None,
            order=self._order,
            targets=self._targets,
            node_labels=self._labels,
            nodes=list(self._nodes.values()),
            error=error,
        )


@lru_cache(maxsize=1)
def get_run_history_store() -> RunHistoryStore:
    """Return the process-wide run-history store."""
    return RunHistoryStore()
