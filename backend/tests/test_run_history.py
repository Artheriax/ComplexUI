"""Run-history storage and recorder tests."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import run_history as run_history_module
from app.executor import GraphPlan, clear_cache
from app.main import app, coordinator
from app.run_history import (
    InvalidRunId,
    RunHistoryStore,
    RunNotFound,
    RunRecorder,
)
from app.schemas import (
    EventType,
    ExecutionEvent,
    RunErrorRecord,
    RunNodeRecord,
    RunNodeStatus,
    RunRecord,
    RunStatus,
)
from app.security import code_hash


@pytest.fixture()
def client() -> Iterator[TestClient]:
    clear_cache()
    coordinator.finish()
    with TestClient(app) as test_client:
        yield test_client
    coordinator.finish()
    clear_cache()


def approve(client: TestClient, node_id: str, code: str) -> None:
    response = client.post(
        "/api/approve",
        json={"node_id": node_id, "code_hash": code_hash(code)},
    )
    assert response.status_code == 200


def read_terminal(websocket) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    while True:
        event = websocket.receive_json()
        events.append(event)
        if event["type"] in {
            "EXECUTION_COMPLETED",
            "EXECUTION_FAILED",
            "EXECUTION_CANCELLED",
        }:
            return events


def run_id(value: int) -> str:
    return f"{value:032x}"


def make_record(
    value: int,
    *,
    started_at: str = "2026-10-20T12:00:00.000Z",
) -> RunRecord:
    return RunRecord(
        run_id=run_id(value),
        status=RunStatus.COMPLETED,
        started_at=started_at,
        finished_at="2026-10-20T12:00:01.000Z",
        duration_ms=1000,
        node_count=1,
        completed_count=1,
        cached_count=0,
        order=["node"],
        node_labels={"node": "Node"},
        nodes=[
            RunNodeRecord(
                node_id="node",
                label="Node",
                status=RunNodeStatus.COMPLETED,
                duration_ms=1000,
                previews={"value": 1},
            )
        ],
    )


def test_store_round_trip_and_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_history_module, "RUN_HISTORY_RETENTION", 2)
    store = RunHistoryStore(tmp_path)

    for index in range(3):
        store.save(make_record(index + 1, started_at=f"2026-10-20T12:00:0{index}.000Z"))
        time.sleep(0.002)

    assert store.count() == 2
    listed = store.list_runs()
    assert len(listed) == 2
    assert listed[0].run_id == run_id(3)

    detail = store.get(run_id(3))
    assert detail.nodes[0].previews == {"value": 1}
    assert detail.node_labels == {"node": "Node"}

    store.delete(run_id(3))
    assert store.count() == 1
    with pytest.raises(RunNotFound):
        store.get(run_id(3))


def test_store_skips_corrupt_records_and_rejects_bad_ids(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    store.save(make_record(1))
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / f"{run_id(2)}.json").write_text("{broken", encoding="utf-8")

    assert [item.run_id for item in store.list_runs()] == [run_id(1)]

    for bad_id in ("../outside", "not-hex", "A" * 32):
        with pytest.raises(InvalidRunId):
            store.get(bad_id)
        with pytest.raises(InvalidRunId):
            store.delete(bad_id)


def test_recorder_aggregates_node_status_outputs_and_errors() -> None:
    recorder = RunRecorder(
        run_id(1),
        GraphPlan(order=["a", "b"], schemas={}, untrusted=[]),
        {"a": "Load Iris"},
        ["b"],
    )
    recorder.observe(ExecutionEvent(type=EventType.NODE_STARTED, node_id="a", data={}))
    recorder.observe(
        ExecutionEvent(
            type=EventType.NODE_COMPLETED,
            node_id="a",
            data={
                "cached": False,
                "outputs": {"df": {"rows": 150, "cols": 5}},
                "rich_outputs": [
                    {
                        "kind": "text",
                        "source": "print",
                        "text": "loaded\n",
                        "truncated": False,
                    }
                ],
                "duration_ms": 12.5,
            },
        )
    )
    recorder.observe(ExecutionEvent(type=EventType.NODE_STARTED, node_id="b", data={}))
    recorder.observe(
        ExecutionEvent(
            type=EventType.NODE_ERROR,
            node_id="b",
            data={
                "line": 2,
                "message": "division by zero",
                "error_type": "ZeroDivisionError",
                "traceback": "trace",
            },
        )
    )
    terminal_error = RunErrorRecord(
        message="division by zero",
        error_type="ZeroDivisionError",
        node_id="b",
        line=2,
        traceback="trace",
    )

    record = recorder.finish(RunStatus.FAILED, duration_ms=30, error=terminal_error)

    assert record.node_count == 2
    assert record.completed_count == 1
    assert record.cached_count == 0
    first, second = record.nodes
    assert first.label == "Load Iris"
    assert first.status == RunNodeStatus.COMPLETED
    assert first.duration_ms == 12.5
    assert first.previews == {"df": {"rows": 150, "cols": 5}}
    assert first.rich_outputs[0].kind == "text"
    assert second.label == "b"
    assert second.status == RunNodeStatus.ERROR
    assert second.error is not None
    assert second.error.line == 2

    reparsed = RunRecord.model_validate_json(record.model_dump_json())
    assert reparsed == record


def test_api_persists_completed_run_and_serves_detail(
    client: TestClient,
) -> None:
    code = "value = 1\nreturn value: ANY"
    approve(client, "run-node", code)

    with client.websocket_connect("/ws/execution") as websocket:
        response = client.post(
            "/api/execute",
            json={
                "graph": {"nodes": [{"id": "run-node", "code": code}], "edges": []},
                "node_labels": {"run-node": "My Node"},
            },
        )
        assert response.status_code == 202
        persisted_run_id = response.json()["run_id"]
        read_terminal(websocket)

    listed = client.get("/api/runs")
    assert listed.status_code == 200
    summaries = listed.json()
    assert summaries[0]["run_id"] == persisted_run_id
    assert summaries[0]["status"] == "completed"
    assert summaries[0]["node_count"] == 1

    detail = client.get(f"/api/runs/{persisted_run_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["node_labels"] == {"run-node": "My Node"}
    assert body["nodes"][0]["label"] == "My Node"
    assert body["nodes"][0]["status"] == "completed"
    assert body["nodes"][0]["previews"] == {"value": 1}

    deleted = client.delete(f"/api/runs/{persisted_run_id}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert client.get(f"/api/runs/{persisted_run_id}").status_code == 404
    assert client.delete(f"/api/runs/{'0' * 32}").status_code == 404
    assert client.get(f"/api/runs/{'A' * 32}").status_code == 422


def test_api_records_failed_and_cancelled_runs(client: TestClient) -> None:
    failing_code = "k: int = 1\nvalue = k / 0\nreturn value: ANY"
    approve(client, "boom", failing_code)
    with client.websocket_connect("/ws/execution") as websocket:
        response = client.post(
            "/api/execute",
            json={"graph": {"nodes": [{"id": "boom", "code": failing_code}]}},
        )
        assert response.status_code == 202
        failed_run_id = response.json()["run_id"]
        read_terminal(websocket)

    failed = client.get(f"/api/runs/{failed_run_id}").json()
    assert failed["status"] == "failed"
    assert failed["error_type"] == "ZeroDivisionError"
    assert failed["nodes"][0]["status"] == "error"
    assert failed["nodes"][0]["error"]["line"] == 2

    slow_code = "import time\ntime.sleep(0.35)\nvalue = 1\nreturn value: ANY"
    fast_code = "value = 2\nreturn value: ANY"
    approve(client, "slow", slow_code)
    approve(client, "fast", fast_code)
    with client.websocket_connect("/ws/execution") as websocket:
        response = client.post(
            "/api/execute",
            json={
                "graph": {
                    "nodes": [
                        {"id": "slow", "code": slow_code},
                        {"id": "fast", "code": fast_code},
                    ],
                    "edges": [],
                }
            },
        )
        assert response.status_code == 202
        cancelled_run_id = response.json()["run_id"]
        while True:
            event = websocket.receive_json()
            if event["type"] == "NODE_STARTED" and event["node_id"] == "slow":
                break
        assert client.post("/api/cancel").status_code == 202
        read_terminal(websocket)

    cancelled = client.get(f"/api/runs/{cancelled_run_id}").json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["node_count"] == 2
    assert cancelled["completed_count"] == 1
    assert cancelled["nodes"][1]["status"] == "pending"
