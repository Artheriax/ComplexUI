"""API tests for the FastAPI server and WebSocket streaming (M3)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.executor import clear_cache
from app.main import app, coordinator
from app.security import code_hash, get_security_registry


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    clear_cache()
    coordinator.finish()
    yield
    coordinator.finish()
    clear_cache()


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def _templates() -> dict[str, str]:
    return {record.name: record.code for record in get_security_registry().templates()}


def _chain_payload(n_neighbors: int = 5) -> dict[str, object]:
    templates = _templates()
    return {
        "nodes": [
            {"id": "load", "code": templates["load_iris"]},
            {"id": "split", "code": templates["train_test_split"]},
            {
                "id": "knn",
                "code": templates["knn_train"],
                "widget_values": {"n_neighbors": n_neighbors},
            },
            {"id": "evaluate", "code": templates["evaluate_metrics"]},
        ],
        "edges": [
            {
                "source_node_id": "load",
                "source_handle": "df",
                "target_node_id": "split",
                "target_handle": "df",
            },
            {
                "source_node_id": "split",
                "source_handle": "train",
                "target_node_id": "knn",
                "target_handle": "train",
            },
            {
                "source_node_id": "knn",
                "source_handle": "model",
                "target_node_id": "evaluate",
                "target_handle": "model",
            },
            {
                "source_node_id": "split",
                "source_handle": "test",
                "target_node_id": "evaluate",
                "target_handle": "test",
            },
        ],
    }


def _read_until_terminal(websocket) -> list[dict]:
    events: list[dict] = []
    while True:
        event = websocket.receive_json()
        events.append(event)
        if event["type"] in {
            "EXECUTION_COMPLETED",
            "EXECUTION_FAILED",
            "EXECUTION_CANCELLED",
        }:
            return events


def _run_graph(
    client: TestClient,
    graph: dict[str, object],
    targets: list[str] | None = None,
) -> tuple[int, list[dict]]:
    with client.websocket_connect("/ws/execution") as websocket:
        response = client.post(
            "/api/execute", json={"graph": graph, "targets": targets}
        )
        if response.status_code != 202:
            return response.status_code, []
        events = _read_until_terminal(websocket)
    return response.status_code, events


def test_parse_endpoint_returns_node_schema(client: TestClient) -> None:
    code = (
        '# /// script\n# dependencies = ["scikit-learn>=1.5"]\n# ///\n'
        "train: DATASET\nk: int = 5\nreturn k: ANY\n"
    )

    response = client.post("/api/parse", json={"code": code})

    assert response.status_code == 200
    body = response.json()
    assert body["dependencies"] == ["scikit-learn>=1.5"]
    assert [socket["name"] for socket in body["inputs"]] == ["train"]
    assert body["widgets"] == [
        {"name": "k", "data_type": "int", "default": 5}
    ]
    assert body["outputs"][0]["name"] == "k"
    assert body["outputs"][0]["data_type"] == "ANY"
    assert body["syntax_errors"] == []


def test_parse_endpoint_reports_syntax_errors(client: TestClient) -> None:
    response = client.post("/api/parse", json={"code": "x = 1\ndef broken(:\n"})

    assert response.status_code == 200
    body = response.json()
    assert body["syntax_errors"][0]["line"] == 2


def test_templates_endpoint_lists_official_templates(client: TestClient) -> None:
    response = client.get("/api/templates")

    assert response.status_code == 200
    templates = {item["name"]: item for item in response.json()}
    assert set(templates) == {
        "load_iris",
        "train_test_split",
        "knn_train",
        "evaluate_metrics",
        "tensor_dataset",
        "mlp_train",
        "evaluate_torch",
    }
    load = templates["load_iris"]
    assert load["title"] == "Load Iris Dataset"
    assert load["code_hash"] == code_hash(load["code"])
    assert get_security_registry().is_official(load["code"])
    assert load["node_schema"]["outputs"] == [
        {"name": "df", "data_type": "DATAFRAME"}
    ]
    knn = templates["knn_train"]
    assert [widget["name"] for widget in knn["node_schema"]["widgets"]] == [
        "n_neighbors"
    ]


def test_websocket_runtime_backend_is_installed() -> None:
    import importlib.util

    assert any(
        importlib.util.find_spec(name) is not None
        for name in ("websockets", "wsproto")
    ), "uvicorn needs 'uvicorn[standard]', 'websockets' or 'wsproto' to serve /ws/execution"


def test_health_endpoint_reports_readiness(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == app.version
    assert body["templates"] == 7


def test_approve_endpoint_records_session_approval(client: TestClient) -> None:
    digest = "a" * 64

    response = client.post(
        "/api/approve", json={"node_id": "approve-test-node", "code_hash": digest}
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "approved",
        "node_id": "approve-test-node",
        "code_hash": digest,
    }
    assert get_security_registry().is_approved("approve-test-node", digest)

    bad = client.post(
        "/api/approve", json={"node_id": "x", "code_hash": "not-a-hash"}
    )
    assert bad.status_code == 422


def test_execute_streams_events_and_completes(client: TestClient) -> None:
    with client.websocket_connect("/ws/execution") as websocket:
        response = client.post("/api/execute", json={"graph": _chain_payload()})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        events = _read_until_terminal(websocket)

    types = [event["type"] for event in events]
    assert types[0] == "EXECUTION_STARTED"
    assert events[0]["data"]["run_id"] == run_id
    assert events[0]["data"]["order"] == ["load", "split", "knn", "evaluate"]
    assert types.count("NODE_STARTED") == 4
    assert types.count("NODE_COMPLETED") == 4
    assert types[-1] == "EXECUTION_COMPLETED"

    started_nodes = [
        event["node_id"] for event in events if event["type"] == "NODE_STARTED"
    ]
    assert started_nodes == ["load", "split", "knn", "evaluate"]

    final = events[-1]["data"]
    assert final["run_id"] == run_id
    assert final["cached"] == {
        "load": False,
        "split": False,
        "knn": False,
        "evaluate": False,
    }
    assert final["previews"]["evaluate"]["metrics"]["accuracy"] > 0.90


def test_second_execution_is_served_from_cache(client: TestClient) -> None:
    status, _ = _run_graph(client, _chain_payload())
    assert status == 202

    status, events = _run_graph(client, _chain_payload())
    assert status == 202

    types = [event["type"] for event in events]
    assert "NODE_STARTED" not in types
    assert types[-1] == "EXECUTION_COMPLETED"
    assert events[-1]["data"]["cached"] == {
        "load": True,
        "split": True,
        "knn": True,
        "evaluate": True,
    }


def test_node_error_streams_raw_line_numbers(client: TestClient) -> None:
    node = {"id": "boom", "code": "k: int = 1\nvalue = k / 0\nreturn value: ANY"}
    approved = client.post(
        "/api/approve",
        json={"node_id": "boom", "code_hash": code_hash(node["code"])},
    )
    assert approved.status_code == 200

    status, events = _run_graph(client, {"nodes": [node], "edges": []})
    assert status == 202

    error = next(event for event in events if event["type"] == "NODE_ERROR")
    assert error["node_id"] == "boom"
    assert error["data"]["line"] == 2
    assert error["data"]["error_type"] == "ZeroDivisionError"
    assert '<node:boom>", line 2' in error["data"]["traceback"]

    failed = events[-1]
    assert failed["type"] == "EXECUTION_FAILED"
    assert failed["data"]["node_id"] == "boom"
    assert failed["data"]["line"] == 2
    assert failed["data"]["error_type"] == "ZeroDivisionError"


def test_untrusted_node_flow_returns_403_then_approve(client: TestClient) -> None:
    modified_code = _templates()["load_iris"] + "\n# modified\n"
    node = {"id": "custom_load", "code": modified_code}

    blocked = client.post(
        "/api/execute", json={"graph": {"nodes": [node], "edges": []}}
    )
    assert blocked.status_code == 403
    detail = blocked.json()["detail"]
    assert detail["untrusted_nodes"] == [
        {"node_id": "custom_load", "code_hash": code_hash(modified_code)}
    ]

    approved = client.post(
        "/api/approve",
        json={
            "node_id": "custom_load",
            "code_hash": code_hash(modified_code),
        },
    )
    assert approved.status_code == 200

    status, events = _run_graph(client, {"nodes": [node], "edges": []})
    assert status == 202
    assert events[-1]["type"] == "EXECUTION_COMPLETED"
    assert events[-1]["data"]["previews"]["custom_load"]["df"]["rows"] == 150


def test_invalid_wiring_returns_422(client: TestClient) -> None:
    node = {"id": "knn", "code": _templates()["knn_train"]}

    response = client.post(
        "/api/execute", json={"graph": {"nodes": [node], "edges": []}}
    )

    assert response.status_code == 422
    assert "train" in response.json()["detail"]["message"]


def test_syntax_error_graph_returns_422_with_schema(client: TestClient) -> None:
    response = client.post(
        "/api/execute",
        json={
            "graph": {
                "nodes": [{"id": "bad", "code": "def broken(:\n"}],
                "edges": [],
            }
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["syntax_errors"][0]["line"] == 1


def test_conflict_returns_409_while_running(client: TestClient) -> None:
    assert coordinator.try_begin() is not None
    try:
        response = client.post("/api/execute", json={"graph": _chain_payload()})
        assert response.status_code == 409
        assert "already running" in response.json()["detail"]
    finally:
        coordinator.finish()


def test_malformed_payload_returns_422(client: TestClient) -> None:
    response = client.post("/api/execute", json={"nodes": "nope", "edges": []})
    assert response.status_code == 422


def test_execute_with_targets_scopes_execution(client: TestClient) -> None:
    status, events = _run_graph(client, _chain_payload(), targets=["knn"])
    assert status == 202

    assert events[0]["type"] == "EXECUTION_STARTED"
    assert events[0]["data"]["order"] == ["load", "split", "knn"]
    assert events[-1]["type"] == "EXECUTION_COMPLETED"
    assert events[-1]["data"]["cached"] == {
        "load": False,
        "split": False,
        "knn": False,
    }
    assert "evaluate" not in events[-1]["data"]["previews"]


def test_execute_with_unknown_target_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/execute", json={"graph": _chain_payload(), "targets": ["ghost"]}
    )

    assert response.status_code == 422
    assert "Unknown target" in response.json()["detail"]["message"]


def test_cancel_endpoint_returns_409_when_idle(client: TestClient) -> None:
    response = client.post("/api/cancel")

    assert response.status_code == 409
    assert response.json()["detail"] == "No execution is running"


def test_cancel_interrupts_between_nodes(client: TestClient) -> None:
    slow_code = (
        "import time\n\ndelay: float = 0.3\ntime.sleep(delay)\nreturn delay: ANY"
    )
    fast_code = "value: ANY\nreturn value: ANY"
    for node_id, code in (("slow", slow_code), ("fast", fast_code)):
        approved = client.post(
            "/api/approve", json={"node_id": node_id, "code_hash": code_hash(code)}
        )
        assert approved.status_code == 200

    graph = {
        "nodes": [
            {"id": "slow", "code": slow_code},
            {"id": "fast", "code": fast_code},
        ],
        "edges": [
            {
                "source_node_id": "slow",
                "source_handle": "delay",
                "target_node_id": "fast",
                "target_handle": "value",
            }
        ],
    }

    with client.websocket_connect("/ws/execution") as websocket:
        response = client.post("/api/execute", json={"graph": graph})
        assert response.status_code == 202
        events: list[dict] = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] == "NODE_STARTED" and event["node_id"] == "slow":
                break

        cancel = client.post("/api/cancel")
        assert cancel.status_code == 202
        assert cancel.json() == {"status": "cancelling"}

        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] in {
                "EXECUTION_CANCELLED",
                "EXECUTION_COMPLETED",
                "EXECUTION_FAILED",
            }:
                break

    types = [event["type"] for event in events]
    assert events[-1]["type"] == "EXECUTION_CANCELLED"
    assert events[-1]["data"]["run_id"] == response.json()["run_id"]
    started = [e["node_id"] for e in events if e["type"] == "NODE_STARTED"]
    assert started == ["slow"]
    completed = [e["node_id"] for e in events if e["type"] == "NODE_COMPLETED"]
    assert completed == ["slow"]
    assert "EXECUTION_FAILED" not in types


def test_coordinator_cancel_semantics() -> None:
    from app.main import ExecutionCoordinator

    local = ExecutionCoordinator()
    assert local.cancel() is False
    assert local.running is False

    begun = local.try_begin()
    assert begun is not None
    run_id, stop_event = begun
    assert run_id
    assert local.running is True
    assert local.try_begin() is None
    assert stop_event.is_set() is False

    assert local.cancel() is True
    assert stop_event.is_set() is True

    local.finish()
    assert local.running is False
    assert local.cancel() is False
