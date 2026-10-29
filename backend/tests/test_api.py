"""API tests for the FastAPI server and WebSocket streaming (M3)."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.execution_host import (
    ENV_MAX_MEMORY_MB,
    ENV_NODE_TIMEOUT,
    ENV_RUN_TIMEOUT,
)
from app.executor import clear_cache
from app.main import app, coordinator
from app.security import code_hash, get_security_registry
from app.workflow_catalog import get_workflows


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
        response = client.post("/api/execute", json={"graph": graph, "targets": targets})
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
        {
            "name": "k",
            "data_type": "int",
            "default": 5,
            "options": [],
            "model_type": None,
            "model_select": "file",
            "model_required": False,
            "model_include": [],
            "visible_when": [],
        }
    ]
    assert body["outputs"][0]["name"] == "k"
    assert body["outputs"][0]["data_type"] == "ANY"
    assert body["syntax_errors"] == []


def test_parse_endpoint_returns_optional_input_sockets(client: TestClient) -> None:
    code = (
        "from typing import Optional\n"
        "image1: Optional[IMAGE] = None\n"
        "image2: IMAGE | None = None\n"
        "return image1: ANY, image2: ANY\n"
    )

    response = client.post("/api/parse", json={"code": code})

    assert response.status_code == 200
    assert response.json()["inputs"] == [
        {"name": "image1", "data_type": "IMAGE", "optional": True},
        {"name": "image2", "data_type": "IMAGE", "optional": True},
    ]


def test_parse_endpoint_returns_model_selector_metadata(client: TestClient) -> None:
    code = (
        'model: ModelSelector = ModelSelector(\n'
        '    type="vae", select="any", required=True, include=["diffusion"]\n'
        ")\n"
        "return model: STRING\n"
    )

    response = client.post("/api/parse", json={"code": code})

    assert response.status_code == 200
    widget = response.json()["widgets"][0]
    assert widget["model_type"] == "vae"
    assert widget["model_select"] == "any"
    assert widget["model_required"] is True
    assert widget["model_include"] == ["diffusion"]


def test_parse_endpoint_returns_literal_choices(client: TestClient) -> None:
    code = (
        "from typing import Literal\n"
        "level: Literal['low', 'medium', 'high'] = 'low'\n"
        "return level: STRING\n"
    )

    response = client.post("/api/parse", json={"code": code})

    assert response.status_code == 200
    assert response.json()["widgets"] == [
        {
            "name": "level",
            "data_type": "str",
            "default": "low",
                "options": ["low", "medium", "high"],
                "model_type": None,
                "model_select": "file",
                "model_required": False,
                "model_include": [],
                "visible_when": [],
            }
        ]


def test_parse_endpoint_returns_widget_visibility(client: TestClient) -> None:
    code = (
        "from typing import Literal\n"
        "mode: Literal['a', 'b'] = 'a'\n"
        "value: int = ShowIf(default=5, mode='b')\n"
        "return value: INT\n"
    )

    response = client.post("/api/parse", json={"code": code})

    assert response.status_code == 200
    widgets = response.json()["widgets"]
    assert widgets[1]["visible_when"] == [
        {"widget": "mode", "values": ["b"]}
    ]


def test_parse_endpoint_reports_syntax_errors(client: TestClient) -> None:
    response = client.post("/api/parse", json={"code": "x = 1\ndef broken(:\n"})

    assert response.status_code == 200
    body = response.json()
    assert body["syntax_errors"][0]["line"] == 2


def test_templates_endpoint_lists_official_templates(client: TestClient) -> None:
    response = client.get("/api/templates")

    assert response.status_code == 200
    templates = {item["name"]: item for item in response.json()}
    official = get_security_registry().templates()
    assert len(templates) == len(official)
    assert {"load_iris", "train_test_split", "knn_train", "evaluate_metrics"} <= set(templates)
    load = templates["load_iris"]
    assert load["title"] == "Load Iris Dataset"
    assert load["category"] == "Data Loaders"
    assert "150 iris rows" in load["description"]
    assert load["hidden"] is True
    assert templates["load_csv"]["hidden"] is False
    assert templates["knn_train"]["hidden"] is False
    assert load["code_hash"] == code_hash(load["code"])
    assert get_security_registry().is_official(load["code"])
    assert load["node_schema"]["outputs"] == [{"name": "df", "data_type": "DATAFRAME"}]
    assert templates["train_test_split"]["category"] == "Preprocessing"
    assert templates["tensor_dataset"]["category"] == "PyTorch"
    knn = templates["knn_train"]
    assert [widget["name"] for widget in knn["node_schema"]["widgets"]] == ["n_neighbors"]


def test_websocket_runtime_backend_is_installed() -> None:
    import importlib.util

    assert any(importlib.util.find_spec(name) is not None for name in ("websockets", "wsproto")), (
        "uvicorn needs 'uvicorn[standard]', 'websockets' or 'wsproto' to serve /ws/execution"
    )


def test_health_endpoint_reports_readiness(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMPLEXUI_ACCELERATOR", "cpu")
    monkeypatch.setattr("app.main.nvidia_available", lambda: False)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == app.version
    assert body["templates"] == len(get_security_registry().templates())
    assert body["workflows"] == len(get_workflows())
    assert body["accelerator"] == "cpu"
    assert body["effective_accelerator"] == "cpu"
    assert "torch" in body
    assert body["torch_cuda"] in {True, False}
    assert body["nvidia"] is False
    assert set(body["limits"]) == {
        "node_timeout_seconds",
        "run_timeout_seconds",
        "max_memory_mb",
    }
    assert set(body["optimizations"]) == {
        "attention",
        "precision",
        "memory",
        "compile",
    }
    assert body["optimizations"]["attention"]["configured"] in {
        "auto",
        "sdpa",
        "sdpa-flash",
        "sdpa-mem-efficient",
        "sdpa-cudnn",
        "sdpa-math",
        "xformers",
        "flash-attn",
        "flash-attn4",
        "sageattention",
        "comfy_kitchen_sol",
        "comfy_kitchen_int8",
    }
    formats = body["quantization"]["formats"]
    assert {"int8_w8a8", "int8_convrot_w4a4", "gguf", "nvfp4"} <= set(formats)
    assert formats["int8_w8a8"]["runtime"] == "comfy_kitchen"
    assert formats["int8_w8a8"]["installed"] in {True, False}


def test_status_endpoint_reports_idle_and_running(client: TestClient) -> None:
    idle = client.get("/api/status")
    assert idle.status_code == 200
    assert idle.json() == {"status": "idle", "run_id": None}

    run = coordinator.try_begin()
    assert run is not None
    running = client.get("/api/status")
    assert running.status_code == 200
    assert running.json() == {"status": "running", "run_id": run[0]}


def test_workflows_endpoint_lists_validated_examples(client: TestClient) -> None:
    response = client.get("/api/workflows")

    assert response.status_code == 200
    body = response.json()
    assert {item["name"] for item in body} == set(get_workflows())
    assert body[0]["category"] == "Getting Started"
    for item in body:
        assert item["nodes"]
        assert item["edges"]
        assert {"name", "title", "description"} <= set(item)


def test_export_endpoint_returns_python_script(client: TestClient) -> None:
    response = client.post("/api/export", json={"graph": _chain_payload(), "format": "python"})

    assert response.status_code == 200
    body = response.json()
    assert body["filename"].endswith(".py")
    assert body["media_type"] == "text/x-python"
    content = body["content"]
    assert "# /// script" in content
    assert '"scikit-learn>=1.5"' in content
    assert "def __node_exec_load" in content
    assert "def __node_exec_evaluate" in content
    assert "def main()" in content


def test_export_endpoint_returns_notebook(client: TestClient) -> None:
    response = client.post(
        "/api/export",
        json={"graph": _chain_payload(), "format": "notebook"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"].endswith(".ipynb")
    assert body["media_type"] == "application/x-ipynb+json"
    notebook = json.loads(body["content"])
    assert notebook["nbformat"] == 4
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert any(
        "__node_exec_load" in "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_export_endpoint_rejects_invalid_wiring(client: TestClient) -> None:
    templates = _templates()
    response = client.post(
        "/api/export",
        json={
            "graph": {
                "nodes": [{"id": "knn", "code": templates["knn_train"]}],
                "edges": [],
            }
        },
    )

    assert response.status_code == 422
    assert "not connected" in response.json()["detail"]["message"]


def test_export_endpoint_rejects_empty_workflow(client: TestClient) -> None:
    response = client.post("/api/export", json={"graph": {}})

    assert response.status_code == 422
    assert "no nodes" in response.json()["detail"]["message"]


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

    bad = client.post("/api/approve", json={"node_id": "x", "code_hash": "not-a-hash"})
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

    started_nodes = [event["node_id"] for event in events if event["type"] == "NODE_STARTED"]
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


def test_execute_streams_rich_outputs_and_replays_cache(client: TestClient) -> None:
    code = 'print("hello from node")\ndisplay("shown value")\nvalue = 1\nreturn value: ANY'
    node = {"id": "rich-api", "code": code}
    approved = client.post(
        "/api/approve",
        json={"node_id": "rich-api", "code_hash": code_hash(code)},
    )
    assert approved.status_code == 200

    status, events = _run_graph(client, {"nodes": [node], "edges": []})
    assert status == 202
    completed = next(event for event in events if event["type"] == "NODE_COMPLETED")
    assert [item["kind"] for item in completed["data"]["rich_outputs"]] == [
        "text",
        "text",
    ]
    assert completed["data"]["rich_outputs"][0]["text"] == "hello from node\n"
    assert completed["data"]["rich_outputs"][1]["text"] == "shown value"
    assert "rich_outputs" not in events[-1]["data"]

    status, replay_events = _run_graph(client, {"nodes": [node], "edges": []})
    assert status == 202
    replay_completed = next(event for event in replay_events if event["type"] == "NODE_COMPLETED")
    assert replay_completed["data"]["cached"] is True
    assert replay_completed["data"]["rich_outputs"] == completed["data"]["rich_outputs"]


def test_node_progress_streams_over_websocket(client: TestClient) -> None:
    code = "report_progress(0.5, 'half')\nvalue = 1\nreturn value: ANY"
    node = {"id": "progress-api", "code": code}
    approved = client.post(
        "/api/approve",
        json={"node_id": "progress-api", "code_hash": code_hash(code)},
    )
    assert approved.status_code == 200

    status, events = _run_graph(client, {"nodes": [node], "edges": []})
    assert status == 202
    progress = [
        event for event in events if event["type"] == "NODE_PROGRESS"
    ]
    assert len(progress) == 1
    assert progress[0]["node_id"] == "progress-api"
    assert progress[0]["data"]["progress"] == 0.5
    assert progress[0]["data"]["message"] == "half"
    assert progress[0]["data"]["run_id"]


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

    blocked = client.post("/api/execute", json={"graph": {"nodes": [node], "edges": []}})
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

    response = client.post("/api/execute", json={"graph": {"nodes": [node], "edges": []}})

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


def test_execute_with_targets_ignores_out_of_scope_broken_node(
    client: TestClient,
) -> None:
    payload = _chain_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes.append({"id": "broken", "code": "def broken(:\n"})

    status, events = _run_graph(client, payload, targets=["knn"])

    assert status == 202
    assert events[0]["data"]["order"] == ["load", "split", "knn"]
    assert events[-1]["type"] == "EXECUTION_COMPLETED"


def test_execute_with_unknown_target_returns_422(client: TestClient) -> None:
    response = client.post("/api/execute", json={"graph": _chain_payload(), "targets": ["ghost"]})

    assert response.status_code == 422
    assert "Unknown target" in response.json()["detail"]["message"]


def test_node_timeout_terminates_the_worker(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = "import time\ntime.sleep(2)\nvalue = 1\nreturn value: ANY"
    approved = client.post("/api/approve", json={"node_id": "slow", "code_hash": code_hash(code)})
    assert approved.status_code == 200
    monkeypatch.setenv(ENV_NODE_TIMEOUT, "0.25")
    monkeypatch.setenv(ENV_RUN_TIMEOUT, "10")
    monkeypatch.setenv(ENV_MAX_MEMORY_MB, "0")

    status, events = _run_graph(client, {"nodes": [{"id": "slow", "code": code}]})

    assert status == 202
    assert events[-1]["type"] == "EXECUTION_FAILED"
    assert events[-1]["data"]["error_type"] == "NodeTimeout"
    assert events[-1]["data"]["node_id"] == "slow"


def test_run_timeout_terminates_the_worker(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = "import time\ntime.sleep(2)\nvalue = 1\nreturn value: ANY"
    approved = client.post("/api/approve", json={"node_id": "slow", "code_hash": code_hash(code)})
    assert approved.status_code == 200
    monkeypatch.setenv(ENV_NODE_TIMEOUT, "0")
    monkeypatch.setenv(ENV_RUN_TIMEOUT, "0.25")
    monkeypatch.setenv(ENV_MAX_MEMORY_MB, "0")

    status, events = _run_graph(client, {"nodes": [{"id": "slow", "code": code}]})

    assert status == 202
    assert events[-1]["type"] == "EXECUTION_FAILED"
    assert events[-1]["data"]["error_type"] == "RunTimeout"


def test_crashed_worker_reports_failure_and_recovers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    crash_code = "import os\nos._exit(7)\nvalue = 1\nreturn value: ANY"
    approved = client.post(
        "/api/approve",
        json={"node_id": "crash", "code_hash": code_hash(crash_code)},
    )
    assert approved.status_code == 200
    # This test only needs the crash to surface promptly; the recovery chain
    # imports pandas/sklearn from a cold worker, so leave ample headroom for
    # slow/loaded machines instead of turning machine speed into a failure.
    monkeypatch.setenv(ENV_NODE_TIMEOUT, "60")
    monkeypatch.setenv(ENV_RUN_TIMEOUT, "120")
    monkeypatch.setenv(ENV_MAX_MEMORY_MB, "0")

    status, events = _run_graph(client, {"nodes": [{"id": "crash", "code": crash_code}]})

    assert status == 202
    assert events[-1]["type"] == "EXECUTION_FAILED"
    assert events[-1]["data"]["error_type"] == "WorkerCrashed"

    status, recovered = _run_graph(client, _chain_payload())
    assert status == 202
    assert recovered[-1]["type"] == "EXECUTION_COMPLETED"


def test_cancel_endpoint_returns_409_when_idle(client: TestClient) -> None:
    response = client.post("/api/cancel")

    assert response.status_code == 409
    assert response.json()["detail"] == "No execution is running"


def test_cancel_interrupts_between_nodes(client: TestClient) -> None:
    slow_code = "import time\n\ndelay: float = 0.3\ntime.sleep(delay)\nreturn delay: ANY"
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


def test_connection_manager_broadcasts_and_drops_failed_clients() -> None:
    import asyncio
    from typing import cast

    from starlette.websockets import WebSocket

    from app.main import ConnectionManager
    from app.schemas import EventType, ExecutionEvent

    class FakeWebSocket:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.received = 0

        async def send_json(self, payload: dict[str, object]) -> None:
            assert payload["type"] == "EXECUTION_STARTED"
            self.received += 1
            if self.fail:
                raise RuntimeError("client closed")

    async def scenario() -> tuple[FakeWebSocket, FakeWebSocket]:
        manager = ConnectionManager()
        good = FakeWebSocket()
        bad = FakeWebSocket(fail=True)
        manager._connections = {
            cast(WebSocket, good),
            cast(WebSocket, bad),
        }
        await manager._broadcast(
            ExecutionEvent(
                type=EventType.EXECUTION_STARTED,
                node_id="",
                data={"run_id": "run-1", "order": []},
            )
        )
        assert manager._connections == {good}
        return good, bad

    good, bad = asyncio.run(scenario())
    assert good.received == 1
    assert bad.received == 1


def test_connection_manager_queue_drops_oldest_event_when_full() -> None:
    import asyncio

    from app.main import ConnectionManager
    from app.schemas import EventType, ExecutionEvent

    async def scenario() -> list[int]:
        manager = ConnectionManager()
        manager._queue = asyncio.Queue(maxsize=2)
        for index in range(3):
            manager.emit(
                ExecutionEvent(
                    type=EventType.NODE_STARTED,
                    node_id="n1",
                    data={"index": index},
                )
            )
        queue = manager._queue
        assert queue is not None
        return [int(queue.get_nowait().data["index"]) for _ in range(2)]

    assert asyncio.run(scenario()) == [1, 2]


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


def test_coordinator_stale_finish_does_not_clear_newer_run() -> None:
    from app.main import ExecutionCoordinator

    local = ExecutionCoordinator()
    first = local.try_begin()
    assert first is not None
    local.finish(first[0])

    second = local.try_begin()
    assert second is not None
    # A slow cleanup path from the previous run must not clear this run.
    local.finish(first[0])
    assert local.run_id == second[0]
    assert local.cancel() is True
    assert second[1].is_set() is True

    local.finish(second[0])
    assert local.running is False


def test_coordinator_finish_begin_race_keeps_stop_event() -> None:
    import threading

    from app.main import ExecutionCoordinator

    def race_once() -> None:
        local = ExecutionCoordinator()
        assert local.try_begin() is not None
        barrier = threading.Barrier(2)

        def finish() -> None:
            barrier.wait()
            local.finish()

        thread = threading.Thread(target=finish)
        thread.start()
        barrier.wait()
        begun = local.try_begin()
        thread.join()
        if begun is not None:
            assert local.cancel() is True
            local.finish()

    for _ in range(200):
        race_once()
