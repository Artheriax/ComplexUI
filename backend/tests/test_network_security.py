"""Tests for Host/Origin validation and request limits."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.executor import clear_cache
from app.main import app, coordinator
from app.network_security import (
    DEFAULT_ALLOWED_HOSTS,
    ENV_ALLOWED_HOSTS,
    RequestSizeLimitMiddleware,
    configured_allowed_hosts,
    origin_is_allowed,
)
from app.schemas import (
    MAX_EXECUTE_TARGETS,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_NODES,
    MAX_NODE_CODE_CHARS,
    MAX_NODE_IDENTIFIER_CHARS,
)


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


def test_allowed_hosts_default_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_ALLOWED_HOSTS, raising=False)

    assert configured_allowed_hosts() == list(DEFAULT_ALLOWED_HOSTS)


def test_allowed_hosts_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ALLOWED_HOSTS, " ui.local, *.example.com ")

    assert configured_allowed_hosts() == ["ui.local", "*.example.com"]


def test_allowed_hosts_empty_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ALLOWED_HOSTS, " ,  , ")

    assert configured_allowed_hosts() == list(DEFAULT_ALLOWED_HOSTS)


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://[::1]:5173",
    ],
)
def test_loopback_origins_are_allowed(origin: str | None) -> None:
    assert origin_is_allowed(origin, DEFAULT_ALLOWED_HOSTS)


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "null",
        "http://evil.example",
        "https://example.com",
        "ftp://localhost",
    ],
)
def test_foreign_origins_are_rejected(origin: str) -> None:
    assert not origin_is_allowed(origin, DEFAULT_ALLOWED_HOSTS)


def test_wildcard_allows_any_origin() -> None:
    assert origin_is_allowed("http://evil.example", ["*"])


def test_unknown_host_is_rejected(client: TestClient) -> None:
    response = client.get("/api/health", headers={"host": "evil.example"})

    assert response.status_code == 400


@pytest.mark.parametrize("host", ["localhost:8000", "127.0.0.1:8000", "[::1]:8000"])
def test_loopback_hosts_are_accepted(client: TestClient, host: str) -> None:
    response = client.get("/api/health", headers={"host": host})

    assert response.status_code == 200


def test_websocket_foreign_origin_is_rejected(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(
            "/ws/execution", headers={"origin": "http://evil.example"}
        ):
            pass

    assert excinfo.value.code == 1008


def test_websocket_loopback_origin_is_accepted(client: TestClient) -> None:
    with client.websocket_connect(
        "/ws/execution", headers={"origin": "http://localhost:5173"}
    ):
        pass


def test_websocket_missing_origin_is_accepted(client: TestClient) -> None:
    with client.websocket_connect("/ws/execution"):
        pass


def test_request_size_limit_rejects_large_content_length() -> None:
    limited = FastAPI()
    limited.add_middleware(RequestSizeLimitMiddleware, max_bytes=32)

    @limited.post("/echo")
    def echo() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(limited) as limited_client:
        accepted = limited_client.post("/echo", content=b"x" * 16)
        rejected = limited_client.post("/echo", content=b"x" * 64)

    assert accepted.status_code == 200
    assert rejected.status_code == 413


def test_request_size_override_uses_longest_matching_prefix() -> None:
    limited = FastAPI()
    limited.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=32,
        overrides={"/upload": 128, "/upload/nested": 16},
    )

    @limited.post("/upload")
    def upload() -> dict[str, bool]:
        return {"ok": True}

    @limited.post("/upload/nested/item")
    def nested() -> dict[str, bool]:
        return {"ok": True}

    @limited.post("/uploader")
    def uploader() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(limited) as limited_client:
        assert limited_client.post("/upload", content=b"x" * 64).status_code == 200
        assert (
            limited_client.post("/upload/nested/item", content=b"x" * 64).status_code
            == 413
        )
        assert limited_client.post("/uploader", content=b"x" * 64).status_code == 413


def test_parse_code_limit(client: TestClient) -> None:
    response = client.post(
        "/api/parse", json={"code": "x" * (MAX_NODE_CODE_CHARS + 1)}
    )

    assert response.status_code == 422


def test_parse_node_id_limit(client: TestClient) -> None:
    response = client.post(
        "/api/parse",
        json={"code": "x = 1", "node_id": "n" * (MAX_NODE_IDENTIFIER_CHARS + 1)},
    )

    assert response.status_code == 422


def test_graph_node_limit(client: TestClient) -> None:
    nodes = [
        {"id": f"n{index}", "code": "x = 1"}
        for index in range(MAX_GRAPH_NODES + 1)
    ]

    response = client.post("/api/execute", json={"graph": {"nodes": nodes}})

    assert response.status_code == 422


def test_graph_edge_limit(client: TestClient) -> None:
    edges = [
        {
            "source_node_id": "a",
            "source_handle": "out",
            "target_node_id": "b",
            "target_handle": "in",
        }
        for _ in range(MAX_GRAPH_EDGES + 1)
    ]

    response = client.post("/api/execute", json={"graph": {"edges": edges}})

    assert response.status_code == 422


def test_execute_target_limit(client: TestClient) -> None:
    response = client.post(
        "/api/execute",
        json={
            "graph": {"nodes": [], "edges": []},
            "targets": ["n" * 8 for _ in range(MAX_EXECUTE_TARGETS + 1)],
        },
    )

    assert response.status_code == 422


def test_approve_node_id_limit(client: TestClient) -> None:
    response = client.post(
        "/api/approve",
        json={
            "node_id": "n" * (MAX_NODE_IDENTIFIER_CHARS + 1),
            "code_hash": "a" * 64,
        },
    )

    assert response.status_code == 422
