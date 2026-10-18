"""Tests for serving the built frontend bundle (prod mode)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import mount_static_frontend

INDEX_HTML = "<!doctype html><title>ComplexUI</title>"


@pytest.fixture()
def dist_dir(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (assets / "app-abc123.js").write_text("console.log('hi')", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    return dist


@pytest.fixture()
def static_client(dist_dir: Path) -> TestClient:
    target = FastAPI()

    @target.get("/api/ping")
    def ping() -> dict[str, str]:
        return {"pong": "ok"}

    mount_static_frontend(target, dist_dir)
    return TestClient(target)


def test_api_routes_and_assets_take_precedence(static_client: TestClient) -> None:
    ping = static_client.get("/api/ping")
    assert ping.status_code == 200
    assert ping.json() == {"pong": "ok"}

    asset = static_client.get("/assets/app-abc123.js")
    assert asset.status_code == 200
    assert asset.text == "console.log('hi')"


def test_root_and_spa_routes_serve_index(static_client: TestClient) -> None:
    for path in ("/", "/workflows/42", "/editor"):
        response = static_client.get(path)
        assert response.status_code == 200
        assert INDEX_HTML in response.text
    assert static_client.get("/").headers["cache-control"] == "no-cache"


def test_unknown_api_paths_return_404_not_index(static_client: TestClient) -> None:
    response = static_client.get("/api/definitely-missing")
    assert response.status_code == 404
    assert INDEX_HTML not in response.text


def test_traversal_outside_dist_falls_back_to_index(static_client: TestClient) -> None:
    response = static_client.get("/..%2Fsecret.txt")
    assert "TOP SECRET" not in response.text


def test_mount_raises_when_index_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="index.html"):
        mount_static_frontend(FastAPI(), tmp_path / "nowhere")
