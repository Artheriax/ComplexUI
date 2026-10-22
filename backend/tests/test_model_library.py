"""Repo-root local model library tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.executor import (
    ModelSelectionError,
    clear_cache,
    execute_graph,
    validate_graph,
)
from app.exporter import export_python
from app.main import app
from app.models import (
    InvalidModelPath,
    ModelNotFound,
    ModelStore,
    get_model_store,
)
from app.schemas import GraphNode, WorkflowGraph
from app.security import SecurityRegistry, code_hash


@pytest.fixture()
def model_store(tmp_path: Path) -> ModelStore:
    get_model_store.cache_clear()
    return ModelStore(tmp_path / "library")


def test_default_type_folders_are_created(model_store: ModelStore) -> None:
    for name in ("yolo", "torchvision", "classification", "diffusion", "unet", "vae"):
        assert (model_store.root / name).is_dir()
    assert [item.name for item in model_store.list_types()] == [
        "yolo",
        "torchvision",
        "classification",
        "diffusion",
        "unet",
        "vae",
    ]
    assert all(item.count == 0 for item in model_store.list_types())


def test_custom_types_and_supported_files_are_listed(model_store: ModelStore) -> None:
    custom = model_store.root / "my-models"
    custom.mkdir()
    (custom / "candidate.onnx").write_bytes(b"onnx")
    (custom / "ignored.pkl").write_bytes(b"pickle")
    (model_store.root / "yolo" / "detector.pt").write_bytes(b"pt")
    (model_store.root / "torchvision" / "weights.safetensors").write_bytes(b"safe")

    types = {item.name: item for item in model_store.list_types()}
    assert types["my-models"].count == 1
    assert types["my-models"].default is False

    models = model_store.list_models("my-models")
    assert [model.name for model in models] == ["candidate.onnx"]
    assert models[0].path == "models/my-models/candidate.onnx"
    assert models[0].format == "ONNX"
    assert models[0].executable is False

    yolo = model_store.list_models("yolo")[0]
    assert yolo.executable is True
    assert yolo.fingerprint.startswith(f"{yolo.size_bytes}:")


def test_resolve_accepts_repo_and_short_paths(model_store: ModelStore) -> None:
    target = model_store.root / "yolo" / "detector.pt"
    target.write_bytes(b"pt")

    assert model_store.resolve("yolo", "models/yolo/detector.pt") == target
    assert model_store.resolve("yolo", "detector.pt") == target
    assert model_store.describe("yolo", "detector.pt").model_type == "yolo"


def test_resolve_rejects_invalid_paths(model_store: ModelStore) -> None:
    (model_store.root / "yolo" / "detector.pt").write_bytes(b"pt")

    with pytest.raises((InvalidModelPath, ModelNotFound)):
        model_store.resolve("torchvision", "detector.pt")
    with pytest.raises(InvalidModelPath):
        model_store.resolve("yolo", "../detector.pt")
    with pytest.raises(InvalidModelPath):
        model_store.resolve("yolo", "models/torchvision/detector.pt")
    with pytest.raises(InvalidModelPath):
        model_store.resolve("yolo", "not-a-model.txt")
    with pytest.raises(ModelNotFound):
        model_store.resolve("yolo", "missing.pt")


def test_resolve_rejects_symlinked_model_files(model_store: ModelStore) -> None:
    target = model_store.root / "yolo" / "target.pt"
    target.write_bytes(b"pt")
    link = model_store.root / "yolo" / "link.pt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks are not available on this platform")

    with pytest.raises(ModelNotFound):
        model_store.resolve("yolo", "link.pt")


def test_fingerprint_tracks_file_changes(model_store: ModelStore) -> None:
    target = model_store.root / "yolo" / "detector.pt"
    target.write_bytes(b"one")
    first = model_store.describe("yolo", "detector.pt")
    os.utime(target, ns=(first.mtime_ns + 1_000_000, first.mtime_ns + 1_000_000))
    second = model_store.describe("yolo", "detector.pt")

    assert first.fingerprint != second.fingerprint


def test_models_api_lists_types_and_files() -> None:
    store = get_model_store()
    (store.root / "custom").mkdir()
    (store.root / "custom" / "model.onnx").write_bytes(b"onnx")

    with TestClient(app) as client:
        types = client.get("/api/models/types")
        assert types.status_code == 200
        assert "custom" in {item["name"] for item in types.json()}

        models = client.get("/api/models", params={"type": "custom"})
        assert models.status_code == 200
        assert models.json()[0]["path"] == "models/custom/model.onnx"

        invalid = client.get("/api/models", params={"type": "../escape"})
        assert invalid.status_code == 422


def test_executor_resolves_model_selector_and_invalidates_cache() -> None:
    store = get_model_store()
    target = store.root / "yolo" / "custom.pt"
    target.write_bytes(b"first")
    code = (
        'local_model: ModelSelector = ModelSelector(type="yolo")\n'
        "return local_model: STRING"
    )
    registry = SecurityRegistry()
    registry.approve("selector", code_hash(code))
    node = GraphNode(
        id="selector",
        code=code,
        widget_values={"local_model": "models/yolo/custom.pt"},
    )
    graph = WorkflowGraph(nodes=[node], edges=[])

    clear_cache()
    try:
        first = execute_graph(graph, security=registry, allow_installs=False)
        assert first.outputs_by_name["selector"]["local_model"] == str(target)
        assert first.cached["selector"] is False

        second = execute_graph(graph, security=registry, allow_installs=False)
        assert second.cached["selector"] is True

        target.write_bytes(b"second")
        os.utime(target, ns=(target.stat().st_mtime_ns + 1_000_000,) * 2)
        third = execute_graph(graph, security=registry, allow_installs=False)
        assert third.cached["selector"] is False
    finally:
        clear_cache()


def test_preflight_blocks_stale_model_selection() -> None:
    code = (
        'local_model: ModelSelector = ModelSelector(type="yolo")\n'
        "return local_model: STRING"
    )
    node = GraphNode(
        id="selector",
        code=code,
        widget_values={"local_model": "models/yolo/missing.pt"},
    )

    with pytest.raises(ModelSelectionError) as excinfo:
        validate_graph(WorkflowGraph(nodes=[node], edges=[]))
    assert excinfo.value.issues[0]["node_id"] == "selector"
    assert excinfo.value.issues[0]["widget"] == "local_model"
    assert "missing.pt" in excinfo.value.issues[0]["path"]


def test_execute_api_reports_structured_model_selection_error() -> None:
    code = (
        'local_model: ModelSelector = ModelSelector(type="yolo")\n'
        "return local_model: STRING"
    )
    payload = {
        "graph": {
            "nodes": [
                {
                    "id": "selector",
                    "code": code,
                    "widget_values": {"local_model": "models/yolo/missing.pt"},
                }
            ],
            "edges": [],
        }
    }

    with TestClient(app) as client:
        response = client.post("/api/execute", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["type"] == "ModelSelection"
    assert detail["models"][0]["widget"] == "local_model"


def test_export_resolves_model_selector_to_absolute_path() -> None:
    store = get_model_store()
    target = store.root / "yolo" / "exported.pt"
    target.write_bytes(b"model")
    code = (
        'local_model: ModelSelector = ModelSelector(type="yolo")\n'
        "return local_model: STRING"
    )
    graph = WorkflowGraph(
        nodes=[
            GraphNode(
                id="selector",
                code=code,
                widget_values={"local_model": "models/yolo/exported.pt"},
            )
        ],
        edges=[],
    )

    exported = export_python(graph)
    assert repr(str(target.resolve())) in exported.content


def test_empty_model_selector_uses_builtin_fallback() -> None:
    code = (
        'local_model: ModelSelector = ModelSelector(type="yolo")\n'
        "return local_model: STRING"
    )
    registry = SecurityRegistry()
    registry.approve("selector", code_hash(code))
    graph = WorkflowGraph(nodes=[GraphNode(id="selector", code=code)], edges=[])

    clear_cache()
    try:
        result = execute_graph(graph, security=registry, allow_installs=False)
    finally:
        clear_cache()
    assert result.outputs_by_name["selector"]["local_model"] == ""
