"""Workspace dataset upload, preview, deletion and File-widget resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import datasets as datasets_module
from app.datasets import get_dataset_store
from app.executor import NodeExecutionError, clear_cache, execute_graph
from app.main import app
from app.schemas import (
    MAX_DATASET_NAME_CHARS,
    EventType,
    GraphNode,
    WorkflowGraph,
)
from app.security import SecurityRegistry, code_hash
from app.transpiler import parse_node


@pytest.fixture()
def upload_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("COMPLEXUI_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("COMPLEXUI_INPUT_DIR", str(tmp_path / "inputs"))
    get_dataset_store.cache_clear()
    with TestClient(app) as client:
        yield client
    get_dataset_store.cache_clear()


def test_default_input_dir_is_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COMPLEXUI_INPUT_DIR", raising=False)
    expected = Path(__file__).resolve().parents[2] / "inputs"
    assert datasets_module.DatasetStore().uploads_dir == expected


def upload_csv(
    client: TestClient,
    *,
    name: str = "iris.csv",
    rows: int = 3,
) -> dict[str, object]:
    body = "sepal_length,species\n" + "\n".join(f"{4.5 + index},setosa" for index in range(rows))
    response = client.post(
        "/api/uploads",
        files={"file": (name, body.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()["dataset"]


def test_upload_list_preview_and_delete(upload_client: TestClient) -> None:
    dataset = upload_csv(upload_client, rows=25)

    assert dataset["id"] == "iris.csv"
    assert dataset["name"] == "iris.csv"
    assert dataset["path"] == "inputs/iris.csv"
    assert dataset["format"] == "CSV"
    assert dataset["size_bytes"] > 0

    listed = upload_client.get("/api/uploads")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == ["iris.csv"]

    preview = upload_client.get("/api/uploads/iris.csv/preview")
    assert preview.status_code == 200
    body = preview.json()
    assert body["columns"] == ["sepal_length", "species"]
    assert len(body["rows"]) == 20
    assert body["truncated"] is True
    assert body["rows"][0] == [4.5, "setosa"]

    deleted = upload_client.delete("/api/uploads/iris.csv")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert upload_client.get("/api/uploads").json() == []
    assert upload_client.get("/api/uploads/iris.csv").status_code == 404


def test_upload_sanitizes_names_and_avoids_collisions(
    upload_client: TestClient,
) -> None:
    first = upload_csv(upload_client, name="../Unsafe Name.csv")
    second = upload_csv(upload_client, name="Unsafe Name.csv")

    assert first["id"] == "Unsafe Name.csv"
    assert second["id"] == "Unsafe Name-2.csv"

    long_name = f"{'a' * MAX_DATASET_NAME_CHARS}.csv"
    long_first = upload_csv(upload_client, name=long_name)
    long_second = upload_csv(upload_client, name=long_name)
    assert len(str(long_first["id"])) == MAX_DATASET_NAME_CHARS
    assert len(str(long_second["id"])) == MAX_DATASET_NAME_CHARS
    assert long_first["id"] != long_second["id"]

    invalid = upload_client.post(
        "/api/uploads",
        files={"file": ("notes.txt", b"not a dataset", "text/plain")},
    )
    assert invalid.status_code == 422

    empty = upload_client.post(
        "/api/uploads",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert empty.status_code == 422

    traversed = upload_client.get("/api/uploads/%2E%2E%2Fsecret.csv")
    assert traversed.status_code in {404, 422}


def test_upload_size_cap(upload_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets_module, "MAX_DATASET_BYTES", 32)
    response = upload_client.post(
        "/api/uploads",
        files={"file": ("large.csv", b"x" * 128, "text/csv")},
    )
    assert response.status_code == 413
    assert upload_client.get("/api/uploads").json() == []


def test_invalid_dataset_preview_is_reported(upload_client: TestClient) -> None:
    response = upload_client.post(
        "/api/uploads",
        files={"file": ("broken.parquet", b"not parquet", "application/octet-stream")},
    )
    assert response.status_code == 200
    preview = upload_client.get("/api/uploads/broken.parquet/preview")
    assert preview.status_code == 422
    assert "Could not preview" in preview.json()["detail"]


def test_file_widget_is_parsed_and_resolved_from_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_dir = tmp_path / "inputs"
    upload_dir.mkdir(parents=True)
    monkeypatch.setenv("COMPLEXUI_INPUT_DIR", str(upload_dir))
    get_dataset_store.cache_clear()
    (upload_dir / "sample.csv").write_text("a\n1\n", encoding="utf-8")

    code = (
        "from pathlib import Path\n"
        'path: File = ""\n'
        "exists = Path(path).is_file()\n"
        "return exists: ANY\n"
    )
    schema = parse_node(code, "load")
    assert schema.widgets[0].name == "path"
    assert schema.widgets[0].data_type == "File"

    node = GraphNode(
        id="load_file_widget",
        code=code,
        widget_values={"path": "inputs/sample.csv"},
    )
    registry = SecurityRegistry()
    registry.approve(node.id, code_hash(node.code))
    clear_cache()
    try:
        result = execute_graph(WorkflowGraph(nodes=[node]), security=registry, allow_installs=False)
    finally:
        clear_cache()
    assert result.outputs_by_name[node.id]["exists"] is True


def test_file_widget_rejects_paths_outside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.csv"
    outside.write_text("secret\n1\n", encoding="utf-8")
    monkeypatch.setenv("COMPLEXUI_WORKSPACE", str(workspace))

    code = (
        "from pathlib import Path\n"
        'path: File = ""\n'
        "exists = Path(path).is_file()\n"
        "return exists: ANY\n"
    )
    node = GraphNode(
        id="blocked_file",
        code=code,
        widget_values={"path": str(outside)},
    )
    registry = SecurityRegistry()
    registry.approve(node.id, code_hash(node.code))
    events: list[object] = []

    with pytest.raises(NodeExecutionError) as excinfo:
        execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            emit=events.append,
        )

    assert excinfo.value.error_type == "FileAccessError"
    assert any(getattr(event, "type", None) is EventType.NODE_ERROR for event in events)


def test_uploaded_dataset_executes_through_loader_template(
    upload_client: TestClient,
) -> None:
    dataset = upload_csv(upload_client, rows=3)
    template_path = Path(__file__).resolve().parents[1] / "app" / "templates" / "load_csv.py"
    code = template_path.read_text(encoding="utf-8")
    node = GraphNode(
        id="load_uploaded_csv",
        code=code,
        widget_values={"path": dataset["path"]},
    )
    registry = SecurityRegistry()
    registry.approve(node.id, code_hash(node.code))
    clear_cache()
    try:
        result = execute_graph(WorkflowGraph(nodes=[node]), security=registry, allow_installs=False)
    finally:
        clear_cache()

    frame = result.outputs_by_name[node.id]["df"]
    assert frame.height == 3
    assert frame.columns == ["sepal_length", "species"]


def test_loader_templates_use_file_widgets() -> None:
    templates = Path(__file__).resolve().parents[1] / "app" / "templates"
    for name in ("load_csv", "load_parquet", "load_excel", "load_json"):
        schema = parse_node((templates / f"{name}.py").read_text(encoding="utf-8"))
        path_widget = next(widget for widget in schema.widgets if widget.name == "path")
        assert path_widget.data_type == "File"
