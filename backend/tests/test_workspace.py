"""Workspace project storage, version snapshots and recovery sessions."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import workspace as workspace_module
from app.main import app
from app.schemas import SavedWorkflowDocument, WorkspaceSessionRequest
from app.workspace import (
    InvalidProjectDocument,
    InvalidProjectSlug,
    ProjectAlreadyExists,
    ProjectNotFound,
    ProjectStore,
    get_project_store,
    slugify_project_name,
)


def make_workflow(*node_ids: str, name: str | None = None) -> SavedWorkflowDocument:
    return SavedWorkflowDocument.model_validate(
        {
            "name": name,
            "version": 1,
            "savedAt": "2026-10-20T12:00:00Z",
            "nodes": [
                {
                    "id": node_id,
                    "label": node_id.title(),
                    "position": {"x": index * 40, "y": index * 20},
                    "code": f"value = {index}\nreturn value: ANY\n",
                    "widgetValues": {"k": index},
                    "templateName": None,
                    "color": "default",
                }
                for index, node_id in enumerate(node_ids)
            ],
            "edges": [],
        }
    )


@pytest.fixture()
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore(tmp_path / "workspace")


def test_project_lifecycle_and_snapshots(store: ProjectStore) -> None:
    workflow = make_workflow("load", "train")
    summary, created = store.create_project("Iris Demo", workflow)

    assert created is True
    assert summary.slug == "iris-demo"
    assert summary.name == "Iris Demo"
    assert summary.node_count == 2
    assert summary.version_count == 0
    assert (store.projects_dir / "iris-demo.cuigraph.json").is_file()

    with pytest.raises(ProjectAlreadyExists):
        store.create_project("Iris Demo", workflow)

    updated = store.update_project("iris-demo", "Iris Demo", make_workflow("load"))
    assert updated.node_count == 1
    assert updated.version_count == 1

    versions = store.list_versions("iris-demo")
    assert len(versions) == 1
    restored = store.get_version("iris-demo", versions[0].version)
    assert len(restored.workflow.nodes) == 2

    detail = store.get_project("iris-demo")
    assert detail.name == "Iris Demo"
    assert detail.workflow.nodes[0].widget_values == {"k": 0}


def make_frame_workflow() -> SavedWorkflowDocument:
    return SavedWorkflowDocument.model_validate(
        {
            "version": 1,
            "savedAt": "2026-10-20T12:00:00Z",
            "nodes": [
                {
                    "id": "frame",
                    "label": "Preprocessing",
                    "position": {"x": 100, "y": 100},
                    "code": "",
                    "widgetValues": {},
                    "templateName": None,
                    "color": "cyan",
                    "isFrame": True,
                    "width": 640,
                    "height": 420,
                },
                {
                    "id": "child",
                    "label": "Load Iris",
                    "position": {"x": 24, "y": 40},
                    "code": "return 1: ANY\n",
                    "widgetValues": {},
                    "templateName": None,
                    "color": "default",
                    "parentId": "frame",
                },
            ],
            "edges": [],
        }
    )


def test_frame_documents_round_trip(store: ProjectStore) -> None:
    _, created = store.create_project("Framed", make_frame_workflow())
    assert created is True

    detail = store.get_project("framed")
    frame, child = detail.workflow.nodes
    assert frame.is_frame is True
    assert frame.width == 640
    assert frame.height == 420
    assert child.parent_id == "frame"

    raw = (store.projects_dir / "framed.cuigraph.json").read_text(encoding="utf-8")
    assert '"isFrame": true' in raw
    assert '"parentId": "frame"' in raw


def make_subgraph_workflow() -> SavedWorkflowDocument:
    return SavedWorkflowDocument.model_validate(
        {
            "version": 1,
            "savedAt": "2026-10-20T12:00:00Z",
            "nodes": [
                {
                    "id": "instance",
                    "label": "Preprocess",
                    "position": {"x": 200, "y": 120},
                    "code": "",
                    "widgetValues": {},
                    "templateName": None,
                    "isSubgraph": True,
                    "subgraphId": "def-1",
                }
            ],
            "edges": [],
            "subgraphs": {
                "def-1": {
                    "id": "def-1",
                    "name": "Preprocess",
                    "inputs": [{"name": "df", "data_type": "DATAFRAME", "optional": True}],
                    "outputs": [{"name": "result", "data_type": "ANY"}],
                    "nodes": [
                        {
                            "id": "inside",
                            "label": "Inside",
                            "position": {"x": 0, "y": 0},
                            "code": "value = 1\nreturn value: ANY\n",
                            "widgetValues": {},
                            "templateName": None,
                        }
                    ],
                    "edges": [
                        {
                            "id": "in",
                            "source": "__subgraph_input__",
                            "sourceHandle": "df",
                            "target": "inside",
                            "targetHandle": "df",
                        },
                        {
                            "id": "out",
                            "source": "inside",
                            "sourceHandle": "result",
                            "target": "__subgraph_output__",
                            "targetHandle": "result",
                        },
                    ],
                    "inputPosition": {"x": -260, "y": 0},
                    "outputPosition": {"x": 300, "y": 0},
                }
            },
        }
    )


def test_subgraph_documents_round_trip(store: ProjectStore) -> None:
    _, created = store.create_project("Nested", make_subgraph_workflow())
    assert created is True

    detail = store.get_project("nested")
    assert detail.workflow.nodes[0].is_subgraph is True
    assert detail.workflow.nodes[0].subgraph_id == "def-1"
    definition = detail.workflow.subgraphs["def-1"]
    assert definition.inputs[0].name == "df"
    assert definition.outputs[0].name == "result"
    assert definition.nodes[0].id == "inside"

    raw = (store.projects_dir / "nested.cuigraph.json").read_text(encoding="utf-8")
    assert '"isSubgraph": true' in raw
    assert '"subgraphId": "def-1"' in raw
    assert '"subgraphs"' in raw


def test_subgraph_relationships_are_validated() -> None:
    missing = make_subgraph_workflow().model_dump(mode="json", by_alias=True)
    missing["subgraphs"] = {}
    with pytest.raises(ValueError, match="missing subgraph"):
        SavedWorkflowDocument.model_validate(missing)

    cyclic = make_subgraph_workflow().model_dump(mode="json", by_alias=True)
    definition = cyclic["subgraphs"]["def-1"]
    definition["nodes"].append(
        {
            "id": "nested",
            "label": "Nested",
            "position": {"x": 0, "y": 200},
            "code": "",
            "widgetValues": {},
            "templateName": None,
            "isSubgraph": True,
            "subgraphId": "def-1",
        }
    )
    with pytest.raises(ValueError, match="subgraph cycle"):
        SavedWorkflowDocument.model_validate(cyclic)


def test_frame_relationships_are_validated() -> None:
    missing_parent = make_frame_workflow().model_dump(mode="json", by_alias=True)
    missing_parent["nodes"][1]["parentId"] = "missing"
    with pytest.raises(ValueError, match="missing parent"):
        SavedWorkflowDocument.model_validate(missing_parent)

    frame_edge = make_frame_workflow().model_dump(mode="json", by_alias=True)
    frame_edge["edges"] = [
        {
            "id": "frame-edge",
            "source": "frame",
            "sourceHandle": "out",
            "target": "child",
            "targetHandle": "in",
        }
    ]
    with pytest.raises(ValueError, match="frame nodes"):
        SavedWorkflowDocument.model_validate(frame_edge)


def test_version_retention_keeps_newest_snapshots(
    store: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_module, "PROJECT_VERSION_RETENTION", 2)
    store.create_project("Demo", make_workflow("a"))

    for index in range(4):
        store.update_project("demo", "Demo", make_workflow(f"n{index}"))

    versions = store.list_versions("demo")
    assert len(versions) == 2

    store.delete_project("demo")
    assert not (store.projects_dir / "demo.cuigraph.json").exists()
    assert list(store.trash_dir.glob("demo-*/demo.cuigraph.json"))
    assert list(store.trash_dir.glob("demo-*/versions/*.cuigraph.json"))


def test_session_round_trip_and_corruption(store: ProjectStore) -> None:
    assert store.load_session() is None

    session = store.save_session(
        WorkspaceSessionRequest(
            workflow=make_workflow("a", name="Draft"),
            project_slug="draft",
            project_name="Draft",
        )
    )
    loaded = store.load_session()
    assert loaded is not None
    assert loaded.saved_at == session.saved_at
    assert loaded.project_slug == "draft"
    assert loaded.workflow.name == "Draft"

    store.clear_session()
    assert store.load_session() is None

    store.session_path.write_text("{broken", encoding="utf-8")
    assert store.load_session() is None


def test_invalid_slugs_and_documents_are_rejected(store: ProjectStore) -> None:
    with pytest.raises(InvalidProjectSlug):
        store.get_project("../outside")

    store.projects_dir.mkdir(parents=True)
    (store.projects_dir / "broken.cuigraph.json").write_text(
        '{"version": 1, "nodes": [{"id": "x"}]}', encoding="utf-8"
    )
    (store.projects_dir / "Bad Slug.cuigraph.json").write_text(
        '{"version": 1, "nodes": [], "edges": []}', encoding="utf-8"
    )
    with pytest.raises(InvalidProjectDocument):
        store.get_project("broken")

    assert store.list_projects() == []

    with pytest.raises(ProjectNotFound):
        store.update_project("missing", "Missing", make_workflow("a"))


def test_slugify_project_name() -> None:
    assert slugify_project_name("Iris Demo / v2") == "iris-demo-v2"
    assert slugify_project_name("  ") == "project"
    assert slugify_project_name("Café ☕") == "cafe"


@pytest.fixture()
def project_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COMPLEXUI_WORKSPACE", str(tmp_path / "api-workspace"))
    get_project_store.cache_clear()
    with TestClient(app) as client:
        yield client
    get_project_store.cache_clear()


def test_project_api_lifecycle(project_client: TestClient) -> None:
    workflow = make_workflow("load", "train").model_dump(mode="json", by_alias=True)

    response = project_client.post("/api/projects", json={"name": "API Demo", "workflow": workflow})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "created"
    assert body["project"]["slug"] == "api-demo"

    conflict = project_client.post("/api/projects", json={"name": "API Demo", "workflow": workflow})
    assert conflict.status_code == 409

    listed = project_client.get("/api/projects")
    assert listed.status_code == 200
    assert listed.json()[0]["node_count"] == 2

    detail = project_client.get("/api/projects/api-demo")
    assert detail.status_code == 200
    assert detail.json()["workflow"]["nodes"][0]["widgetValues"] == {"k": 0}

    updated = project_client.put(
        "/api/projects/api-demo",
        json={
            "name": "API Demo",
            "workflow": make_workflow("load").model_dump(mode="json", by_alias=True),
        },
    )
    assert updated.status_code == 200
    assert updated.json()["project"]["version_count"] == 1

    versions = project_client.get("/api/projects/api-demo/versions")
    assert versions.status_code == 200
    version = versions.json()[0]["version"]
    snapshot = project_client.get(f"/api/projects/api-demo/versions/{version}")
    assert snapshot.status_code == 200
    assert len(snapshot.json()["workflow"]["nodes"]) == 2

    deleted = project_client.delete("/api/projects/api-demo")
    assert deleted.status_code == 200
    assert project_client.get("/api/projects/api-demo").status_code == 404


def test_project_api_preserves_subgraphs(project_client: TestClient) -> None:
    workflow = make_subgraph_workflow().model_dump(mode="json", by_alias=True)
    response = project_client.post(
        "/api/projects", json={"name": "Nested API", "workflow": workflow}
    )
    assert response.status_code == 200

    detail = project_client.get("/api/projects/nested-api").json()
    assert detail["workflow"]["nodes"][0]["isSubgraph"] is True
    definition = detail["workflow"]["subgraphs"]["def-1"]
    assert definition["inputs"] == [
        {"name": "df", "data_type": "DATAFRAME", "optional": True}
    ]
    assert definition["nodes"][0]["id"] == "inside"


def test_project_api_preserves_frames(project_client: TestClient) -> None:
    workflow = make_frame_workflow().model_dump(mode="json", by_alias=True)
    response = project_client.post(
        "/api/projects", json={"name": "Framed API", "workflow": workflow}
    )
    assert response.status_code == 200

    detail = project_client.get("/api/projects/framed-api").json()
    frame, child = detail["workflow"]["nodes"]
    assert frame["isFrame"] is True
    assert frame["width"] == 640
    assert frame["height"] == 420
    assert child["parentId"] == "frame"


def test_project_api_validates_documents_and_slugs(project_client: TestClient) -> None:
    workflow = make_workflow("a").model_dump(mode="json", by_alias=True)
    workflow["edges"] = [
        {
            "id": "e1",
            "source": "a",
            "sourceHandle": "out",
            "target": "missing",
            "targetHandle": "in",
        }
    ]

    invalid = project_client.post("/api/projects", json={"name": "Bad", "workflow": workflow})
    assert invalid.status_code == 422

    assert project_client.get("/api/projects/..%2Foutside").status_code in {404, 422}
    assert project_client.get("/api/projects/no-such/versions").status_code == 404


def test_session_api_round_trip(project_client: TestClient) -> None:
    assert project_client.get("/api/session").json() is None

    workflow = make_workflow("a", name="Recovered").model_dump(mode="json", by_alias=True)
    saved = project_client.put(
        "/api/session",
        json={
            "workflow": workflow,
            "project_slug": "recovered",
            "project_name": "Recovered",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["status"] == "saved"

    session = project_client.get("/api/session").json()
    assert session["project_slug"] == "recovered"
    assert session["workflow"]["nodes"][0]["id"] == "a"
