"""Tests for the example workflow catalog: validation, materialisation and E2E."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.executor import clear_cache, execute_graph
from app.security import SecurityRegistry
from app.workflow_catalog import (
    WORKFLOW_CATEGORIES,
    WorkflowCatalogError,
    get_workflows,
    load_workflows,
    materialize_workflow,
    validate_workflow,
)

EXPECTED_WORKFLOWS = {
    "breast_cancer_random_forest",
    "diabetes_ridge_regression",
    "digits_xgboost_classifier",
    "eda_iris_report",
    "iris_knn_classification",
    "messy_data_cleaning_pipeline",
    "pca_kmeans_segmentation",
    "smote_imbalanced_pipeline",
    "torch_mlp_synthetic",
    "torch_wine_mlp",
    "wine_lightgbm_classifier",
    "wine_logistic_regression",
}

OPTIONAL_MODULES = {
    "digits_xgboost_classifier": "xgboost",
    "wine_lightgbm_classifier": "lightgbm",
    "torch_mlp_synthetic": "torch",
    "torch_wine_mlp": "torch",
}

ACCURACY_FLOORS = {
    "iris_knn_classification": 0.90,
    "wine_logistic_regression": 0.90,
    "breast_cancer_random_forest": 0.90,
    "digits_xgboost_classifier": 0.90,
    "wine_lightgbm_classifier": 0.90,
    "smote_imbalanced_pipeline": 0.85,
    "torch_mlp_synthetic": 0.70,
    "torch_wine_mlp": 0.70,
}


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_cache()
    yield
    clear_cache()


def test_bundled_workflows_load_with_expected_names() -> None:
    workflows = get_workflows()

    assert set(workflows) == EXPECTED_WORKFLOWS
    for workflow in workflows.values():
        assert len(workflow.nodes) >= 3
        assert workflow.edges
        assert workflow.category in WORKFLOW_CATEGORIES
        assert workflow.title.strip()
        assert workflow.description.strip()
        validate_workflow(workflow)


def _write_catalog(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_payload(name: str, nodes: list[dict], edges: list[dict]) -> dict:
    return {
        "name": name,
        "title": "Broken Workflow",
        "category": "Getting Started",
        "description": "Invalid on purpose.",
        "nodes": nodes,
        "edges": edges,
    }


def test_catalog_rejects_unknown_template(tmp_path: Path) -> None:
    payload = _base_payload(
        "broken",
        [{"id": "a", "template": "does_not_exist", "position": {"x": 0, "y": 0}}],
        [],
    )
    _write_catalog(tmp_path, "broken", payload)

    with pytest.raises(WorkflowCatalogError, match="unknown template"):
        load_workflows(tmp_path)


def test_catalog_rejects_socket_type_mismatch(tmp_path: Path) -> None:
    payload = _base_payload(
        "broken",
        [
            {"id": "load", "template": "load_iris", "position": {"x": 0, "y": 0}},
            {"id": "knn", "template": "knn_train", "position": {"x": 200, "y": 0}},
        ],
        [
            {
                "source_node_id": "load",
                "source_handle": "df",
                "target_node_id": "knn",
                "target_handle": "train",
            }
        ],
    )
    _write_catalog(tmp_path, "broken", payload)

    with pytest.raises(WorkflowCatalogError, match="type mismatch"):
        load_workflows(tmp_path)


def test_catalog_accepts_widget_parameter_edges(tmp_path: Path) -> None:
    payload = _base_payload(
        "widget_wire",
        [
            {"id": "load", "template": "load_iris", "position": {"x": 0, "y": 0}},
            {
                "id": "size",
                "template": "float_value",
                "position": {"x": 0, "y": 200},
                "widget_values": {"value": 0.5},
            },
            {
                "id": "split",
                "template": "train_test_split",
                "position": {"x": 300, "y": 100},
            },
        ],
        [
            {
                "source_node_id": "load",
                "source_handle": "df",
                "target_node_id": "split",
                "target_handle": "df",
            },
            {
                "source_node_id": "size",
                "source_handle": "result",
                "target_node_id": "split",
                "target_handle": "test_size",
            },
        ],
    )
    _write_catalog(tmp_path, "widget_wire", payload)

    workflow = load_workflows(tmp_path)["widget_wire"]
    result = execute_graph(materialize_workflow(workflow))

    assert result.previews["split"]["train"]["X_shape"] == [75, 4]
    assert result.previews["split"]["test"]["X_shape"] == [75, 4]


def test_catalog_rejects_values_outside_literal_choices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "choice_node.py").write_text(
        "from typing import Literal\n"
        "level: Literal['low', 'medium', 'high'] = 'low'\n"
        "return level: STRING\n",
        encoding="utf-8",
    )
    registry = SecurityRegistry(templates_dir)
    monkeypatch.setattr(
        "app.workflow_catalog.get_security_registry",
        lambda: registry,
    )
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    payload = _base_payload(
        "bad_choice",
        [
            {
                "id": "choice",
                "template": "choice_node",
                "position": {"x": 0, "y": 0},
                "widget_values": {"level": "extreme"},
            }
        ],
        [],
    )
    _write_catalog(workflows_dir, "bad_choice", payload)

    with pytest.raises(WorkflowCatalogError, match="must be one of"):
        load_workflows(workflows_dir)


def test_catalog_rejects_cycles(tmp_path: Path) -> None:
    payload = _base_payload(
        "broken",
        [
            {
                "id": "a",
                "template": "drop_columns",
                "position": {"x": 0, "y": 0},
                "widget_values": {"columns": "x"},
            },
            {
                "id": "b",
                "template": "drop_columns",
                "position": {"x": 200, "y": 0},
                "widget_values": {"columns": "x"},
            },
        ],
        [
            {
                "source_node_id": "a",
                "source_handle": "dropped",
                "target_node_id": "b",
                "target_handle": "df",
            },
            {
                "source_node_id": "b",
                "source_handle": "dropped",
                "target_node_id": "a",
                "target_handle": "df",
            },
        ],
    )
    _write_catalog(tmp_path, "broken", payload)

    with pytest.raises(WorkflowCatalogError, match="cycle"):
        load_workflows(tmp_path)


def test_catalog_rejects_unknown_widgets_and_duplicate_inputs(
    tmp_path: Path,
) -> None:
    widget_dir = tmp_path / "widgets"
    widget_dir.mkdir()
    unknown_widget = _base_payload(
        "broken",
        [
            {
                "id": "split",
                "template": "train_test_split",
                "position": {"x": 0, "y": 0},
                "widget_values": {"nope": 3},
            }
        ],
        [],
    )
    _write_catalog(widget_dir, "broken", unknown_widget)
    with pytest.raises(WorkflowCatalogError, match="unknown widget"):
        load_workflows(widget_dir)

    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    duplicate_input = _base_payload(
        "double",
        [
            {"id": "a", "template": "load_iris", "position": {"x": 0, "y": 0}},
            {"id": "b", "template": "load_iris", "position": {"x": 0, "y": 200}},
            {"id": "split", "template": "train_test_split", "position": {"x": 300, "y": 100}},
        ],
        [
            {
                "source_node_id": "a",
                "source_handle": "df",
                "target_node_id": "split",
                "target_handle": "df",
            },
            {
                "source_node_id": "b",
                "source_handle": "df",
                "target_node_id": "split",
                "target_handle": "df",
            },
        ],
    )
    _write_catalog(duplicate_dir, "double", duplicate_input)
    with pytest.raises(WorkflowCatalogError, match="wired twice"):
        load_workflows(duplicate_dir)


def test_catalog_rejects_name_mismatch_and_wrong_widget_types(
    tmp_path: Path,
) -> None:
    mismatch_dir = tmp_path / "mismatch"
    mismatch_dir.mkdir()
    mismatched = _base_payload(
        "other_name",
        [{"id": "load", "template": "load_iris", "position": {"x": 0, "y": 0}}],
        [],
    )
    _write_catalog(mismatch_dir, "mismatched", mismatched)
    with pytest.raises(WorkflowCatalogError, match="does not match file stem"):
        load_workflows(mismatch_dir)

    type_dir = tmp_path / "types"
    type_dir.mkdir()
    bad_type = _base_payload(
        "bad_type",
        [
            {
                "id": "split",
                "template": "train_test_split",
                "position": {"x": 0, "y": 0},
                "widget_values": {"test_size": "many"},
            }
        ],
        [],
    )
    _write_catalog(type_dir, "bad_type", bad_type)
    with pytest.raises(WorkflowCatalogError, match="expects float"):
        load_workflows(type_dir)


def test_materialized_workflow_executes_with_high_accuracy() -> None:
    workflow = get_workflows()["iris_knn_classification"]
    graph = materialize_workflow(workflow)

    result = execute_graph(graph)

    assert set(result.cached) == {node.id for node in workflow.nodes}
    assert all(cached is False for cached in result.cached.values())
    metrics = result.outputs_by_name["evaluate"]["metrics"]
    assert metrics["accuracy"] > 0.90


@pytest.mark.parametrize("name", sorted(EXPECTED_WORKFLOWS))
def test_every_workflow_runs_end_to_end(name: str) -> None:
    module = OPTIONAL_MODULES.get(name)
    if module is not None:
        pytest.importorskip(module)

    workflow = get_workflows()[name]
    result = execute_graph(materialize_workflow(workflow))

    assert set(result.cached) == {node.id for node in workflow.nodes}
    for node in workflow.nodes:
        if node.template in {"evaluate_metrics", "evaluate_torch"}:
            metrics = result.outputs_by_name[node.id]["metrics"]
            assert metrics["accuracy"] > ACCURACY_FLOORS.get(name, 0.0), name
        if node.template == "evaluate_regression":
            metrics = result.outputs_by_name[node.id]["metrics"]
            assert metrics["r2"] > 0.10
