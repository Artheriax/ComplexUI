"""Tests for the example workflow catalog: validation, materialisation and E2E."""

from __future__ import annotations

import json
import os
from datetime import datetime
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
    "imagegen_anima",
    "imagegen_fibo2",
    "imagegen_flux1_kontext",
    "imagegen_flux2",
    "imagegen_flux3",
    "imagegen_ideogram4",
    "imagegen_krea2",
    "imagegen_moss_image",
    "imagegen_qwen_image",
    "imagegen_sd15_advanced",
    "imagegen_sd15_dreamshaper",
    "imagegen_sd15_img2img",
    "imagegen_sd15_inpaint",
    "imagegen_sd15_ksampler_advanced",
    "imagegen_sd15_sampler_custom",
    "imagegen_sdxl_lightning",
    "imagegen_z_image",
    "iris_knn_classification",
    "iris_visual_eda",
    "messy_data_cleaning_pipeline",
    "pca_kmeans_segmentation",
    "smote_imbalanced_pipeline",
    "torch_mlp_synthetic",
    "torch_wine_mlp",
    "vision_image_classification",
    "vision_image_detection",
    "vision_image_segmentation",
    "vision_media_smoke",
    "vision_video_classification",
    "vision_video_detection",
    "vision_video_segmentation",
    "vision_yolo_detection",
    "vision_yolo_segmentation",
    "wine_lightgbm_classifier",
    "wine_logistic_regression",
}

OPTIONAL_MODULES = {
    "digits_xgboost_classifier": "xgboost",
    "wine_lightgbm_classifier": "lightgbm",
    "torch_mlp_synthetic": "torch",
    "torch_wine_mlp": "torch",
    "vision_image_classification": "torchvision",
    "vision_image_detection": "torchvision",
    "vision_image_segmentation": "torchvision",
    "vision_video_classification": "torchvision",
    "vision_video_detection": "ultralytics",
    "vision_video_segmentation": "ultralytics",
    "vision_yolo_detection": "ultralytics",
    "vision_yolo_segmentation": "ultralytics",
}

GENERATION_WORKFLOWS = {
    "imagegen_anima",
    "imagegen_fibo2",
    "imagegen_flux1_kontext",
    "imagegen_flux2",
    "imagegen_flux3",
    "imagegen_ideogram4",
    "imagegen_krea2",
    "imagegen_moss_image",
    "imagegen_qwen_image",
    "imagegen_sd15_advanced",
    "imagegen_sd15_dreamshaper",
    "imagegen_sd15_img2img",
    "imagegen_sd15_inpaint",
    "imagegen_sd15_ksampler_advanced",
    "imagegen_sd15_sampler_custom",
    "imagegen_sdxl_lightning",
    "imagegen_z_image",
}

def _today_output(model: str, counter: int = 1) -> str:
    day = datetime.now().strftime("%Y-%m-%d")
    return f"outputs/{day}/{model}_{counter:03d}.png"


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


def test_complex_family_workflows_cover_adapters() -> None:
    workflows = get_workflows()
    for name in (
        "imagegen_anima",
        "imagegen_fibo2",
        "imagegen_ideogram4",
        "imagegen_moss_image",
    ):
        templates = {node.template for node in workflows[name].nodes}
        assert "load_diffusion_model" in templates
        assert "conditioning_zero_out" in templates
        assert "ksampler" in templates
    fibo_templates = {
        node.template for node in workflows["imagegen_fibo2"].nodes
    }
    assert "encode_reference_images" in fibo_templates
    for name in ("imagegen_anima", "imagegen_moss_image"):
        templates = {node.template for node in workflows[name].nodes}
        assert "inpaint_conditioner" in templates
    moss_templates = {
        node.template for node in workflows["imagegen_moss_image"].nodes
    }
    assert "moss_guidance" in moss_templates


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


def test_torchvision_workflows_emit_typed_vision_outputs() -> None:
    pytest.importorskip("torchvision")
    workflows = get_workflows()

    classification = execute_graph(
        materialize_workflow(workflows["vision_image_classification"])
    )
    predictions = classification.outputs_by_name["classify"]["classifications"]
    assert classification.previews["classify"]["classifications"]["kind"] == "classifications"
    assert len(predictions["top_k"]) == 5
    assert all(0.0 <= item["score"] <= 1.0 for item in predictions["top_k"])
    assert classification.previews["model"]["model"]["kind"] == "vision_model"

    detection = execute_graph(
        materialize_workflow(workflows["vision_image_detection"])
    )
    detections = detection.outputs_by_name["detect"]["detections"]
    assert detection.previews["detect"]["detections"]["kind"] == "detections"
    assert detections["count"] == len(detections["detections"])
    assert detection.previews["detect"]["annotated"]["kind"] == "image"
    assert detection.previews["save"]["saved"]["kind"] == "image"
    assert detection.outputs_by_name["save"]["saved_path"] == _today_output(
        "fasterrcnn_mobilenet_v3_large_320_fpn"
    )

    segmentation = execute_graph(
        materialize_workflow(workflows["vision_image_segmentation"])
    )
    masks = segmentation.outputs_by_name["segment"]["masks"]
    assert segmentation.previews["segment"]["masks"]["kind"] == "masks"
    assert len(masks["masks"]) <= 10
    assert all(item["mask"] for item in masks["masks"])
    assert segmentation.previews["segment"]["annotated"]["kind"] == "image"
    assert segmentation.outputs_by_name["save"]["saved_path"] == _today_output(
        "deeplabv3_mobilenet_v3_large"
    )


def test_ultralytics_workflows_emit_typed_vision_outputs() -> None:
    pytest.importorskip("ultralytics")
    workflows = get_workflows()

    detection = execute_graph(
        materialize_workflow(workflows["vision_yolo_detection"])
    )
    detections = detection.outputs_by_name["detect"]["detections"]
    assert detection.previews["model"]["model"]["kind"] == "vision_model"
    assert detection.previews["detect"]["detections"]["kind"] == "detections"
    assert detections["backend"] == "ultralytics"
    assert detections["count"] == len(detections["detections"])
    assert detection.previews["save"]["saved"]["kind"] == "image"
    assert detection.outputs_by_name["save"]["saved_path"] == _today_output(
        "yolo11n"
    )

    segmentation = execute_graph(
        materialize_workflow(workflows["vision_yolo_segmentation"])
    )
    masks = segmentation.outputs_by_name["segment"]["masks"]
    assert segmentation.previews["segment"]["masks"]["kind"] == "masks"
    assert masks["backend"] == "ultralytics"
    assert len(masks["masks"]) <= 10
    assert segmentation.previews["segment"]["annotated"]["kind"] == "image"
    assert segmentation.outputs_by_name["save"]["saved_path"] == _today_output(
        "yolo11n-seg"
    )


def test_video_workflows_emit_annotated_media() -> None:
    pytest.importorskip("torchvision")
    pytest.importorskip("ultralytics")
    workflows = get_workflows()

    classification = execute_graph(
        materialize_workflow(workflows["vision_video_classification"])
    )
    predictions = classification.outputs_by_name["classify"]["classifications"]
    assert predictions["frames"] == 6
    assert len(predictions["top_k"]) == 5
    assert classification.previews["demo"]["video"]["kind"] == "video"
    assert classification.previews["classify"]["classifications"]["kind"] == (
        "classifications"
    )

    detection = execute_graph(
        materialize_workflow(workflows["vision_video_detection"])
    )
    detections = detection.outputs_by_name["detect"]["detections"]
    detection_video = detection.outputs_by_name["detect"]["annotated"]
    assert detections["frames"] == 24
    assert detections["backend"] == "ultralytics"
    assert detection.previews["detect"]["annotated"]["kind"] == "video"
    assert detection.previews["detect"]["annotated"]["url"] == (
        "/api/outputs/vision-video-detection.mp4/raw"
    )
    assert detection_video["relative_path"] == "outputs/vision-video-detection.mp4"
    assert Path(detection_video["path"]).is_file()

    segmentation = execute_graph(
        materialize_workflow(workflows["vision_video_segmentation"])
    )
    masks = segmentation.outputs_by_name["segment"]["masks"]
    segmentation_video = segmentation.outputs_by_name["segment"]["annotated"]
    assert masks["frames"] == 24
    assert masks["backend"] == "ultralytics"
    assert segmentation.previews["segment"]["masks"]["kind"] == "masks"
    assert segmentation_video["relative_path"] == "outputs/vision-video-segmentation.mp4"
    assert Path(segmentation_video["path"]).is_file()


@pytest.mark.parametrize("name", sorted(EXPECTED_WORKFLOWS))
def test_every_workflow_runs_end_to_end(name: str) -> None:
    if name in GENERATION_WORKFLOWS and os.getenv(
        "COMPLEXUI_TEST_GENERATION"
    ) != "1":
        pytest.skip("set COMPLEXUI_TEST_GENERATION=1 to run diffusion workflows")
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
