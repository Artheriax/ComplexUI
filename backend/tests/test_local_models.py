"""Local model loading through the built-in vision templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.executor import clear_cache, execute_graph
from app.models import get_model_store
from app.schemas import GraphEdge, GraphNode, WorkflowGraph
from app.security import SecurityRegistry


def _graph(*nodes: GraphNode, edges: list[GraphEdge] | None = None) -> WorkflowGraph:
    return WorkflowGraph(nodes=list(nodes), edges=edges or [])


def _torchvision_graph(local_path: str, node_prefix: str) -> WorkflowGraph:
    registry = SecurityRegistry()
    demo = GraphNode(
        id=f"{node_prefix}-demo",
        code=registry.get_template("make_demo_image").code,
    )
    model = GraphNode(
        id=f"{node_prefix}-model",
        code=registry.get_template("torchvision_classifier").code,
        widget_values={
            "local_model": local_path,
            "architecture": "mobilenet_v3_small",
            "weights": "none",
            "device": "cpu",
        },
    )
    classify = GraphNode(
        id=f"{node_prefix}-classify",
        code=registry.get_template("classify_image").code,
        widget_values={"top_k": 3},
    )
    return _graph(
        demo,
        model,
        classify,
        edges=[
            GraphEdge(
                source_node_id=demo.id,
                source_handle="image",
                target_node_id=classify.id,
                target_handle="image",
            ),
            GraphEdge(
                source_node_id=model.id,
                source_handle="model",
                target_node_id=classify.id,
                target_handle="model",
            ),
        ],
    )


def _assert_local_classification(result, model_node: str, path: str, fmt: str) -> None:
    preview = result.previews[model_node]["model"]
    assert preview["source"] == "local"
    assert preview["format"] == fmt
    assert preview["path"] == path
    assert result.previews[f"{model_node.rsplit('-', 1)[0]}-classify"][
        "classifications"
    ]["kind"] == "classifications"


@pytest.mark.parametrize("suffix", [".pth", ".safetensors"])
def test_torchvision_classifier_loads_local_state_dict(suffix: str) -> None:
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    store = get_model_store()
    path = store.root / "classification" / f"local{suffix}"
    model = torchvision.models.get_model("mobilenet_v3_small", weights=None)
    if suffix == ".safetensors":
        safetensors = pytest.importorskip("safetensors.torch")
        safetensors.save_file(model.state_dict(), str(path))
    else:
        torch.save(model.state_dict(), path)

    clear_cache()
    try:
        result = execute_graph(
            _torchvision_graph(f"models/classification/{path.name}", "state"),
            allow_installs=False,
        )
    finally:
        clear_cache()
    _assert_local_classification(
        result,
        "state-model",
        str(path.resolve()),
        suffix.lstrip("."),
    )


def test_torchvision_classifier_loads_local_onnx() -> None:
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    store = get_model_store()
    path = store.root / "classification" / "local.onnx"
    model = torchvision.models.get_model("mobilenet_v3_small", weights=None).eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )

    clear_cache()
    try:
        result = execute_graph(
            _torchvision_graph("models/classification/local.onnx", "onnx"),
            allow_installs=False,
        )
    finally:
        clear_cache()
    _assert_local_classification(result, "onnx-model", str(path.resolve()), "onnx")


def test_yolo_detector_loads_local_checkpoint() -> None:
    pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")
    store = get_model_store()
    path: Path = store.root / "yolo" / "local.pt"
    ultralytics.YOLO("yolo11n.yaml").save(str(path))
    registry = SecurityRegistry()
    demo = GraphNode(id="demo", code=registry.get_template("make_demo_image").code)
    model = GraphNode(
        id="model",
        code=registry.get_template("yolo_detector").code,
        widget_values={
            "local_model": "models/yolo/local.pt",
            "architecture": "yolo11n",
            "weights": "none",
            "device": "cpu",
        },
    )
    detect = GraphNode(
        id="detect",
        code=registry.get_template("yolo_detect").code,
        widget_values={"score_threshold": 0.25, "max_detections": 5},
    )
    graph = _graph(
        demo,
        model,
        detect,
        edges=[
            GraphEdge(
                source_node_id="demo",
                source_handle="image",
                target_node_id="detect",
                target_handle="image",
            ),
            GraphEdge(
                source_node_id="model",
                source_handle="model",
                target_node_id="detect",
                target_handle="model",
            ),
        ],
    )

    clear_cache()
    try:
        result = execute_graph(graph, allow_installs=False)
    finally:
        clear_cache()
    assert result.previews["model"]["model"]["source"] == "local"
    assert result.previews["model"]["model"]["format"] == "pt"
    assert result.previews["detect"]["detections"]["kind"] == "detections"
