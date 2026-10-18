"""PyTorch path: TENSOR sockets, tensor-aware previews, optional torch chain."""

from __future__ import annotations

from typing import Any

import pytest

from app.executor import build_output_preview, clear_cache, execute_graph
from app.schemas import GraphEdge, GraphNode, SocketType, WorkflowGraph
from app.security import SecurityRegistry
from app.transpiler import parse_node


def _templates() -> dict[str, str]:
    return {record.name: record.code for record in SecurityRegistry().templates()}


def test_torch_template_socket_types() -> None:
    templates = _templates()

    dataset = parse_node(templates["tensor_dataset"])
    assert dataset.syntax_errors == []
    assert dataset.outputs[0].data_type is SocketType.DATASET
    assert [widget.data_type for widget in dataset.widgets] == [
        "int",
        "int",
        "float",
        "str",
    ]

    train = parse_node(templates["mlp_train"])
    assert train.syntax_errors == []
    assert [socket.data_type for socket in train.inputs] == [SocketType.DATASET]
    assert [socket.data_type for socket in train.outputs] == [
        SocketType.ML_MODEL,
        SocketType.TENSOR,
    ]

    evaluate = parse_node(templates["evaluate_torch"])
    assert evaluate.syntax_errors == []
    assert [socket.data_type for socket in evaluate.inputs] == [
        SocketType.ML_MODEL,
        SocketType.DATASET,
    ]
    assert evaluate.outputs[0].data_type is SocketType.METRICS


def test_torch_templates_declare_torch_dependency() -> None:
    templates = _templates()
    for name in ("tensor_dataset", "mlp_train", "evaluate_torch"):
        schema = parse_node(templates[name])
        assert any(dep.startswith("torch") for dep in schema.dependencies), name


class FakeTensor:
    """Minimal torch.Tensor stand-in for preview tests."""

    def __init__(
        self,
        values: list[float],
        shape: tuple[int, ...],
        dtype: str = "float32",
        device: str = "cpu",
    ) -> None:
        self._values = values
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def detach(self) -> "FakeTensor":
        return self

    def reshape(self, *_shape: int) -> "FakeTensor":
        return self

    def __getitem__(self, item: Any) -> Any:
        return self._values[item]

    def tolist(self) -> list[float]:
        return list(self._values)


def test_tensor_preview_from_fake_tensor() -> None:
    tensor = FakeTensor(
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], (2, 3), dtype="float32", device="cpu"
    )

    preview = build_output_preview(SocketType.TENSOR, tensor)

    assert preview == {
        "shape": [2, 3],
        "dtype": "float32",
        "device": "cpu",
        "sample": [0.5, 1.5, 2.5, 3.5, 4.5],
    }


def test_tensor_preview_handles_foreign_objects() -> None:
    preview = build_output_preview(SocketType.TENSOR, object())

    assert isinstance(preview, str)
    assert preview.startswith("<object object at")


class FakeParameter:
    def __init__(self, numel: int) -> None:
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class FakeTorchModule:
    """Minimal nn.Module stand-in: no ``get_params``, but ``state_dict``."""

    def parameters(self) -> list[FakeParameter]:
        return [FakeParameter(64), FakeParameter(16)]

    def state_dict(self) -> dict[str, int]:
        return {"0.weight": 1, "0.bias": 2, "2.weight": 3, "2.bias": 4}


def test_ml_model_preview_for_torch_like_module() -> None:
    preview = build_output_preview(SocketType.ML_MODEL, FakeTorchModule())

    assert preview == {
        "class": "FakeTorchModule",
        "parameters": 80,
        "layers": 4,
    }


def test_ml_model_preview_falls_back_to_repr() -> None:
    preview = build_output_preview(SocketType.ML_MODEL, object())

    assert isinstance(preview, str)
    assert preview.startswith("<object object at")


def test_torch_chain_end_to_end_when_torch_available() -> None:
    pytest.importorskip("torch", reason="torch is not installed in this environment")

    templates = _templates()
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="data", code=templates["tensor_dataset"]),
            GraphNode(id="train", code=templates["mlp_train"]),
            GraphNode(id="evaluate", code=templates["evaluate_torch"]),
        ],
        edges=[
            GraphEdge(
                source_node_id="data",
                source_handle="data",
                target_node_id="train",
                target_handle="data",
            ),
            GraphEdge(
                source_node_id="train",
                source_handle="model",
                target_node_id="evaluate",
                target_handle="model",
            ),
            GraphEdge(
                source_node_id="data",
                source_handle="data",
                target_node_id="evaluate",
                target_handle="data",
            ),
        ],
    )

    clear_cache()
    result = execute_graph(graph, security=SecurityRegistry())

    metrics = result.outputs_by_name["evaluate"]["metrics"]
    assert metrics["accuracy"] > 0.85
    assert metrics["n_test_samples"] == 600

    history = result.previews["train"]["history"]
    assert history["shape"] == [80]
    assert history["sample"][0] > history["sample"][4]

    model_preview = result.previews["train"]["model"]
    assert model_preview["class"] == "Sequential"
    assert model_preview["parameters"] > 0
