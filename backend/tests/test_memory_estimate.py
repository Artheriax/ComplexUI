"""Memory-estimate contract tests for models and workflows."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from app.main import app
from app.memory_estimate import estimate_model_memory
from app.models import MODELS_ENV, ModelStore, get_model_store
from app.schemas import GraphNode, QuantizationInfo, WorkflowGraph
from app.workflow_estimate import estimate_workflow

_NODE_CODE = """\
checkpoint_name: ModelSelector = ModelSelector(type="diffusion")
return checkpoint_name
"""

_REQUIRED_NODE_CODE = """\
checkpoint_name: ModelSelector = ModelSelector(type="diffusion", required=True)
return checkpoint_name
"""


def _graph(*nodes: GraphNode) -> WorkflowGraph:
    return WorkflowGraph(nodes=list(nodes), edges=[])


def _node(
    node_id: str, value: str | None, *, code: str = _NODE_CODE
) -> GraphNode:
    return GraphNode(
        id=node_id,
        code=code,
        widget_values={"checkpoint_name": value} if value is not None else {},
    )


def _store(tmp_path: Path) -> ModelStore:
    store = ModelStore(tmp_path / "library")
    save_file(
        {"weight": torch.zeros((8, 16), dtype=torch.float16)},
        store.root / "diffusion" / "tiny.safetensors",
    )
    return store


def test_unquantized_memory_uses_checkpoint_size_and_overhead() -> None:
    estimate = estimate_model_memory(size_bytes=2 * 1024**3, parameter_count=10**9)

    assert estimate.weights_bytes == 2 * 1024**3
    assert estimate.quantization_overhead_bytes == 0
    assert estimate.resident_estimate_bytes >= estimate.weights_bytes
    assert estimate.offload_ram_estimate_bytes >= estimate.weights_bytes
    assert estimate.confidence == "medium"
    assert estimate.unknown is False


def test_quantized_memory_keeps_declared_precision() -> None:
    info = QuantizationInfo(
        format="int8_w8a8",
        label="INT8 W8A8",
        runtime="comfy_kitchen",
        source="comfy_quant",
        mode="pre_quantized",
        weight_bits=8,
        valid=True,
    )

    estimate = estimate_model_memory(size_bytes=1024**3, quantization=info)

    assert estimate.weights_bytes == 1024**3
    assert estimate.quantization_overhead_bytes == 0
    assert any("pre-quantized" in line for line in estimate.assumptions)
    assert any("8-bit" in line for line in estimate.assumptions)
    assert estimate.confidence == "medium"


def test_emulated_single_file_memory_expands_to_compute_width() -> None:
    info = QuantizationInfo(
        format="int8_w8a8",
        label="INT8 W8A8",
        runtime="comfy_kitchen",
        mode="emulated",
        weight_bits=8,
        valid=True,
    )

    estimate = estimate_model_memory(size_bytes=1024**3, quantization=info)

    assert estimate.weights_bytes == 2 * 1024**3
    assert any("expands" in line for line in estimate.assumptions)


def test_empty_model_is_marked_unknown() -> None:
    estimate = estimate_model_memory(size_bytes=0)

    assert estimate.unknown is True
    assert estimate.weights_bytes == 0


def test_workflow_estimator_deduplicates_models(tmp_path: Path) -> None:
    store = _store(tmp_path)

    estimate = estimate_workflow(
        _graph(_node("a", "diffusion/tiny.safetensors"), _node("b", "diffusion/tiny.safetensors")),
        store,
    )

    assert len(estimate.models) == 2
    assert estimate.weights_bytes == (store.root / "diffusion" / "tiny.safetensors").stat().st_size
    assert estimate.resident_estimate_bytes >= estimate.weights_bytes
    assert estimate.unknown is False


def test_workflow_estimator_reports_invalid_selection(tmp_path: Path) -> None:
    store = _store(tmp_path)

    estimate = estimate_workflow(_graph(_node("a", "diffusion/missing.safetensors")), store)

    assert len(estimate.models) == 1
    assert estimate.models[0].error is not None
    assert estimate.models[0].memory is None
    assert estimate.unknown is True
    assert estimate.weights_bytes == 0


def test_workflow_estimator_warns_for_missing_required_value(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    estimate = estimate_workflow(
        _graph(_node("a", None, code=_REQUIRED_NODE_CODE)), store
    )

    assert any("is not selected" in warning for warning in estimate.warnings)
    assert estimate.models == []


def test_workflow_estimate_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODELS_ENV, str(tmp_path / "library"))
    get_model_store.cache_clear()
    _store(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/estimate/workflow",
                json={
                    "workflow": {
                        "nodes": [
                            {
                                "id": "checkpoint",
                                "code": _NODE_CODE,
                                "widget_values": {
                                    "checkpoint_name": "diffusion/tiny.safetensors"
                                },
                            }
                        ],
                        "edges": [],
                    }
                },
            )
    finally:
        get_model_store.cache_clear()

    assert response.status_code == 200
    body = response.json()
    assert body["models"][0]["model_id"] == "diffusion/tiny.safetensors"
    assert body["weights_bytes"] > 0
    assert body["unknown"] is False
