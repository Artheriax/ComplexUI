"""Switch node tests: fixed inputs, labels and selected-value behavior."""

from __future__ import annotations

import pytest
from PIL import Image

from app.executor import NodeExecutionError, clear_cache, execute_graph
from app.schemas import GraphEdge, GraphNode, WorkflowGraph
from app.security import SecurityRegistry, code_hash


def _run_switch(widget_values: dict[str, object]):
    registry = SecurityRegistry()
    node = GraphNode(
        id="switch",
        code=registry.get_template("switch").code,
        widget_values=widget_values,
    )
    clear_cache()
    try:
        return execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()


def test_switch_returns_the_selected_input() -> None:
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (4, 4), "green")
    result = _run_switch(
        {
            "input_1": first,
            "input_2": second,
            "choices": "Red|Green|Blue|Cyan|Magenta",
            "selected": 1,
        }
    )

    assert result.outputs_by_name["switch"]["chosen"] is second
    assert result.previews["switch"]["chosen"]["kind"] == "image"
    assert result.previews["switch"]["chosen"]["width"] == 4


def test_switch_accepts_values_through_any_widget_handles() -> None:
    source = (
        "from PIL import Image\n"
        "image = Image.new('RGB', (6, 5), 'blue')\n"
        "return image: IMAGE\n"
    )
    registry = SecurityRegistry()
    registry.approve("source", code_hash(source))
    node = GraphNode(
        id="switch",
        code=registry.get_template("switch").code,
        widget_values={"choices": "First|Second|Third|Fourth|Fifth", "selected": 2},
    )
    clear_cache()
    try:
        result = execute_graph(
            WorkflowGraph(
                nodes=[GraphNode(id="source", code=source), node],
                edges=[
                    GraphEdge(
                        source_node_id="source",
                        source_handle="image",
                        target_node_id="switch",
                        target_handle="input_3",
                    )
                ],
            ),
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()

    assert result.outputs_by_name["switch"]["chosen"].size == (6, 5)


def test_switch_ignores_unconnected_inputs_that_are_not_selected() -> None:
    first = Image.new("RGB", (3, 3))
    result = _run_switch(
        {"input_1": first, "choices": "One|Two|Three|Four|Five", "selected": 0}
    )
    assert result.outputs_by_name["switch"]["chosen"] is first


def test_switch_rejects_an_unconnected_selected_input() -> None:
    with pytest.raises(NodeExecutionError) as excinfo:
        _run_switch(
            {
                "input_1": Image.new("RGB", (2, 2)),
                "choices": "One|Two|Three|Four|Five",
                "selected": 1,
            }
        )
    assert "Switch input 'Two' is not connected" in excinfo.value.message


def test_switch_rejects_out_of_range_selection() -> None:
    with pytest.raises(NodeExecutionError, match="between 0 and 4"):
        _run_switch(
            {"input_1": Image.new("RGB", (2, 2)), "choices": "", "selected": 7}
        )
