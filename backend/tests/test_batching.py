"""Executor-level image batch broadcasting and cache fingerprint tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.executor import (
    MAX_BATCH_ITEMS,
    ExecutionCancelled,
    ExecutionResult,
    NodeExecutionError,
    build_output_preview,
    clear_cache,
    execute_graph,
)
from app.schemas import (
    GraphEdge,
    GraphNode,
    SocketType,
    WorkflowGraph,
    socket_types_compatible,
)
from app.security import SecurityRegistry, code_hash

SOURCE_BATCH = (
    "from PIL import Image\n"
    "images = [Image.new('RGB', (4, 4), (index, 0, 0)) for index in range(3)]\n"
    "return images: IMAGE_BATCH\n"
)

EMPTY_BATCH = (
    "from PIL import Image\n"
    "images = []\n"
    "return images: IMAGE_BATCH\n"
)

RESIZE_SINGLE = (
    "image: IMAGE\n"
    "result = image.resize((image.width + 1, image.height + 1))\n"
    "return result: IMAGE\n"
)


def _registry(*codes: str, node_ids: list[str] | None = None) -> SecurityRegistry:
    registry = SecurityRegistry()
    ids = node_ids or [f"node-{index}" for index in range(len(codes))]
    for node_id, code in zip(ids, codes, strict=True):
        registry.approve(node_id, code_hash(code))
    return registry


def _run(
    registry: SecurityRegistry,
    nodes: list[GraphNode],
    edges: list[GraphEdge] | None = None,
    *,
    should_stop=None,
    emit=None,
) -> ExecutionResult:
    clear_cache()
    try:
        return execute_graph(
            WorkflowGraph(nodes=nodes, edges=edges or []),
            security=registry,
            allow_installs=False,
            should_stop=should_stop,
            emit=emit,
        )
    finally:
        clear_cache()


def test_image_batch_broadcasts_over_scalar_image_inputs() -> None:
    registry = _registry(SOURCE_BATCH, RESIZE_SINGLE, node_ids=["source", "resize"])
    result = _run(
        registry,
        [
            GraphNode(id="source", code=SOURCE_BATCH),
            GraphNode(id="resize", code=RESIZE_SINGLE),
        ],
        [
            GraphEdge(
                source_node_id="source",
                source_handle="images",
                target_node_id="resize",
                target_handle="image",
            )
        ],
    )

    images = result.outputs_by_name["resize"]["result"]
    assert isinstance(images, list)
    assert len(images) == 3
    assert all(image.size == (5, 5) for image in images)
    assert result.previews["resize"]["result"]["kind"] == "image_batch"
    assert result.previews["resize"]["result"]["count"] == 3


def test_scalar_image_is_repeated_across_a_batch() -> None:
    scalar_source = (
        "from PIL import Image\n"
        "image = Image.new('RGB', (2, 2), 'white')\n"
        "return image: IMAGE\n"
    )
    code = (
        "from PIL import Image\n"
        "first: IMAGE\n"
        "second: IMAGE\n"
        "result = Image.new('RGB', (first.width + second.width, first.height))\n"
        "return result: IMAGE\n"
    )
    registry = _registry(
        SOURCE_BATCH,
        scalar_source,
        code,
        node_ids=["source", "scalar", "combine"],
    )
    result = _run(
        registry,
        [
            GraphNode(id="source", code=SOURCE_BATCH),
            GraphNode(id="scalar", code=scalar_source),
            GraphNode(id="combine", code=code),
        ],
        [
            GraphEdge(
                source_node_id="source",
                source_handle="images",
                target_node_id="combine",
                target_handle="first",
            ),
            GraphEdge(
                source_node_id="scalar",
                source_handle="image",
                target_node_id="combine",
                target_handle="second",
            ),
        ],
    )
    images = result.outputs_by_name["combine"]["result"]
    assert len(images) == 3
    assert all(image.size == (6, 4) for image in images)

def test_mismatched_batch_lengths_fail() -> None:
    registry = _registry(SOURCE_BATCH, node_ids=["source"])
    other = (
        "from PIL import Image\n"
        "images = [Image.new('RGB', (4, 4)) for _ in range(2)]\n"
        "return images: IMAGE_BATCH\n"
    )
    registry.approve("other", code_hash(other))
    code = (
        "first: IMAGE\n"
        "second: IMAGE\n"
        "result = first\n"
        "return result: IMAGE\n"
    )
    registry.approve("combine", code_hash(code))
    with pytest.raises(NodeExecutionError, match="matching lengths"):
        _run(
            registry,
            [
                GraphNode(id="source", code=SOURCE_BATCH),
                GraphNode(id="other", code=other),
                GraphNode(id="combine", code=code),
            ],
            [
                GraphEdge(
                    source_node_id="source",
                    source_handle="images",
                    target_node_id="combine",
                    target_handle="first",
                ),
                GraphEdge(
                    source_node_id="other",
                    source_handle="images",
                    target_node_id="combine",
                    target_handle="second",
                ),
            ],
        )


def test_empty_batch_produces_empty_outputs() -> None:
    registry = _registry(EMPTY_BATCH, RESIZE_SINGLE, node_ids=["source", "resize"])
    result = _run(
        registry,
        [
            GraphNode(id="source", code=EMPTY_BATCH),
            GraphNode(id="resize", code=RESIZE_SINGLE),
        ],
        [
            GraphEdge(
                source_node_id="source",
                source_handle="images",
                target_node_id="resize",
                target_handle="image",
            )
        ],
    )
    assert result.outputs_by_name["resize"]["result"] == []
    assert result.previews["resize"]["result"]["kind"] == "image_batch"
    assert result.previews["resize"]["result"]["count"] == 0


def test_batch_errors_report_the_item_index() -> None:
    code = (
        "image: IMAGE\n"
        "if image.getpixel((0, 0))[0] == 1:\n"
        "    raise ValueError('bad item')\n"
        "return image: IMAGE\n"
    )
    registry = _registry(SOURCE_BATCH, code, node_ids=["source", "fail"])
    events: list[object] = []
    with pytest.raises(NodeExecutionError) as excinfo:
        _run(
            registry,
            [
                GraphNode(id="source", code=SOURCE_BATCH),
                GraphNode(id="fail", code=code),
            ],
            [
                GraphEdge(
                    source_node_id="source",
                    source_handle="images",
                    target_node_id="fail",
                    target_handle="image",
                )
            ],
            emit=events.append,
        )
    assert "Batch item 2/3" in excinfo.value.message
    error_events = [
        event
        for event in events
        if getattr(event, "type", None).value == "NODE_ERROR"
    ]
    assert error_events[-1].data["batch_index"] == 1
    assert error_events[-1].data["batch_size"] == 3


def test_batch_progress_is_mapped_per_item() -> None:
    code = (
        "image: IMAGE\n"
        "report_progress(0.5, 'half')\n"
        "return image: IMAGE\n"
    )
    registry = _registry(SOURCE_BATCH, code, node_ids=["source", "progress"])
    progress: list[float] = []
    _run(
        registry,
        [
            GraphNode(id="source", code=SOURCE_BATCH),
            GraphNode(id="progress", code=code),
        ],
        [
            GraphEdge(
                source_node_id="source",
                source_handle="images",
                target_node_id="progress",
                target_handle="image",
            )
        ],
        emit=lambda event: (
            progress.append(float(event.data["progress"]))
            if event.type.value == "NODE_PROGRESS"
            else None
        ),
    )
    assert progress == pytest.approx([1 / 6, 1 / 2, 5 / 6])


def test_cancellation_is_checked_between_batch_items() -> None:
    registry = _registry(SOURCE_BATCH, RESIZE_SINGLE, node_ids=["source", "resize"])
    calls = {"count": 0}

    def should_stop() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    with pytest.raises(ExecutionCancelled):
        _run(
            registry,
            [
                GraphNode(id="source", code=SOURCE_BATCH),
                GraphNode(id="resize", code=RESIZE_SINGLE),
            ],
            [
                GraphEdge(
                    source_node_id="source",
                    source_handle="images",
                    target_node_id="resize",
                    target_handle="image",
                )
            ],
            should_stop=should_stop,
        )


def test_batch_item_limit() -> None:
    code = (
        "from PIL import Image\n"
        f"images = [Image.new('RGB', (1, 1)) for _ in range({MAX_BATCH_ITEMS + 1})]\n"
        "return images: IMAGE_BATCH\n"
    )
    registry = _registry(code, RESIZE_SINGLE, node_ids=["source", "resize"])
    with pytest.raises(NodeExecutionError, match="limit"):
        _run(
            registry,
            [
                GraphNode(id="source", code=code),
                GraphNode(id="resize", code=RESIZE_SINGLE),
            ],
            [
                GraphEdge(
                    source_node_id="source",
                    source_handle="images",
                    target_node_id="resize",
                    target_handle="image",
                )
            ],
        )


def test_batch_item_limit_applies_to_source_outputs() -> None:
    code = (
        "from PIL import Image\n"
        f"images = [Image.new('RGB', (1, 1)) for _ in range({MAX_BATCH_ITEMS + 1})]\n"
        "return images: IMAGE_BATCH\n"
    )
    registry = _registry(code, node_ids=["source"])
    with pytest.raises(NodeExecutionError, match="limit"):
        _run(registry, [GraphNode(id="source", code=code)])


def test_generic_batch_previews() -> None:
    preview = build_output_preview(
        SocketType.DETECTIONS,
        [{"count": 1}, {"count": 2}],
    )
    assert preview == {"kind": "batch", "count": 2}
    image = Image.new("RGB", (2, 2))
    any_preview = build_output_preview(SocketType.ANY, [image, image])
    assert any_preview["kind"] == "image_batch"
    assert any_preview["count"] == 2
    large = build_output_preview(SocketType.ANY, {"blob": "x" * 30_000})
    assert isinstance(large, str)
    assert len(large) <= 200
    text = build_output_preview(SocketType.ANY, "x" * 30_000)
    assert isinstance(text, str)
    assert len(text) <= 200


def test_folder_widget_fingerprint_invalidates_cache(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"one")
    code = (
        "from pathlib import Path\n"
        'folder: FolderPath = ""\n'
        "count = len([item for item in Path(folder).iterdir() if item.is_file()])\n"
        "return count: INT\n"
    )
    registry = _registry(code, node_ids=["count"])
    node = GraphNode(
        id="count",
        code=code,
        widget_values={"folder": str(folder)},
    )
    clear_cache()
    try:
        first = execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            allow_installs=False,
        )
        assert first.outputs_by_name["count"]["count"] == 1

        (folder / "b.png").write_bytes(b"two")
        second = execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            allow_installs=False,
        )
        assert second.outputs_by_name["count"]["count"] == 2
    finally:
        clear_cache()


def test_image_batch_is_compatible_with_image_socket() -> None:
    assert socket_types_compatible(SocketType.IMAGE_BATCH, SocketType.IMAGE)
    assert socket_types_compatible(SocketType.IMAGE, SocketType.IMAGE_BATCH)


def test_connected_folder_widget_fingerprint_invalidates_cache(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"one")
    source = f"folder = {str(folder)!r}\nreturn folder: STRING"
    loader = (
        "from pathlib import Path\n"
        'folder: FolderPath = ""\n'
        "images = [item.name for item in Path(folder).iterdir() if item.is_file()]\n"
        "return images: IMAGE_BATCH\n"
    )
    registry = _registry(source, loader, node_ids=["source", "load"])
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="source", code=source),
            GraphNode(id="load", code=loader),
        ],
        edges=[
            GraphEdge(
                source_node_id="source",
                source_handle="folder",
                target_node_id="load",
                target_handle="folder",
            )
        ],
    )
    clear_cache()
    try:
        first = execute_graph(graph, security=registry, allow_installs=False)
        assert first.outputs_by_name["load"]["images"] == ["a.png"]

        (folder / "b.png").write_bytes(b"two")
        second = execute_graph(graph, security=registry, allow_installs=False)
        assert second.outputs_by_name["load"]["images"] == ["a.png", "b.png"]
    finally:
        clear_cache()
