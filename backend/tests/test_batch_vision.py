"""Folder loader and batch vision node tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.executor import NodeExecutionError, clear_cache, execute_graph
from app.schemas import GraphEdge, GraphNode, WorkflowGraph
from app.security import SecurityRegistry


def _image(path: Path, color: str) -> None:
    Image.new("RGB", (5, 4), color).save(path)


def _run_loader(folder: Path, **widgets: object):
    registry = SecurityRegistry()
    node = GraphNode(
        id="load",
        code=registry.get_template("load_images_from_folder").code,
        widget_values={"folder": str(folder), **widgets},
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


def test_folder_loader_returns_images_in_sorted_order(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    _image(folder / "b.png", "green")
    _image(folder / "a.jpg", "red")
    (folder / "notes.txt").write_text("ignore", encoding="utf-8")

    result = _run_loader(folder)

    images = result.outputs_by_name["load"]["images"]
    assert isinstance(images, list)
    assert len(images) == 2
    assert all(image.size == (5, 4) for image in images)
    assert result.previews["load"]["images"]["kind"] == "image_batch"
    assert result.previews["load"]["images"]["count"] == 2


def test_folder_loader_rejects_empty_and_limited_folders(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NodeExecutionError, match="No supported images"):
        _run_loader(empty)

    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(3):
        _image(folder / f"{index}.png", "red")
    with pytest.raises(NodeExecutionError, match="increase max_images"):
        _run_loader(folder, max_images=2)


def test_folder_loader_cache_invalidates_when_files_change(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    _image(folder / "a.png", "red")
    registry = SecurityRegistry()
    node = GraphNode(
        id="load",
        code=registry.get_template("load_images_from_folder").code,
        widget_values={"folder": str(folder)},
    )
    clear_cache()
    try:
        first = execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            allow_installs=False,
        )
        assert len(first.outputs_by_name["load"]["images"]) == 1

        _image(folder / "b.png", "green")
        second = execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            allow_installs=False,
        )
        assert len(second.outputs_by_name["load"]["images"]) == 2
    finally:
        clear_cache()

def test_folder_loader_broadcasts_through_resize_node(tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    _image(folder / "a.png", "red")
    _image(folder / "b.png", "green")
    registry = SecurityRegistry()
    load = GraphNode(
        id="load",
        code=registry.get_template("load_images_from_folder").code,
        widget_values={"folder": str(folder)},
    )
    resize = GraphNode(
        id="resize",
        code=registry.get_template("resize_by_side").code,
        widget_values={
            "side": "longer",
            "side_resolution": 10,
            "round_to": 0,
            "aspect_ratio": "keep",
            "allow_upscale": True,
        },
    )
    clear_cache()
    try:
        result = execute_graph(
            WorkflowGraph(
                nodes=[load, resize],
                edges=[
                    GraphEdge(
                        source_node_id="load",
                        source_handle="images",
                        target_node_id="resize",
                        target_handle="image",
                    )
                ],
            ),
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()

    images = result.outputs_by_name["resize"]["result"]
    assert isinstance(images, list)
    assert len(images) == 2
    assert all(image.width == 10 for image in images)
    assert result.previews["resize"]["result"]["kind"] == "image_batch"
