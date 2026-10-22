"""Image/video asset uploads, raw serving and OutputFile resolution."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import datasets as datasets_module
from app.datasets import get_dataset_store, get_output_store
from app.executor import NodeExecutionError, clear_cache, execute_graph
from app.main import app
from app.schemas import GraphEdge, GraphNode, WorkflowGraph
from app.security import SecurityRegistry, code_hash
from app.transpiler import parse_node

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.fixture()
def media_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COMPLEXUI_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("COMPLEXUI_INPUT_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("COMPLEXUI_OUTPUT_DIR", str(tmp_path / "outputs"))
    get_dataset_store.cache_clear()
    get_output_store.cache_clear()
    with TestClient(app) as client:
        yield client
    get_dataset_store.cache_clear()
    get_output_store.cache_clear()


def test_default_output_dir_is_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COMPLEXUI_OUTPUT_DIR", raising=False)
    expected = Path(__file__).resolve().parents[2] / "outputs"
    assert datasets_module.OutputStore().outputs_dir == expected


def test_image_upload_is_listed_and_served_inline(media_client: TestClient) -> None:
    response = media_client.post(
        "/api/uploads",
        files={"file": ("sample.png", _PNG, "image/png")},
    )
    assert response.status_code == 200, response.text
    asset = response.json()["dataset"]
    assert asset["kind"] == "image"
    assert asset["format"] == "PNG"

    listed = media_client.get("/api/uploads").json()
    assert [(item["id"], item["kind"]) for item in listed] == [("sample.png", "image")]

    raw = media_client.get("/api/uploads/sample.png/raw")
    assert raw.status_code == 200
    assert raw.headers["content-type"].startswith("image/png")
    assert raw.headers["content-disposition"].startswith("inline;")
    assert raw.content == _PNG

    ranged = media_client.get(
        "/api/uploads/sample.png/raw",
        headers={"Range": "bytes=0-9"},
    )
    assert ranged.status_code == 206
    assert ranged.content == _PNG[:10]

    preview = media_client.get("/api/uploads/sample.png/preview")
    assert preview.status_code == 422


def test_video_upload_uses_media_limit(
    media_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded = media_client.post(
        "/api/uploads",
        files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["dataset"]["kind"] == "video"
    assert media_client.get("/api/uploads/clip.mp4/raw").content == b"video-bytes"

    monkeypatch.setattr(datasets_module, "MAX_MEDIA_BYTES", 4)
    too_large = media_client.post(
        "/api/uploads",
        files={"file": ("large.mp4", b"x" * 16, "video/mp4")},
    )
    assert too_large.status_code == 413


def test_media_upload_rejects_svg(media_client: TestClient) -> None:
    response = media_client.post(
        "/api/uploads",
        files={"file": ("vector.svg", b"<svg/>", "image/svg+xml")},
    )
    assert response.status_code == 422


def test_generated_outputs_are_served_and_confined(
    media_client: TestClient,
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "result.png").write_bytes(_PNG)

    response = media_client.get("/api/outputs/result.png/raw")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == _PNG

    assert media_client.get("/api/outputs/missing.png/raw").status_code == 404
    assert media_client.get("/api/outputs/%2E%2E%2Fsecret.png/raw").status_code in {
        404,
        422,
    }


def test_generated_outputs_are_listed_and_deleted(
    media_client: TestClient,
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "result.png").write_bytes(_PNG)
    (outputs / "clip.mp4").write_bytes(b"video-bytes")
    (outputs / "notes.txt").write_text("not media", encoding="utf-8")

    listed = media_client.get("/api/outputs")
    assert listed.status_code == 200
    items = {item["name"]: item for item in listed.json()}
    assert set(items) == {"result.png", "clip.mp4"}
    assert items["result.png"]["path"] == "outputs/result.png"
    assert items["result.png"]["kind"] == "image"
    assert items["clip.mp4"]["kind"] == "video"

    deleted = media_client.delete("/api/outputs/result.png")
    assert deleted.status_code == 200
    assert deleted.json()["output"]["name"] == "result.png"
    assert not (outputs / "result.png").exists()
    assert (outputs / "clip.mp4").exists()

    assert media_client.delete("/api/outputs/missing.png").status_code == 404
    assert media_client.delete("/api/outputs/%2E%2E%2Fsecret.png").status_code in {
        404,
        422,
    }


def test_imagefile_and_outputfile_widgets_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "input.png").write_bytes(_PNG)
    monkeypatch.setenv("COMPLEXUI_INPUT_DIR", str(inputs))
    get_dataset_store.cache_clear()
    get_output_store.cache_clear()

    code = (
        "from pathlib import Path\n"
        "from PIL import Image\n"
        'source: ImageFile = ""\n'
        'target: OutputFile = ""\n'
        "with Image.open(source) as image:\n"
        "    image.load()\n"
        "    image.save(target)\n"
        "name = Path(target).name\n"
        "return name: STRING\n"
    )
    schema = parse_node(code)
    assert schema.widgets[0].data_type == "ImageFile"
    assert schema.widgets[1].data_type == "OutputFile"

    node = GraphNode(
        id="copy_image",
        code=code,
        widget_values={"source": "inputs/input.png", "target": "copied.png"},
    )
    registry = SecurityRegistry()
    registry.approve(node.id, code_hash(node.code))
    clear_cache()
    try:
        result = execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()
    assert result.outputs_by_name[node.id]["name"] == "copied.png"
    assert (get_output_store().outputs_dir / "copied.png").is_file()


def test_load_video_and_extract_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    video_path = inputs / "clip.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (80, 64),
    )
    assert writer.isOpened()
    for index in range(4):
        frame = np.full((64, 80, 3), (index * 40, 30, 180), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    monkeypatch.setenv("COMPLEXUI_INPUT_DIR", str(inputs))
    get_dataset_store.cache_clear()
    registry = SecurityRegistry()
    nodes = [
        GraphNode(
            id="load",
            code=registry.get_template("load_video").code,
            widget_values={"path": "inputs/clip.mp4"},
        ),
        GraphNode(
            id="frame",
            code=registry.get_template("extract_video_frame").code,
            widget_values={"frame_index": 2},
        ),
    ]
    edges = [
        GraphEdge(
            source_node_id="load",
            source_handle="video",
            target_node_id="frame",
            target_handle="video",
        )
    ]
    clear_cache()
    try:
        result = execute_graph(
            WorkflowGraph(nodes=nodes, edges=edges),
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()

    video = result.outputs_by_name["load"]["video"]
    assert video["media_type"] == "video"
    assert video["frames"] == 4
    assert video["fps"] == pytest.approx(10.0)
    assert [video["width"], video["height"]] == [80, 64]
    assert result.previews["load"]["video"]["kind"] == "video"
    assert result.previews["load"]["video"]["url"] == "/api/uploads/clip.mp4/raw"

    image = result.outputs_by_name["frame"]["image"]
    assert image.mode == "RGB"
    assert image.size == (80, 64)
    assert result.previews["frame"]["image"]["kind"] == "image"


def test_outputfile_rejects_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("COMPLEXUI_WORKSPACE", str(workspace))
    get_output_store.cache_clear()
    code = (
        "from pathlib import Path\n"
        'target: OutputFile = ""\n'
        "Path(target).write_text('bad')\n"
        "return target: STRING\n"
    )
    node = GraphNode(
        id="escape",
        code=code,
        widget_values={"target": "../outside.txt"},
    )
    registry = SecurityRegistry()
    registry.approve(node.id, code_hash(node.code))

    with pytest.raises(NodeExecutionError) as excinfo:
        execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            allow_installs=False,
        )
    assert excinfo.value.error_type == "FileAccessError"


@pytest.mark.parametrize(
    ("template_name", "expected_kind"),
    [("make_demo_image", "image"), ("make_demo_video", "video")],
)
def test_preview_media_passes_media_without_saving(
    template_name: str,
    expected_kind: str,
) -> None:
    registry = SecurityRegistry()
    source = GraphNode(
        id="source",
        code=registry.get_template(template_name).code,
    )
    preview = GraphNode(
        id="preview",
        code=registry.get_template("preview_media").code,
    )
    graph = WorkflowGraph(
        nodes=[source, preview],
        edges=[
            GraphEdge(
                source_node_id="source",
                source_handle="image" if expected_kind == "image" else "video",
                target_node_id="preview",
                target_handle="media",
            )
        ],
    )

    clear_cache()
    try:
        result = execute_graph(
            graph,
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()

    assert result.previews["preview"]["media"]["kind"] == expected_kind
    assert "saved_path" not in result.outputs_by_name["preview"]
