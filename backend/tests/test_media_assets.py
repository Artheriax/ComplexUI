"""Image/video asset uploads, raw serving and OutputFile resolution."""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import datasets as datasets_module
from app import main as main_module
from app.artifacts import get_output_artifact_registry
from app.datasets import get_dataset_store, get_output_store
from app.executor import (
    ExecutionResult,
    NodeExecutionError,
    clear_cache,
    execute_graph,
)
from app.main import app
from app.run_history import get_run_history_store
from app.schemas import (
    GraphEdge,
    GraphNode,
    RunNodeRecord,
    RunNodeStatus,
    RunRecord,
    RunStatus,
    WorkflowGraph,
)
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


def test_generated_outputs_support_date_folders(
    media_client: TestClient,
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    day = outputs / "2026-10-23"
    day.mkdir(parents=True)
    nested = day / "shots"
    nested.mkdir()
    (day / "dated.png").write_bytes(_PNG)
    (nested / "deep.png").write_bytes(_PNG)
    (outputs / "root.png").write_bytes(_PNG)

    listed = media_client.get("/api/outputs")
    assert listed.status_code == 200
    items = {item["id"]: item for item in listed.json()}
    assert set(items) == {
        "2026-10-23/dated.png",
        "2026-10-23/shots/deep.png",
        "root.png",
    }
    assert items["2026-10-23/dated.png"]["name"] == "dated.png"
    assert items["2026-10-23/dated.png"]["path"] == "outputs/2026-10-23/dated.png"

    raw = media_client.get("/api/outputs/2026-10-23/dated.png/raw")
    assert raw.status_code == 200
    assert raw.content == _PNG
    nested_raw = media_client.get("/api/outputs/2026-10-23/shots/deep.png/raw")
    assert nested_raw.status_code == 200

    deleted = media_client.delete("/api/outputs/2026-10-23/dated.png")
    assert deleted.status_code == 200
    assert deleted.json()["output"]["id"] == "2026-10-23/dated.png"
    assert not (day / "dated.png").exists()

    assert media_client.get("/api/outputs/2026-10-23/missing.png/raw").status_code == 404
    escaped = media_client.get("/api/outputs/%2E%2E%2Fsecret.png/raw")
    assert escaped.status_code in {404, 422}


def test_outputfile_and_imagefile_support_date_folders() -> None:
    outputs = get_output_store().outputs_dir
    day = outputs / "2026-10-23"
    day.mkdir(parents=True)
    (day / "dated.png").write_bytes(_PNG)

    registry = SecurityRegistry()
    node = GraphNode(
        id="load",
        code=registry.get_template("load_image").code,
        widget_values={"path": "outputs/2026-10-23/dated.png"},
    )
    clear_cache()
    try:
        result = execute_graph(
            WorkflowGraph(nodes=[node]),
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()
    assert result.outputs_by_name["load"]["image"].size == (1, 1)

    code = (
        "from pathlib import Path\n"
        'target: OutputFile = ""\n'
        "Path(target).write_text('ok', encoding='utf-8')\n"
        "result = str(target)\n"
        "return result: STRING\n"
    )
    writer = GraphNode(
        id="write",
        code=code,
        widget_values={"target": "2026-10-23/nested.txt"},
    )
    registry = SecurityRegistry()
    registry.approve(writer.id, code_hash(writer.code))
    clear_cache()
    try:
        execute_graph(
            WorkflowGraph(nodes=[writer]),
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()
    assert (day / "nested.txt").read_text(encoding="utf-8") == "ok"


def _save_result(
    *,
    model_name: str = "demo",
    source_code: str | None = None,
    source_handle: str = "image",
    save_widgets: dict[str, object] | None = None,
) -> ExecutionResult:
    registry = SecurityRegistry()
    source = GraphNode(
        id="source",
        code=source_code or registry.get_template("make_demo_image").code,
        widget_values={} if source_code else {"width": 64, "height": 64},
    )
    if source_code:
        registry.approve(source.id, code_hash(source.code))
    save = GraphNode(
        id="save",
        code=registry.get_template("save_media").code,
        widget_values={"model_name": model_name, **(save_widgets or {})},
    )
    graph = WorkflowGraph(
        nodes=[source, save],
        edges=[
            GraphEdge(
                source_node_id="source",
                source_handle=source_handle,
                target_node_id="save",
                target_handle="media",
            )
        ],
    )
    clear_cache()
    try:
        return execute_graph(
            graph,
            security=registry,
            allow_installs=False,
        )
    finally:
        clear_cache()


def test_save_media_uses_dated_model_counter_names() -> None:
    day = datetime.now().strftime("%Y-%m-%d")
    result = _save_result()
    saved_path = result.outputs_by_name["save"]["saved_path"]
    assert saved_path == f"outputs/{day}/demo_001.png"
    target = get_output_store().outputs_dir / day / "demo_001.png"
    assert target.is_file()
    assert result.previews["save"]["saved"]["kind"] == "image"

    result = _save_result()
    assert result.outputs_by_name["save"]["saved_path"] == (
        f"outputs/{day}/demo_002.png"
    )

    _save_result(model_name="demo-jpeg", save_widgets={"format": "jpg", "quality": 80})
    target = get_output_store().outputs_dir / day / "demo-jpeg_001.jpg"
    assert target.is_file()
    with Image.open(target) as saved:
        assert saved.format == "JPEG"

    other = _save_result(model_name="other")
    assert other.outputs_by_name["save"]["saved_path"] == (
        f"outputs/{day}/other_001.png"
    )


def test_save_media_indexes_batches() -> None:
    day = datetime.now().strftime("%Y-%m-%d")
    code = (
        "from PIL import Image\n"
        "images = [\n"
        "    Image.new('RGB', (4, 4), (255, 0, 0)),\n"
        "    Image.new('RGB', (4, 4), (0, 255, 0)),\n"
        "]\n"
        "return images: IMAGE_BATCH\n"
    )
    result = _save_result(
        model_name="batch",
        source_code=code,
        source_handle="images",
    )
    assert result.outputs_by_name["save"]["saved_path"] == (
        f"outputs/{day}/batch_001_000.png"
    )
    directory = get_output_store().outputs_dir / day
    assert (directory / "batch_001_000.png").is_file()
    assert (directory / "batch_001_001.png").is_file()
    assert result.previews["save"]["saved"]["kind"] == "image_batch"


def test_output_open_and_reveal_actions(
    media_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    day = tmp_path / "outputs" / "2026-10-23"
    day.mkdir(parents=True)
    (day / "dated.png").write_bytes(_PNG)
    opened: list[Path] = []
    revealed: list[Path] = []
    monkeypatch.setattr(main_module, "open_path", opened.append)
    monkeypatch.setattr(main_module, "reveal_path", revealed.append)

    opened_response = media_client.post("/api/outputs/2026-10-23/dated.png/open")
    assert opened_response.status_code == 200
    assert opened_response.json() == {"status": "opened"}
    assert [path.name for path in opened] == ["dated.png"]

    revealed_response = media_client.post("/api/outputs/2026-10-23/dated.png/reveal")
    assert revealed_response.status_code == 200
    assert revealed_response.json() == {"status": "revealed"}
    assert [path.name for path in revealed] == ["dated.png"]

    assert media_client.post("/api/outputs/2026-10-23/missing.png/open").status_code == 404

    monkeypatch.setenv("COMPLEXUI_ALLOWED_HOSTS", "*")
    blocked = media_client.post("/api/outputs/2026-10-23/dated.png/open")
    assert blocked.status_code == 403

def test_save_media_handles_videos_with_dated_names() -> None:
    day = datetime.now().strftime("%Y-%m-%d")
    registry = SecurityRegistry()
    source = GraphNode(
        id="source",
        code=registry.get_template("make_demo_video").code,
        widget_values={"width": 64, "height": 64, "fps": 4, "frames": 2},
    )
    save = GraphNode(
        id="save",
        code=registry.get_template("save_media").code,
        widget_values={"model_name": "demo-video"},
    )
    graph = WorkflowGraph(
        nodes=[source, save],
        edges=[
            GraphEdge(
                source_node_id="source",
                source_handle="video",
                target_node_id="save",
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

    saved_path = result.outputs_by_name["save"]["saved_path"]
    assert saved_path == f"outputs/{day}/demo-video_001.mp4"
    target = get_output_store().outputs_dir / day / "demo-video_001.mp4"
    assert target.is_file()
    assert result.outputs_by_name["save"]["saved"]["relative_path"] == saved_path
    assert result.previews["save"]["saved"]["kind"] == "video"
    assert result.previews["save"]["saved"]["url"] == (
        f"/api/outputs/{day}/demo-video_001.mp4/raw"
    )

def test_save_media_supports_external_folders_and_tokens(tmp_path: Path) -> None:
    external = tmp_path / "exports" / "Akali video"
    day = datetime.now().strftime("%Y-%m-%d")
    result = _save_result(
        model_name="external",
        save_widgets={
            "folder": str(external),
            "filename": "%model%_%date%_%width%x%height%_%counter%",
        },
    )

    expected = (external / f"external_{day}_64x64_001.png").resolve()
    assert result.outputs_by_name["save"]["saved_path"] == str(expected)
    assert expected.is_file()
    assert result.previews["save"]["saved"]["kind"] == "image"


def test_save_media_supports_nested_output_folders() -> None:
    day = datetime.now().strftime("%Y-%m-%d")
    result = _save_result(
        model_name="nested",
        save_widgets={"filename": "shots/%model%/%date%/%counter%"},
    )

    expected = f"outputs/{day}/shots/nested/{day}/001.png"
    assert result.outputs_by_name["save"]["saved_path"] == expected
    listed = {item.id for item in get_output_store().list_outputs()}
    assert expected.removeprefix("outputs/") in listed


def test_save_media_rejects_unknown_and_metadata_tokens() -> None:
    with pytest.raises(NodeExecutionError) as excinfo:
        _save_result(model_name="bad", save_widgets={"filename": "%seed%_%counter%"})
    assert "metadata" in excinfo.value.message

    with pytest.raises(NodeExecutionError) as excinfo:
        _save_result(model_name="bad", save_widgets={"filename": "%nope%_%counter%"})
    assert "Unknown" in excinfo.value.message

    with pytest.raises(NodeExecutionError) as excinfo:
        _save_result(model_name="bad", save_widgets={"filename": "no counter"})
    assert "must include %counter%" in excinfo.value.message


def test_artifact_endpoints_only_serve_run_produced_paths(
    media_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = tmp_path / "outside"
    external.mkdir()
    artifact = external / "result.png"
    artifact.write_bytes(_PNG)
    get_output_artifact_registry().register(artifact)

    raw = media_client.get("/api/artifacts/raw", params={"path": str(artifact)})
    assert raw.status_code == 200
    assert raw.content == _PNG

    unregistered = external / "other.png"
    unregistered.write_bytes(_PNG)
    assert (
        media_client.get("/api/artifacts/raw", params={"path": str(unregistered)}).status_code
        == 404
    )

    opened: list[Path] = []
    revealed: list[Path] = []
    monkeypatch.setattr(main_module, "open_path", opened.append)
    monkeypatch.setattr(main_module, "reveal_path", revealed.append)
    assert media_client.post(
        "/api/artifacts/open", json={"path": str(artifact)}
    ).status_code == 200
    assert media_client.post(
        "/api/artifacts/reveal", json={"path": str(artifact)}
    ).status_code == 200
    assert [path.name for path in opened] == ["result.png"]
    assert [path.name for path in revealed] == ["result.png"]

    monkeypatch.setenv("COMPLEXUI_ALLOWED_HOSTS", "*")
    blocked = media_client.get("/api/artifacts/raw", params={"path": str(artifact)})
    assert blocked.status_code == 403


def test_run_history_saved_paths_are_available(
    tmp_path: Path,
) -> None:
    external = tmp_path / "history-output.png"
    external.write_bytes(_PNG)
    store = get_run_history_store()
    record = RunRecord(
        run_id="a" * 32,
        status=RunStatus.COMPLETED,
        started_at="2026-10-23T12:00:00Z",
        finished_at="2026-10-23T12:00:01Z",
        duration_ms=1000,
        node_count=1,
        completed_count=1,
        cached_count=0,
        order=["save"],
        node_labels={"save": "Save Media"},
        nodes=[
            RunNodeRecord(
                node_id="save",
                label="Save Media",
                status=RunNodeStatus.COMPLETED,
                duration_ms=1000,
                cached=False,
                previews={"saved_path": str(external)},
            )
        ],
    )
    store.save(record)

    assert str(external) in store.saved_paths()

def test_native_folder_picker_endpoint(
    media_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "chosen"
    selected.mkdir()
    monkeypatch.setattr(main_module, "pick_folder", lambda initial: str(selected))

    response = media_client.post("/api/fs/pick-folder", json={"initial": None})
    assert response.status_code == 200
    assert response.json() == {"path": str(selected)}

    monkeypatch.setattr(main_module, "pick_folder", lambda initial: None)
    canceled = media_client.post("/api/fs/pick-folder", json={})
    assert canceled.status_code == 200
    assert canceled.json() == {"path": None}

    def busy(initial: str | None) -> str | None:
        raise main_module.FolderDialogBusy("busy")

    monkeypatch.setattr(main_module, "pick_folder", busy)
    assert media_client.post("/api/fs/pick-folder", json={}).status_code == 409

    def unavailable(initial: str | None) -> str | None:
        raise main_module.PathActionError("no dialog")

    monkeypatch.setattr(main_module, "pick_folder", unavailable)
    assert media_client.post("/api/fs/pick-folder", json={}).status_code == 501

    monkeypatch.setenv("COMPLEXUI_ALLOWED_HOSTS", "*")
    assert media_client.post("/api/fs/pick-folder", json={}).status_code == 403

def test_external_save_is_available_through_artifact_api(
    media_client: TestClient,
    tmp_path: Path,
) -> None:
    registry = SecurityRegistry()
    external = tmp_path / "external"
    external.mkdir()
    graph = {
        "nodes": [
            {
                "id": "demo",
                "code": registry.get_template("make_demo_image").code,
                "widget_values": {"width": 64, "height": 64},
            },
            {
                "id": "save",
                "code": registry.get_template("save_media").code,
                "widget_values": {
                    "folder": str(external),
                    "filename": "%model%_%counter%",
                    "model_name": "api-external",
                },
            },
        ],
        "edges": [
            {
                "source_node_id": "demo",
                "source_handle": "image",
                "target_node_id": "save",
                "target_handle": "media",
            }
        ],
    }
    with media_client.websocket_connect("/ws/execution") as websocket:
        response = media_client.post("/api/execute", json={"graph": graph})
        assert response.status_code == 202
        events = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] in {
                "EXECUTION_COMPLETED",
                "EXECUTION_FAILED",
                "EXECUTION_CANCELLED",
            }:
                break

    completed = next(
        event
        for event in events
        if event["type"] == "NODE_COMPLETED" and event["node_id"] == "save"
    )
    saved_path = completed["data"]["outputs"]["saved_path"]
    assert Path(saved_path).is_file()
    assert Path(saved_path).parent == external.resolve()

    raw = media_client.get("/api/artifacts/raw", params={"path": saved_path})
    assert raw.status_code == 200
    assert raw.content == getattr(completed, "content", raw.content)

    history = get_run_history_store().saved_paths()
    assert saved_path in history
