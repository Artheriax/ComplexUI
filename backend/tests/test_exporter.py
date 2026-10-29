"""Tests for standalone Python and Jupyter notebook exports."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.datasets import get_output_store
from app.exporter import ExportError, export_notebook, export_python
from app.schemas import GraphEdge, GraphNode, WorkflowGraph
from app.security import SecurityRegistry

GENERATED_AT = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)

SOURCE_CODE = """# /// script
# dependencies = ["zeta>=1", "shared>=1"]
# ///
import math

value: int = 3

result = math.ceil(value / 2)
return result: ANY
"""

MIDDLE_CODE = """# /// script
# dependencies = ["shared>=1", "alpha>=2"]
# ///
x: ANY
offset: int = 1

out = x + offset
print("computed", out)
return out: ANY
"""


def _graph() -> WorkflowGraph:
    return WorkflowGraph(
        nodes=[
            GraphNode(id="source", code=SOURCE_CODE),
            GraphNode(id="middle", code=MIDDLE_CODE),
        ],
        edges=[
            GraphEdge(
                source_node_id="source",
                source_handle="result",
                target_node_id="middle",
                target_handle="x",
            )
        ],
    )


def _run_exported_script(content: str, path: Path) -> subprocess.CompletedProcess[str]:
    path.write_text(content, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_python_export_is_runnable_and_deduplicates_dependencies(
    tmp_path: Path,
) -> None:
    exported = export_python(
        _graph(), generated_at=GENERATED_AT, name="Iris Demo"
    )

    assert exported.filename == "iris-demo-20261020-120000.py"
    assert exported.media_type == "text/x-python"
    assert "# /// script" in exported.content
    assert '# requires-python = ">=3.12"' in exported.content
    assert exported.content.count('"shared>=1"') == 1
    assert exported.content.index('"alpha>=2"') < exported.content.index(
        '"shared>=1"'
    )
    assert "    import math" in exported.content
    assert "def report_progress" in exported.content
    assert "def raise_if_cancelled" in exported.content
    assert "outputs_root = Path(" in exported.content
    assert "node_outputs['source']['result']" in exported.content
    assert "offset=1" in exported.content
    assert exported.content.index("def __node_exec_source") < exported.content.index(
        "def __node_exec_middle"
    )

    completed = _run_exported_script(
        exported.content, tmp_path / "workflow.py"
    )

    assert completed.returncode == 0, completed.stderr
    assert "computed 3" in completed.stdout
    assert "ComplexUI workflow completed: 2 node(s)." in completed.stdout


def test_python_export_includes_display_helper(tmp_path: Path) -> None:
    graph = WorkflowGraph(
        nodes=[
            GraphNode(
                id="display",
                code=(
                    "value = 'visible'\n"
                    "display(value)\n"
                    "return value: STRING\n"
                ),
            )
        ]
    )

    exported = export_python(graph, generated_at=GENERATED_AT)
    assert "def display(value: object) -> None:" in exported.content
    assert 'globals().get("get_ipython")' in exported.content

    completed = _run_exported_script(
        exported.content, tmp_path / "display.py"
    )
    assert completed.returncode == 0, completed.stderr
    assert "visible" in completed.stdout

    notebook = json.loads(
        export_notebook(graph, generated_at=GENERATED_AT).content
    )
    setup = "".join(notebook["cells"][1]["source"])
    assert "def display(value: object) -> None:" in setup


def test_export_keeps_future_imports_before_runtime_helpers() -> None:
    graph = WorkflowGraph(
        nodes=[
            GraphNode(
                id="future",
                code=(
                    "from __future__ import annotations\n"
                    "value = 1\n"
                    "return value: ANY\n"
                ),
            )
        ]
    )

    content = export_python(graph, generated_at=GENERATED_AT).content
    assert content.index("from __future__ import annotations") < content.index(
        "def report_progress"
    )

    notebook = json.loads(
        export_notebook(graph, generated_at=GENERATED_AT).content
    )
    setup = "".join(notebook["cells"][1]["source"])
    assert setup.index("from __future__ import annotations") < setup.index(
        "def report_progress"
    )


def test_python_export_isolates_import_aliases(tmp_path: Path) -> None:
    graph = WorkflowGraph(
        nodes=[
            GraphNode(
                id="math-node",
                code=(
                    "import math as helper\n"
                    "value = helper.ceil(1.2)\n"
                    "return value: ANY\n"
                ),
            ),
            GraphNode(
                id="stats-node",
                code=(
                    "import statistics as helper\n"
                    "x: ANY\n"
                    "value = helper.mean([x, 3])\n"
                    "print('mean', value)\n"
                    "return value: ANY\n"
                ),
            ),
        ],
        edges=[
            GraphEdge(
                source_node_id="math-node",
                source_handle="value",
                target_node_id="stats-node",
                target_handle="x",
            )
        ],
    )

    exported = export_python(graph, generated_at=GENERATED_AT)
    completed = _run_exported_script(
        exported.content, tmp_path / "aliases.py"
    )

    assert completed.returncode == 0, completed.stderr
    assert "mean 2.5" in completed.stdout


def test_notebook_export_has_valid_cells_and_node_execution() -> None:
    exported = export_notebook(_graph(), generated_at=GENERATED_AT)

    assert exported.filename == "workflow-20261020-120000.ipynb"
    assert exported.media_type == "application/x-ipynb+json"
    notebook = json.loads(exported.content)
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"

    cells = notebook["cells"]
    assert [cell["cell_type"] for cell in cells] == [
        "markdown",
        "code",
        "code",
        "code",
        "code",
    ]
    assert len({cell["id"] for cell in cells}) == len(cells)
    intro = "".join(cells[0]["source"])
    assert "`alpha>=2`" in intro
    assert "`shared>=1`" in intro
    assert "`zeta>=1`" in intro
    setup = "".join(cells[1]["source"])
    assert "    import math" in setup
    assert "def __node_exec_source" in setup
    assert "node_outputs = {}" in setup
    assert "def report_progress" in setup
    assert "def raise_if_cancelled" in setup
    assert "outputs_root = Path(" in setup
    assert "node_outputs['source']['result']" in "".join(cells[3]["source"])
    assert "node_outputs" in "".join(cells[-1]["source"])


def test_export_scope_includes_only_target_ancestors() -> None:
    graph = WorkflowGraph(
        nodes=[
            GraphNode(
                id="first",
                code="value = 1\nreturn value: ANY\n",
            ),
            GraphNode(
                id="second",
                code=(
                    "x: ANY\n"
                    "value = x + 1\n"
                    "return value: ANY\n"
                ),
            ),
            GraphNode(
                id="third",
                code=(
                    "x: ANY\n"
                    "value = x + 1\n"
                    "return value: ANY\n"
                ),
            ),
        ],
        edges=[
            GraphEdge(
                source_node_id="first",
                source_handle="value",
                target_node_id="second",
                target_handle="x",
            ),
            GraphEdge(
                source_node_id="second",
                source_handle="value",
                target_node_id="third",
                target_handle="x",
            ),
        ],
    )

    exported = export_python(
        graph, targets=["second"], generated_at=GENERATED_AT
    )

    assert "__node_exec_first" in exported.content
    assert "__node_exec_second" in exported.content
    assert "__node_exec_third" not in exported.content


def test_empty_graph_cannot_be_exported() -> None:
    with pytest.raises(ExportError, match="no nodes"):
        export_python(WorkflowGraph())


def test_python_export_resolves_outputfile_widgets(tmp_path: Path) -> None:
    graph = WorkflowGraph(
        nodes=[
            GraphNode(
                id="write",
                code=(
                    "from pathlib import Path\n"
                    'target: OutputFile = ""\n'
                    "Path(target).write_text('ok', encoding='utf-8')\n"
                    "result = str(target)\n"
                    "return result: STRING\n"
                ),
                widget_values={"target": "2026-10-23/exported.txt"},
            )
        ]
    )

    exported = export_python(graph, generated_at=GENERATED_AT)
    target = get_output_store().outputs_dir / "2026-10-23" / "exported.txt"
    assert repr(str(target)) in exported.content

    completed = _run_exported_script(exported.content, tmp_path / "export.py")
    assert completed.returncode == 0, completed.stderr
    assert target.read_text(encoding="utf-8") == "ok"


def test_exported_save_media_uses_outputs_root(tmp_path: Path) -> None:
    registry = SecurityRegistry()
    graph = WorkflowGraph(
        nodes=[
            GraphNode(
                id="source",
                code=registry.get_template("make_demo_image").code,
                widget_values={"width": 64, "height": 64},
            ),
            GraphNode(
                id="save",
                code=registry.get_template("save_media").code,
                widget_values={"model_name": "exported"},
            ),
        ],
        edges=[
            GraphEdge(
                source_node_id="source",
                source_handle="image",
                target_node_id="save",
                target_handle="media",
            )
        ],
    )

    exported = export_python(graph, generated_at=GENERATED_AT)
    completed = _run_exported_script(exported.content, tmp_path / "save.py")

    assert completed.returncode == 0, completed.stderr
    saved = list(get_output_store().outputs_dir.glob("*/exported_001.png"))
    assert len(saved) == 1


def test_exported_workflow_broadcasts_image_batches(tmp_path: Path) -> None:
    source = GraphNode(
        id="source",
        code=(
            "from PIL import Image\n"
            "images = [Image.new('RGB', (4, 4)) for _ in range(3)]\n"
            "return images: IMAGE_BATCH\n"
        ),
    )
    transform = GraphNode(
        id="transform",
        code=(
            "image: IMAGE\n"
            "result = image.resize((image.width + 1, image.height + 1))\n"
            "return result: IMAGE\n"
        ),
    )
    graph = WorkflowGraph(
        nodes=[source, transform],
        edges=[
            GraphEdge(
                source_node_id="source",
                source_handle="images",
                target_node_id="transform",
                target_handle="image",
            )
        ],
    )

    exported = export_python(graph, generated_at=GENERATED_AT)
    assert "_run_batch_node(" in exported.content

    completed = _run_exported_script(
        exported.content, tmp_path / "batch.py"
    )
    assert completed.returncode == 0, completed.stderr


def test_export_passes_none_for_unconnected_optional_inputs(tmp_path: Path) -> None:
    code = (
        "image1: Optional[IMAGE] = None\n"
        "return image1: ANY\n"
    )
    graph = WorkflowGraph(
        nodes=[GraphNode(id="optional_export", code=code)],
        edges=[],
    )

    exported = export_python(graph, generated_at=GENERATED_AT)

    assert "def __node_exec_optional_export(image1=None):" in exported.content
    assert "image1=None" in exported.content
    completed = _run_exported_script(
        exported.content, tmp_path / "optional.py"
    )
    assert completed.returncode == 0, completed.stderr
