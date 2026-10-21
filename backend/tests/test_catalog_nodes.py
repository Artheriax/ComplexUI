"""Integration tests for the data loader, cleaning and EDA template families."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from openpyxl import Workbook

from app.executor import (
    NodeExecutionError,
    clear_cache,
    execute_graph,
)
from app.schemas import GraphEdge, GraphNode, WorkflowGraph
from app.security import get_security_registry


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _templates() -> dict[str, str]:
    return {record.name: record.code for record in get_security_registry().templates()}


def _edge(source: str, handle: str, target: str, target_handle: str = "df") -> GraphEdge:
    return GraphEdge(
        source_node_id=source,
        source_handle=handle,
        target_node_id=target,
        target_handle=target_handle,
    )


def _run(nodes: list[GraphNode], edges: list[GraphEdge]):
    return execute_graph(WorkflowGraph(nodes=nodes, edges=edges))


def _single(code: str, widgets: dict[str, object] | None = None):
    return _run(
        [GraphNode(id="node", code=code, widget_values=widgets or {})],
        [],
    )


@pytest.mark.parametrize(
    ("name", "rows", "cols"),
    [
        ("load_iris", 150, 5),
        ("load_wine", 178, 14),
        ("load_breast_cancer", 569, 31),
        ("load_diabetes", 442, 11),
        ("load_digits", 1797, 65),
    ],
)
def test_builtin_sklearn_loaders_produce_dataframes(name: str, rows: int, cols: int) -> None:
    result = _single(_templates()[name])
    preview = result.previews["node"]["df"]
    assert preview["rows"] == rows
    assert preview["cols"] == cols
    assert all(row.get("target") is not None for row in preview["head"])


def test_make_classification_dataset_respects_widgets() -> None:
    result = _single(
        _templates()["make_classification_dataset"],
        {"n_samples": 120, "n_features": 6, "n_informative": 4, "n_classes": 2},
    )
    preview = result.previews["node"]["df"]
    assert preview["rows"] == 120
    assert preview["cols"] == 7


def test_messy_cleaning_pipeline_reduces_nulls_and_duplicates() -> None:
    templates = _templates()
    result = _run(
        [
            GraphNode(
                id="raw",
                code=templates["make_messy_dataset"],
                widget_values={"n_rows": 150, "seed": 7},
            ),
            GraphNode(
                id="clean",
                code=templates["drop_missing"],
                widget_values={"columns": "age,score", "how": "any"},
            ),
            GraphNode(
                id="dedupe",
                code=templates["drop_duplicates"],
                widget_values={"subset": "", "keep": "first"},
            ),
            GraphNode(id="report", code=templates["describe_dataframe"]),
        ],
        [
            _edge("raw", "df", "clean"),
            _edge("clean", "cleaned", "dedupe"),
            _edge("dedupe", "deduped", "report"),
        ],
    )

    assert result.previews["raw"]["df"]["rows"] == 155
    cleaned_rows = result.previews["clean"]["cleaned"]["rows"]
    assert cleaned_rows <= 155
    assert result.previews["dedupe"]["deduped"]["rows"] == cleaned_rows - 5
    summary = result.outputs_by_name["report"]["summary"]
    assert summary["rows"] == cleaned_rows - 5
    assert summary["missing"]["age"] == 0
    assert summary["missing"]["score"] == 0
    assert "statistic" in summary["numeric_summary"][0]
    assert any(row["statistic"] == "mean" for row in summary["numeric_summary"])


def test_fill_missing_and_trim_strings_chain() -> None:
    templates = _templates()
    result = _run(
        [
            GraphNode(
                id="raw",
                code=templates["make_messy_dataset"],
                widget_values={"n_rows": 80, "seed": 3},
            ),
            GraphNode(
                id="trim",
                code=templates["trim_strings"],
                widget_values={
                    "columns": "name,city",
                    "collapse_whitespace": True,
                    "lowercase": True,
                },
            ),
            GraphNode(
                id="fill",
                code=templates["fill_missing"],
                widget_values={"columns": "age,score", "strategy": "mean"},
            ),
            GraphNode(id="report", code=templates["missing_values_report"]),
        ],
        [
            _edge("raw", "df", "trim"),
            _edge("trim", "trimmed", "fill"),
            _edge("fill", "filled", "report"),
        ],
    )

    trimmed_head = result.previews["trim"]["trimmed"]["head"]
    for row in trimmed_head:
        if row["name"] is not None:
            assert row["name"] == row["name"].strip().lower()
    report = result.outputs_by_name["report"]["report"]
    assert report["columns"]["age"]["null_count"] == 0
    assert report["columns"]["score"]["null_count"] == 0


def test_clip_outliers_winsorises_numeric_columns() -> None:
    templates = _templates()
    result = _run(
        [
            GraphNode(
                id="raw",
                code=templates["make_messy_dataset"],
                widget_values={"n_rows": 200, "seed": 11},
            ),
            GraphNode(
                id="clip",
                code=templates["clip_outliers"],
                widget_values={
                    "columns": "score",
                    "lower_quantile": 0.05,
                    "upper_quantile": 0.95,
                },
            ),
            GraphNode(
                id="counts",
                code=templates["value_counts"],
                widget_values={"column": "id", "top_k": 5},
            ),
        ],
        [_edge("raw", "df", "clip"), _edge("clip", "clipped", "counts")],
    )

    clipped = result.previews["clip"]["clipped"]
    assert all(row["score"] < 200 for row in clipped["head"])
    counts = result.outputs_by_name["counts"]["counts"]
    assert counts["total"] == 205
    assert len(counts["rows"]) == 5
    assert counts["rows"][0]["count"] == 2


def test_filter_rows_with_contains_and_numeric_operator() -> None:
    templates = _templates()
    result = _run(
        [
            GraphNode(id="wine", code=templates["load_wine"]),
            GraphNode(
                id="filter",
                code=templates["filter_rows"],
                widget_values={"column": "target", "operator": "==", "value": "1"},
            ),
            GraphNode(
                id="balance",
                code=templates["class_balance"],
                widget_values={"target_column": "target"},
            ),
        ],
        [_edge("wine", "df", "filter"), _edge("filter", "filtered", "balance")],
    )

    balance = result.outputs_by_name["balance"]["balance"]
    assert balance["rows"] == 71
    assert balance["n_classes"] == 1
    assert result.previews["filter"]["filtered"]["rows"] == 71


def test_correlation_matrix_is_json_safe() -> None:
    templates = _templates()
    result = _run(
        [
            GraphNode(id="iris", code=templates["load_iris"]),
            GraphNode(
                id="corr",
                code=templates["correlation_matrix"],
                widget_values={"method": "pearson"},
            ),
        ],
        [_edge("iris", "df", "corr")],
    )

    matrix = result.outputs_by_name["corr"]["correlation"]
    assert matrix["columns"] == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "target",
    ]
    size = len(matrix["columns"])
    assert len(matrix["matrix"]) == size
    assert all(len(row) == size for row in matrix["matrix"])
    assert matrix["matrix"][0][0] == pytest.approx(1.0)


def test_drop_columns_without_selection_raises() -> None:
    templates = _templates()
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="iris", code=templates["load_iris"]),
            GraphNode(
                id="drop",
                code=templates["drop_columns"],
                widget_values={"columns": ""},
            ),
        ],
        edges=[_edge("iris", "df", "drop")],
    )

    with pytest.raises(NodeExecutionError, match="at least one column"):
        execute_graph(graph)


def test_file_loaders_read_csv_parquet_excel_and_ndjson(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPLEXUI_WORKSPACE", str(tmp_path))
    templates = _templates()
    table = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    csv_path = tmp_path / "data.csv"
    table.write_csv(csv_path)
    parquet_path = tmp_path / "data.parquet"
    table.write_parquet(parquet_path)
    ndjson_path = tmp_path / "data.ndjson"
    table.write_ndjson(ndjson_path)

    excel_path = tmp_path / "data.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["a", "b"])
    sheet.append([1, "x"])
    sheet.append([2, "y"])
    workbook.save(excel_path)

    cases: list[tuple[str, dict[str, object]]] = [
        ("load_csv", {"path": str(csv_path), "separator": ",", "has_header": True}),
        ("load_parquet", {"path": str(parquet_path)}),
        ("load_excel", {"path": str(excel_path), "sheet_id": 1}),
        ("load_json", {"path": str(ndjson_path)}),
    ]
    for name, widgets in cases:
        result = _single(templates[name], widgets)
        preview = result.previews["node"]["df"]
        assert preview["rows"] >= 2, name
        assert preview["cols"] == 2, name
