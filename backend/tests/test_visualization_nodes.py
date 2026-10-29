"""Execution tests for the visualization node templates."""

from __future__ import annotations

import math

import pytest

from app.executor import NodeExecutionError, clear_cache, execute_graph
from app.schemas import GraphEdge, GraphNode, WorkflowGraph
from app.security import get_security_registry
from app.transpiler import transpile_node


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _templates() -> dict[str, str]:
    return {record.name: record.code for record in get_security_registry().templates()}


def _node(name: str, template: str, **widgets: object) -> GraphNode:
    return GraphNode(id=name, code=_templates()[template], widget_values=widgets)


def _edge(source: str, handle: str, target: str, target_handle: str = "df") -> GraphEdge:
    return GraphEdge(
        source_node_id=source,
        source_handle=handle,
        target_node_id=target,
        target_handle=target_handle,
    )


def _run(nodes: list[GraphNode], edges: list[GraphEdge]):
    return execute_graph(WorkflowGraph(nodes=nodes, edges=edges))


def _plotly(result, node_id: str) -> dict:
    outputs = result.rich_outputs[node_id]
    assert [output.kind for output in outputs] == ["plotly"]
    return outputs[0].figure


def test_histogram_renders_grouped_distribution() -> None:
    result = _run(
        [
            _node("load", "load_iris"),
            _node(
                "hist",
                "histogram",
                column="sepal length (cm)",
                bins=12,
                group_by="target",
                max_groups=3,
            ),
        ],
        [_edge("load", "df", "hist")],
    )

    figure = _plotly(result, "hist")
    assert [trace["type"] for trace in figure["data"]] == ["histogram"] * 3
    assert figure["layout"]["barmode"] == "overlay"
    summary = result.outputs_by_name["hist"]["summary"]
    assert summary["column"] == "sepal length (cm)"
    assert summary["bins"] == 12
    assert summary["groups"] == ["0", "1", "2"]
    assert summary["non_null"] == 150


def test_scatter_plot_samples_points_and_encodes_color_and_size() -> None:
    result = _run(
        [
            _node(
                "data",
                "make_classification_dataset",
                n_samples=120,
                n_features=6,
                n_informative=3,
                n_classes=2,
            ),
            _node(
                "scatter",
                "scatter_plot",
                x="feature_0",
                y="feature_1",
                color_by="target",
                size_by="feature_2",
                max_points=100,
            ),
        ],
        [_edge("data", "df", "scatter")],
    )

    figure = _plotly(result, "scatter")
    trace = figure["data"][0]
    assert trace["type"] == "scatter"
    assert trace["mode"] == "markers"
    summary = result.outputs_by_name["scatter"]["summary"]
    assert summary["points"] == 120
    assert summary["sampled"] is True
    assert 0 < summary["plotted"] <= 100
    assert summary["correlation"] is not None
    assert len(trace["marker"]["color"]) == summary["plotted"]
    assert len(trace["marker"]["size"]) == summary["plotted"]
    assert trace["marker"]["showscale"] is True


def test_scatter_plot_encodes_categorical_colors() -> None:
    result = _run(
        [
            _node("data", "make_messy_dataset", n_rows=60),
            _node(
                "scatter",
                "scatter_plot",
                x="age",
                y="score",
                color_by="city",
                max_points=100,
            ),
        ],
        [_edge("data", "df", "scatter")],
    )

    figure = _plotly(result, "scatter")
    marker = figure["data"][0]["marker"]
    assert marker["showscale"] is True
    assert marker["color"]
    assert all(isinstance(value, int) for value in marker["color"])
    assert marker["colorbar"]["tickvals"][0] == -1
    assert marker["colorbar"]["ticktext"][0] == "missing"
    assert all(isinstance(label, str) for label in marker["colorbar"]["ticktext"])
    assert result.outputs_by_name["scatter"]["summary"]["color_by"] == "city"


def test_box_plot_limits_groups_and_reports_quartiles() -> None:
    result = _run(
        [
            _node("load", "load_iris"),
            _node(
                "box",
                "box_plot",
                value_column="petal width (cm)",
                group_by="target",
                points="all",
            ),
        ],
        [_edge("load", "df", "box")],
    )

    figure = _plotly(result, "box")
    assert [trace["type"] for trace in figure["data"]] == ["box"] * 3
    assert all(trace["boxpoints"] == "all" for trace in figure["data"])
    summary = result.outputs_by_name["box"]["summary"]
    assert summary["groups_total"] == 3
    assert summary["groups_shown"] == 3
    assert summary["min"] == pytest.approx(0.1)
    assert summary["max"] == pytest.approx(2.5)


def test_bar_chart_counts_categories() -> None:
    result = _run(
        [
            _node("load", "load_iris"),
            _node("bar", "bar_chart", category="target", horizontal=True),
        ],
        [_edge("load", "df", "bar")],
    )

    figure = _plotly(result, "bar")
    trace = figure["data"][0]
    assert trace["type"] == "bar"
    assert trace["orientation"] == "h"
    assert trace["x"] == [50.0, 50.0, 50.0]
    assert figure["layout"]["yaxis"]["autorange"] == "reversed"
    summary = result.outputs_by_name["bar"]["summary"]
    assert summary["aggregate"] == "count"
    assert summary["groups_total"] == 3
    assert summary["max_value"] == pytest.approx(50.0)


def test_bar_chart_aggregates_numeric_values() -> None:
    result = _run(
        [
            _node("data", "make_messy_dataset", n_rows=80),
            _node(
                "bar",
                "bar_chart",
                category="city",
                value="score",
                aggregate="mean",
                top_k=2,
            ),
        ],
        [_edge("data", "df", "bar")],
    )

    summary = result.outputs_by_name["bar"]["summary"]
    assert summary["aggregate"] == "mean"
    assert summary["value"] == "score"
    assert summary["groups_shown"] == 2
    assert summary["rows"] <= 80
    assert _plotly(result, "bar")["data"][0]["type"] == "bar"


def test_correlation_heatmap_supports_spearman_and_column_caps() -> None:
    result = _run(
        [
            _node("load", "load_iris"),
            _node(
                "subset",
                "correlation_heatmap",
                method="spearman",
                columns="sepal length (cm), petal length (cm)",
            ),
            _node(
                "capped",
                "correlation_heatmap",
                method="pearson",
                max_columns=2,
            ),
        ],
        [
            _edge("load", "df", "subset"),
            _edge("load", "df", "capped"),
        ],
    )

    subset = _plotly(result, "subset")
    assert subset["data"][0]["type"] == "heatmap"
    assert subset["data"][0]["texttemplate"] == "%{text}"
    subset_summary = result.outputs_by_name["subset"]["summary"]
    assert subset_summary["n_columns"] == 2
    assert subset_summary["truncated"] is False
    assert subset_summary["strongest"]["correlation"] > 0.8
    diagonal = subset["data"][0]["z"][0][0]
    assert diagonal == pytest.approx(1.0)

    capped_summary = result.outputs_by_name["capped"]["summary"]
    assert capped_summary["n_columns"] == 2
    assert capped_summary["truncated"] is True


def test_confusion_matrix_and_feature_importance_render_model_views() -> None:
    result = _run(
        [
            _node("load", "load_iris"),
            _node("split", "train_test_split", test_size=0.2, stratify=True),
            _node("model", "random_forest_classifier", n_estimators=40),
            _node("cm", "confusion_matrix", normalize="true"),
            _node("fi", "feature_importance", top_k=3),
        ],
        [
            _edge("load", "df", "split"),
            _edge("split", "train", "model", "train"),
            _edge("model", "model", "cm", "model"),
            _edge("split", "test", "cm", "test"),
            _edge("model", "model", "fi", "model"),
        ],
    )

    figure = _plotly(result, "cm")
    assert figure["data"][0]["type"] == "heatmap"
    summary = result.outputs_by_name["cm"]["summary"]
    assert summary["n_classes"] == 3
    assert summary["normalize"] == "true"
    assert summary["accuracy"] > 0.8
    assert all(
        math.isclose(sum(row), 1.0, abs_tol=1e-9) for row in summary["confusion_matrix"]
    )

    importance = result.outputs_by_name["fi"]["summary"]
    assert importance["model_type"] == "RandomForestClassifier"
    assert importance["importance"] == "feature_importances_"
    assert importance["n_features"] == 4
    assert importance["features_shown"] == 3
    trace = _plotly(result, "fi")["data"][0]
    assert trace["type"] == "bar"
    assert trace["orientation"] == "h"


def test_feature_importance_uses_linear_coefficients() -> None:
    names = "age,sex,bmi,bp,s1,s2,s3,s4,s5,s6"
    result = _run(
        [
            _node("load", "load_diabetes"),
            _node("split", "train_test_split", test_size=0.25, random_state=11),
            _node("model", "ridge_regression", alpha=0.5),
            _node("fi", "feature_importance", feature_names=names, top_k=4),
            _node("diag", "regression_diagnostics"),
        ],
        [
            _edge("load", "df", "split"),
            _edge("split", "train", "model", "train"),
            _edge("model", "model", "fi", "model"),
            _edge("model", "model", "diag", "model"),
            _edge("split", "test", "diag", "test"),
        ],
    )

    importance = result.outputs_by_name["fi"]["summary"]
    assert importance["model_type"] == "Ridge"
    assert importance["importance"] == "coefficient"
    assert importance["n_features"] == 10
    assert importance["features_shown"] == 4
    assert {item["feature"] for item in importance["top_features"]} <= set(
        names.split(",")
    )

    diagnostics = result.outputs_by_name["diag"]["summary"]
    assert diagnostics["r2"] > 0.2
    assert diagnostics["rmse"] > 0
    assert diagnostics["n_test_samples"] == 111
    figure = _plotly(result, "diag")
    assert len(figure["data"]) == 4
    assert figure["data"][1]["line"]["dash"] == "dash"
    assert figure["data"][3]["line"]["dash"] == "dash"


def test_training_history_smooths_loss_curve() -> None:
    pytest.importorskip("torch")
    result = _run(
        [
            _node("data", "tensor_dataset", samples=120),
            _node("train", "mlp_train", epochs=20, hidden=8),
            _node(
                "history",
                "training_history",
                smooth_window=5,
                show_raw=True,
                max_points=100,
            ),
        ],
        [
            _edge("data", "data", "train", "data"),
            _edge("train", "history", "history", "history"),
        ],
    )

    figure = _plotly(result, "history")
    assert [trace["type"] for trace in figure["data"]] == ["scatter", "scatter"]
    assert figure["data"][0]["opacity"] == pytest.approx(0.45)
    summary = result.outputs_by_name["history"]["summary"]
    assert summary["epochs"] == 20
    assert summary["smooth_window"] == 5
    assert summary["plotted_points"] == 20
    assert summary["lower_is_better"] is True
    assert 1 <= summary["best_epoch"] <= 20
    assert summary["best_value"] == summary["min"]
    assert summary["min"] <= summary["final"] <= summary["max"]


def test_training_history_supports_higher_is_better_metrics() -> None:
    pytest.importorskip("torch")
    result = _run(
        [
            _node("data", "tensor_dataset", samples=80),
            _node("train", "mlp_train", epochs=8, hidden=8),
            _node(
                "history",
                "training_history",
                metric_name="accuracy",
                show_raw=False,
                lower_is_better=False,
            ),
        ],
        [
            _edge("data", "data", "train", "data"),
            _edge("train", "history", "history", "history"),
        ],
    )

    summary = result.outputs_by_name["history"]["summary"]
    assert summary["lower_is_better"] is False
    assert summary["best_value"] == summary["max"]
    assert summary["metric"] == "accuracy"
    figure = _plotly(result, "history")
    assert len(figure["data"]) == 1
    assert figure["data"][0]["name"] == "accuracy"


def test_training_history_rejects_non_positive_log_scale() -> None:
    function = transpile_node(_templates()["training_history"], "history").func
    function.__globals__["display"] = lambda value: None
    with pytest.raises(ValueError, match="strictly positive"):
        function(history=[1.0, 0.5, 0.0], metric_name="loss", log_y=True)


def test_visualization_nodes_reject_invalid_widgets() -> None:
    with pytest.raises(NodeExecutionError, match="Unknown value column"):
        _run(
            [
                _node("load", "load_iris"),
                _node("box", "box_plot", value_column="missing"),
            ],
            [_edge("load", "df", "box")],
        )

    with pytest.raises(NodeExecutionError, match="must be numeric"):
        _run(
            [
                _node("data", "make_messy_dataset"),
                _node("scatter", "scatter_plot", x="city", y="age"),
            ],
            [_edge("data", "df", "scatter")],
        )

    with pytest.raises(NodeExecutionError, match="Unknown value column"):
        _run(
            [
                _node("load", "load_iris"),
                _node("bar", "bar_chart", category="target", aggregate="sum"),
            ],
            [_edge("load", "df", "bar")],
        )

    with pytest.raises(NodeExecutionError, match="Unknown columns"):
        _run(
            [
                _node("load", "load_iris"),
                _node("corr", "correlation_heatmap", columns="missing"),
            ],
            [_edge("load", "df", "corr")],
        )

    with pytest.raises(NodeExecutionError, match="increase max_classes"):
        _run(
            [
                _node("load", "load_iris"),
                _node("split", "train_test_split"),
                _node("model", "knn_train"),
                _node("cm", "confusion_matrix", max_classes=2),
            ],
            [
                _edge("load", "df", "split"),
                _edge("split", "train", "model", "train"),
                _edge("model", "model", "cm", "model"),
                _edge("split", "test", "cm", "test"),
            ],
        )

    with pytest.raises(NodeExecutionError, match="neither"):
        _run(
            [
                _node("load", "load_iris"),
                _node("split", "train_test_split"),
                _node("model", "kmeans_clustering"),
                _node("fi", "feature_importance"),
            ],
            [
                _edge("load", "df", "split"),
                _edge("split", "train", "model", "train"),
                _edge("model", "model", "fi", "model"),
            ],
        )


