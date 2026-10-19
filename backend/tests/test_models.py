"""Integration tests for preprocessing, imbalanced-data, model and evaluation nodes."""

from __future__ import annotations

import numpy as np
import pytest

from app.executor import clear_cache, execute_graph
from app.schemas import GraphEdge, GraphNode, WorkflowGraph
from app.security import get_security_registry


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _templates() -> dict[str, str]:
    return {record.name: record.code for record in get_security_registry().templates()}


def _edge(source: str, handle: str, target: str, target_handle: str) -> GraphEdge:
    return GraphEdge(
        source_node_id=source,
        source_handle=handle,
        target_node_id=target,
        target_handle=target_handle,
    )


def _model_chain(
    model_name: str,
    *,
    loader: str = "load_iris",
    evaluate: str = "evaluate_metrics",
    model_widgets: dict[str, object] | None = None,
):
    templates = _templates()
    nodes = [
        GraphNode(id="load", code=templates[loader]),
        GraphNode(id="split", code=templates["train_test_split"]),
        GraphNode(
            id="model",
            code=templates[model_name],
            widget_values=model_widgets or {},
        ),
        GraphNode(id="evaluate", code=templates[evaluate]),
    ]
    edges = [
        _edge("load", "df", "split", "df"),
        _edge("split", "train", "model", "train"),
        _edge("model", "model", "evaluate", "model"),
        _edge("split", "test", "evaluate", "test"),
    ]
    return execute_graph(WorkflowGraph(nodes=nodes, edges=edges))


def _summary(result, node_id: str = "report"):
    return result.outputs_by_name[node_id]["summary"]


def test_stratified_split_respects_class_ratios() -> None:
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_breast_cancer"]),
                GraphNode(
                    id="split",
                    code=templates["train_test_split"],
                    widget_values={"stratify": True, "test_size": 0.25},
                ),
            ],
            edges=[_edge("load", "df", "split", "df")],
        )
    )

    train = result.previews["split"]["train"]
    test = result.previews["split"]["test"]
    assert train["X_shape"][0] + test["X_shape"][0] == 569
    assert train["X_shape"][1] == 30
    assert test["X_shape"][0] in (142, 143)
    y_train = result.outputs_by_name["split"]["train"][1]
    minority_fraction = float(np.mean(np.asarray(y_train) == 0))
    assert minority_fraction == pytest.approx(0.37, abs=0.02)


def test_standard_scaler_zeroes_feature_means_and_keeps_target() -> None:
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_iris"]),
                GraphNode(id="scaler", code=templates["standard_scaler"]),
                GraphNode(id="report", code=templates["describe_dataframe"]),
            ],
            edges=[
                _edge("load", "df", "scaler", "df"),
                _edge("scaler", "scaled", "report", "df"),
            ],
        )
    )

    summary = _summary(result)
    means = {row["statistic"]: row for row in summary["numeric_summary"]}["mean"]
    assert means["sepal length (cm)"] == pytest.approx(0.0, abs=1e-9)
    assert means["petal width (cm)"] == pytest.approx(0.0, abs=1e-9)
    assert means["target"] == pytest.approx(1.0, abs=1e-9)


def test_minmax_scaler_bounds_features_to_zero_one() -> None:
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_iris"]),
                GraphNode(id="scaler", code=templates["minmax_scaler"]),
                GraphNode(id="report", code=templates["describe_dataframe"]),
            ],
            edges=[
                _edge("load", "df", "scaler", "df"),
                _edge("scaler", "scaled", "report", "df"),
            ],
        )
    )

    summary = _summary(result)
    stats = {row["statistic"]: row for row in summary["numeric_summary"]}
    assert stats["min"]["sepal width (cm)"] == pytest.approx(0.0, abs=1e-9)
    assert stats["max"]["petal length (cm)"] == pytest.approx(1.0, abs=1e-9)


def test_pca_reduce_keeps_knn_accurate() -> None:
    templates = _templates()
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="load", code=templates["load_iris"]),
            GraphNode(id="split", code=templates["train_test_split"]),
            GraphNode(
                id="pca",
                code=templates["pca_reduce"],
                widget_values={"n_components": 2},
            ),
            GraphNode(id="model", code=templates["knn_train"]),
            GraphNode(
                id="cv",
                code=templates["cross_validate_classifier"],
                widget_values={"cv_folds": 5},
            ),
        ],
        edges=[
            _edge("load", "df", "split", "df"),
            _edge("split", "train", "pca", "train"),
            _edge("pca", "reduced", "model", "train"),
            _edge("model", "model", "cv", "model"),
            _edge("pca", "reduced", "cv", "train"),
        ],
    )
    result = execute_graph(graph)

    assert result.previews["pca"]["reduced"]["X_shape"] == [120, 2]
    metrics = result.outputs_by_name["cv"]["metrics"]
    assert metrics["accuracy_mean"] > 0.90


def test_select_k_best_reduces_feature_count() -> None:
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_iris"]),
                GraphNode(id="split", code=templates["train_test_split"]),
                GraphNode(
                    id="select",
                    code=templates["select_k_best"],
                    widget_values={"k": 2},
                ),
            ],
            edges=[
                _edge("load", "df", "split", "df"),
                _edge("split", "train", "select", "train"),
            ],
        )
    )

    assert result.previews["select"]["reduced"]["X_shape"] == [120, 2]


@pytest.mark.parametrize(
    "name", ["random_oversample", "random_undersample", "smote_resample"]
)
def test_imbalanced_samplers_balance_train_classes(name: str) -> None:
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_breast_cancer"]),
                GraphNode(id="split", code=templates["train_test_split"]),
                GraphNode(id="sampler", code=templates[name]),
            ],
            edges=[
                _edge("load", "df", "split", "df"),
                _edge("split", "train", "sampler", "train"),
            ],
        )
    )

    _, y_resampled = result.outputs_by_name["sampler"]["resampled"]
    counts = np.bincount(np.asarray(y_resampled))
    assert counts[0] == counts[1]
    assert counts[0] > 140


@pytest.mark.parametrize(
    "model_name",
    [
        "knn_train",
        "logistic_regression",
        "decision_tree_classifier",
        "random_forest_classifier",
        "gradient_boosting_classifier",
        "svm_classifier",
        "gaussian_nb_classifier",
    ],
)
def test_classifiers_reach_high_accuracy_on_iris(model_name: str) -> None:
    result = _model_chain(model_name)

    metrics = result.outputs_by_name["evaluate"]["metrics"]
    assert metrics["accuracy"] > 0.85, model_name
    assert metrics["n_test_samples"] == 30


def test_cross_validation_reports_fold_spread() -> None:
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_iris"]),
                GraphNode(id="split", code=templates["train_test_split"]),
                GraphNode(id="model", code=templates["random_forest_classifier"]),
                GraphNode(
                    id="cv",
                    code=templates["cross_validate_classifier"],
                    widget_values={"cv_folds": 5},
                ),
            ],
            edges=[
                _edge("load", "df", "split", "df"),
                _edge("split", "train", "model", "train"),
                _edge("model", "model", "cv", "model"),
                _edge("split", "train", "cv", "train"),
            ],
        )
    )

    metrics = result.outputs_by_name["cv"]["metrics"]
    assert metrics["cv_folds"] == 5
    assert len(metrics["fold_scores"]) == 5
    assert metrics["accuracy_mean"] > 0.85
    assert metrics["accuracy_std"] >= 0.0


@pytest.mark.parametrize(
    "model_name",
    [
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "random_forest_regressor",
        "gradient_boosting_regressor",
    ],
)
def test_regressors_fit_diabetes(model_name: str) -> None:
    result = _model_chain(
        model_name,
        loader="load_diabetes",
        evaluate="evaluate_regression",
    )

    metrics = result.outputs_by_name["evaluate"]["metrics"]
    assert metrics["r2"] > 0.10, model_name
    assert metrics["rmse"] > 0.0
    assert metrics["n_test_samples"] == 89


def test_decision_tree_regressor_executes() -> None:
    result = _model_chain(
        "decision_tree_regressor",
        loader="load_diabetes",
        evaluate="evaluate_regression",
    )

    metrics = result.outputs_by_name["evaluate"]["metrics"]
    assert {"mae", "mse", "rmse", "r2"} <= set(metrics)


def test_kmeans_clustering_on_pca_components() -> None:
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_iris"]),
                GraphNode(
                    id="dataset",
                    code=templates["dataframe_to_dataset"],
                ),
                GraphNode(
                    id="pca",
                    code=templates["pca_reduce"],
                    widget_values={"n_components": 2},
                ),
                GraphNode(
                    id="kmeans",
                    code=templates["kmeans_clustering"],
                    widget_values={"n_clusters": 3},
                ),
            ],
            edges=[
                _edge("load", "df", "dataset", "df"),
                _edge("dataset", "dataset", "pca", "train"),
                _edge("pca", "reduced", "kmeans", "train"),
            ],
        )
    )

    preview = result.previews["kmeans"]["model"]
    assert preview["class"] == "KMeans"
    assert preview["params"]["n_clusters"] == 3


def test_drop_target_and_one_hot_encode() -> None:
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_iris"]),
                GraphNode(id="drop", code=templates["drop_target"]),
                GraphNode(id="report", code=templates["describe_dataframe"]),
            ],
            edges=[
                _edge("load", "df", "drop", "df"),
                _edge("drop", "features", "report", "df"),
            ],
        )
    )
    assert _summary(result)["columns"] == 4
    assert "target" not in _summary(result)["column_names"]

    encoded = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(
                    id="raw",
                    code=templates["make_messy_dataset"],
                    widget_values={"n_rows": 60, "seed": 5},
                ),
                GraphNode(
                    id="trim",
                    code=templates["trim_strings"],
                    widget_values={"columns": "city", "lowercase": True},
                ),
                GraphNode(
                    id="encode",
                    code=templates["one_hot_encode"],
                    widget_values={"columns": "city", "drop_first": True},
                ),
                GraphNode(id="report", code=templates["describe_dataframe"]),
            ],
            edges=[
                _edge("raw", "df", "trim", "df"),
                _edge("trim", "trimmed", "encode", "df"),
                _edge("encode", "encoded", "report", "df"),
            ],
        )
    )
    names = _summary(encoded)["column_names"]
    assert "city_paris" in names or "city_lyon" in names
    assert "city" not in names


def test_xgboost_classifier_and_regressor() -> None:
    pytest.importorskip("xgboost")
    classifier = _model_chain("xgb_classifier")
    metrics = classifier.outputs_by_name["evaluate"]["metrics"]
    assert metrics["accuracy"] > 0.85

    regressor = _model_chain(
        "xgb_regressor",
        loader="load_diabetes",
        evaluate="evaluate_regression",
    )
    assert regressor.outputs_by_name["evaluate"]["metrics"]["r2"] > 0.10


def test_lightgbm_classifier_and_regressor() -> None:
    pytest.importorskip("lightgbm")
    classifier = _model_chain("lgbm_classifier")
    metrics = classifier.outputs_by_name["evaluate"]["metrics"]
    assert metrics["accuracy"] > 0.85

    regressor = _model_chain(
        "lgbm_regressor",
        loader="load_diabetes",
        evaluate="evaluate_regression",
    )
    assert regressor.outputs_by_name["evaluate"]["metrics"]["r2"] > 0.10


def test_dataframe_to_tensors_feeds_torch_mlp() -> None:
    pytest.importorskip("torch")
    templates = _templates()
    result = execute_graph(
        WorkflowGraph(
            nodes=[
                GraphNode(id="load", code=templates["load_iris"]),
                GraphNode(id="tensors", code=templates["dataframe_to_tensors"]),
                GraphNode(
                    id="mlp",
                    code=templates["mlp_train"],
                    widget_values={"hidden": 16, "epochs": 40, "lr": 0.01},
                ),
                GraphNode(id="evaluate", code=templates["evaluate_torch"]),
            ],
            edges=[
                _edge("load", "df", "tensors", "df"),
                _edge("tensors", "data", "mlp", "data"),
                _edge("mlp", "model", "evaluate", "model"),
                _edge("tensors", "data", "evaluate", "data"),
            ],
        )
    )

    metrics = result.outputs_by_name["evaluate"]["metrics"]
    assert metrics["accuracy"] > 0.85
    data_preview = result.previews["tensors"]["data"]
    assert data_preview["X_shape"] == [150, 4]
    assert data_preview["y_shape"] == [150]
