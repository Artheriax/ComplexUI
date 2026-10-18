"""Tests for the DAG executor, caching, security and env management (M2)."""

from __future__ import annotations

import json
import subprocess

import pytest

from app.env_manager import (
    DependencyInstallError,
    install_dependencies,
    missing_dependencies,
)
from app.executor import (
    ExecutionCancelled,
    GraphCycleError,
    GraphValidationError,
    NodeExecutionError,
    clear_cache,
    execute_graph,
    topological_sort,
    validate_graph,
)
from app.schemas import (
    EventType,
    ExecutionEvent,
    GraphEdge,
    GraphNode,
    SocketType,
    WorkflowGraph,
)
from app.security import (
    SecurityRegistry,
    UntrustedNodeError,
    code_hash,
    get_security_registry,
)
from app.transpiler import TranspileError, parse_node


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _templates() -> dict[str, str]:
    return {record.name: record.code for record in get_security_registry().templates()}


def _custom_registry(*nodes: GraphNode) -> SecurityRegistry:
    registry = SecurityRegistry()
    for node in nodes:
        registry.approve(node.id, code_hash(node.code))
    return registry


def _chain_graph(n_neighbors: int = 5) -> WorkflowGraph:
    templates = _templates()
    return WorkflowGraph(
        nodes=[
            GraphNode(id="load", code=templates["load_iris"]),
            GraphNode(id="split", code=templates["train_test_split"]),
            GraphNode(
                id="knn",
                code=templates["knn_train"],
                widget_values={"n_neighbors": n_neighbors},
            ),
            GraphNode(id="evaluate", code=templates["evaluate_metrics"]),
        ],
        edges=[
            GraphEdge(
                source_node_id="load",
                source_handle="df",
                target_node_id="split",
                target_handle="df",
            ),
            GraphEdge(
                source_node_id="split",
                source_handle="train",
                target_node_id="knn",
                target_handle="train",
            ),
            GraphEdge(
                source_node_id="knn",
                source_handle="model",
                target_node_id="evaluate",
                target_handle="model",
            ),
            GraphEdge(
                source_node_id="split",
                source_handle="test",
                target_node_id="evaluate",
                target_handle="test",
            ),
        ],
    )


EXPECTED_SOCKETS: dict[str, tuple[list[str], list[str], list[str]]] = {
    "load_iris": ([], ["df"], []),
    "train_test_split": (["df"], ["train", "test"], ["test_size"]),
    "knn_train": (["train"], ["model"], ["n_neighbors"]),
    "evaluate_metrics": (["model", "test"], ["metrics"], []),
    "tensor_dataset": ([], ["data"], ["samples", "seed", "noise", "device"]),
    "mlp_train": (
        ["data"],
        ["model", "history"],
        ["hidden", "epochs", "lr", "seed"],
    ),
    "evaluate_torch": (["model", "data"], ["metrics"], []),
}


def test_chain_executes_and_accuracy_above_90() -> None:
    events: list[ExecutionEvent] = []
    result = execute_graph(_chain_graph(), emit=events.append)

    assert result.cached == {
        "load": False,
        "split": False,
        "knn": False,
        "evaluate": False,
    }
    metrics = result.outputs_by_name["evaluate"]["metrics"]
    assert metrics["accuracy"] > 0.90

    df_preview = result.previews["load"]["df"]
    assert df_preview["rows"] == 150
    assert df_preview["cols"] == 5
    assert len(df_preview["head"]) == 5

    train_preview = result.previews["split"]["train"]
    assert train_preview["X_shape"] == [120, 4]
    assert train_preview["y_shape"] == [120]
    assert train_preview["dtype"] == "float64"

    model_preview = result.previews["knn"]["model"]
    assert model_preview["class"] == "KNeighborsClassifier"
    assert model_preview["params"]["n_neighbors"] == 5

    assert result.previews["evaluate"]["metrics"]["accuracy"] == pytest.approx(
        metrics["accuracy"]
    )

    types = [event.type for event in events]
    assert types.count(EventType.NODE_STARTED) == 4
    assert types.count(EventType.NODE_COMPLETED) == 4
    json.dumps([event.model_dump(mode="json") for event in events])


def test_identical_rerun_is_fully_cached() -> None:
    execute_graph(_chain_graph())

    events: list[ExecutionEvent] = []
    result = execute_graph(_chain_graph(), emit=events.append)

    assert result.cached == {
        "load": True,
        "split": True,
        "knn": True,
        "evaluate": True,
    }
    assert all(event.type is not EventType.NODE_STARTED for event in events)
    completed = [
        event for event in events if event.type is EventType.NODE_COMPLETED
    ]
    assert len(completed) == 4
    assert all(event.data["cached"] is True for event in completed)


def test_modified_widget_reruns_only_downstream_nodes() -> None:
    execute_graph(_chain_graph(5))

    events: list[ExecutionEvent] = []
    result = execute_graph(_chain_graph(7), emit=events.append)

    assert result.cached == {
        "load": True,
        "split": True,
        "knn": False,
        "evaluate": False,
    }
    started = {
        event.node_id
        for event in events
        if event.type is EventType.NODE_STARTED
    }
    assert started == {"knn", "evaluate"}
    assert result.previews["knn"]["model"]["params"]["n_neighbors"] == 7
    assert result.outputs_by_name["evaluate"]["metrics"]["accuracy"] > 0.90


def test_wire_overrides_widget_value() -> None:
    templates = _templates()
    const_node = GraphNode(id="const", code="n = 11\nreturn n: ANY")
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="load", code=templates["load_iris"]),
            GraphNode(id="split", code=templates["train_test_split"]),
            GraphNode(
                id="knn",
                code=templates["knn_train"],
                widget_values={"n_neighbors": 3},
            ),
            const_node,
        ],
        edges=[
            GraphEdge(
                source_node_id="load",
                source_handle="df",
                target_node_id="split",
                target_handle="df",
            ),
            GraphEdge(
                source_node_id="split",
                source_handle="train",
                target_node_id="knn",
                target_handle="train",
            ),
            GraphEdge(
                source_node_id="const",
                source_handle="n",
                target_node_id="knn",
                target_handle="n_neighbors",
            ),
        ],
    )

    result = execute_graph(graph, security=_custom_registry(const_node))

    assert result.previews["knn"]["model"]["params"]["n_neighbors"] == 11


def test_cycle_detection() -> None:
    code = "inp: ANY\nout = inp\nreturn out: ANY"
    graph = WorkflowGraph(
        nodes=[GraphNode(id="a", code=code), GraphNode(id="b", code=code)],
        edges=[
            GraphEdge(
                source_node_id="a",
                source_handle="out",
                target_node_id="b",
                target_handle="inp",
            ),
            GraphEdge(
                source_node_id="b",
                source_handle="out",
                target_node_id="a",
                target_handle="inp",
            ),
        ],
    )

    with pytest.raises(GraphCycleError, match="cycle"):
        topological_sort(graph)
    with pytest.raises(GraphCycleError):
        execute_graph(graph)


def test_untrusted_node_refused_until_approved() -> None:
    registry = SecurityRegistry()
    tampered_code = registry.get_template("load_iris").code + "\n# tweaked\n"
    graph = WorkflowGraph(nodes=[GraphNode(id="custom_load", code=tampered_code)])

    with pytest.raises(UntrustedNodeError) as excinfo:
        execute_graph(graph, security=registry)
    assert excinfo.value.node_id == "custom_load"
    assert excinfo.value.code_hash == code_hash(tampered_code)

    registry.approve("wrong_node", code_hash(tampered_code))
    with pytest.raises(UntrustedNodeError):
        execute_graph(graph, security=registry)

    registry.approve("custom_load", code_hash(tampered_code))
    result = execute_graph(graph, security=registry)
    assert result.outputs_by_name["custom_load"]["df"].height == 150

    tampering_registry = SecurityRegistry()
    tampering_registry.approve("custom_load", code_hash(tampered_code + "\n"))
    with pytest.raises(UntrustedNodeError):
        execute_graph(graph, security=tampering_registry)


def test_node_runtime_error_reports_raw_line() -> None:
    code = "k: int = 1\nvalue = k / 0\nreturn value: ANY"
    node = GraphNode(id="boom", code=code)
    graph = WorkflowGraph(nodes=[node])
    events: list[ExecutionEvent] = []

    with pytest.raises(NodeExecutionError) as excinfo:
        execute_graph(graph, security=_custom_registry(node), emit=events.append)

    error = excinfo.value
    assert error.line == 2
    assert error.error_type == "ZeroDivisionError"
    assert '<node:boom>", line 2' in error.traceback_text

    error_events = [
        event for event in events if event.type is EventType.NODE_ERROR
    ]
    assert len(error_events) == 1
    assert error_events[0].data["line"] == 2
    assert error_events[0].data["error_type"] == "ZeroDivisionError"
    assert '<node:boom>", line 2' in error_events[0].data["traceback"]
    assert (
        [event.type for event in events][:2]
        == [EventType.NODE_STARTED, EventType.NODE_ERROR]
    )


def test_output_contract_mismatch_reports_error() -> None:
    code = (
        "flag: bool = False\n"
        "if flag:\n"
        "    a = 1\n"
        "    return a: ANY\n"
        "return\n"
    )
    ok_graph = WorkflowGraph(
        nodes=[GraphNode(id="maybe", code=code, widget_values={"flag": True})]
    )
    registry = _custom_registry(ok_graph.nodes[0])
    ok_result = execute_graph(ok_graph, security=registry)
    assert ok_result.outputs_by_name["maybe"]["a"] == 1

    clear_cache()
    failing_graph = WorkflowGraph(nodes=[GraphNode(id="maybe", code=code)])
    events: list[ExecutionEvent] = []
    with pytest.raises(NodeExecutionError) as excinfo:
        execute_graph(failing_graph, security=registry, emit=events.append)

    assert "non-tuple" in excinfo.value.message
    assert excinfo.value.line is None
    error_events = [
        event for event in events if event.type is EventType.NODE_ERROR
    ]
    assert error_events[0].data["line"] is None


def test_missing_input_connection_raises() -> None:
    code = _templates()["knn_train"]
    graph = WorkflowGraph(nodes=[GraphNode(id="knn", code=code)])

    with pytest.raises(GraphValidationError, match="train"):
        execute_graph(graph)


def test_unknown_source_handle_raises() -> None:
    templates = _templates()
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="load", code=templates["load_iris"]),
            GraphNode(id="split", code=templates["train_test_split"]),
        ],
        edges=[
            GraphEdge(
                source_node_id="load",
                source_handle="nope",
                target_node_id="split",
                target_handle="df",
            )
        ],
    )

    with pytest.raises(GraphValidationError, match="nope"):
        execute_graph(graph)


def test_unknown_target_handle_raises() -> None:
    templates = _templates()
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="load", code=templates["load_iris"]),
            GraphNode(id="split", code=templates["train_test_split"]),
        ],
        edges=[
            GraphEdge(
                source_node_id="load",
                source_handle="df",
                target_node_id="split",
                target_handle="nope",
            )
        ],
    )

    with pytest.raises(GraphValidationError, match="nope"):
        execute_graph(graph)


def test_multiple_connections_to_same_input_raise() -> None:
    templates = _templates()
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="load_a", code=templates["load_iris"]),
            GraphNode(id="load_b", code=templates["load_iris"]),
            GraphNode(id="split", code=templates["train_test_split"]),
        ],
        edges=[
            GraphEdge(
                source_node_id="load_a",
                source_handle="df",
                target_node_id="split",
                target_handle="df",
            ),
            GraphEdge(
                source_node_id="load_b",
                source_handle="df",
                target_node_id="split",
                target_handle="df",
            ),
        ],
    )

    with pytest.raises(GraphValidationError, match="multiple"):
        execute_graph(graph)


def test_dangling_edge_raises() -> None:
    graph = WorkflowGraph(
        nodes=[GraphNode(id="load", code=_templates()["load_iris"])],
        edges=[
            GraphEdge(
                source_node_id="load",
                source_handle="df",
                target_node_id="ghost",
                target_handle="df",
            )
        ],
    )

    with pytest.raises(GraphValidationError, match="ghost"):
        execute_graph(graph)


def test_duplicate_node_ids_raise() -> None:
    code = _templates()["load_iris"]
    graph = WorkflowGraph(
        nodes=[GraphNode(id="load", code=code), GraphNode(id="load", code=code)]
    )

    with pytest.raises(GraphValidationError, match="Duplicate"):
        execute_graph(graph)


def test_node_syntax_error_raises_transpile_error() -> None:
    graph = WorkflowGraph(nodes=[GraphNode(id="bad", code="def broken(:\n")])

    with pytest.raises(TranspileError):
        execute_graph(graph)


def test_all_official_templates_parse_cleanly() -> None:
    registry = get_security_registry()
    names = {record.name for record in registry.templates()}
    assert names == set(EXPECTED_SOCKETS)

    for name, (inputs, outputs, widgets) in EXPECTED_SOCKETS.items():
        record = registry.get_template(name)
        schema = parse_node(record.code, name)
        assert schema.syntax_errors == []
        assert [socket.name for socket in schema.inputs] == inputs
        assert [socket.name for socket in schema.outputs] == outputs
        assert [widget.name for widget in schema.widgets] == widgets
        assert registry.is_official(record.code)
        assert registry.template_for_hash(record.code_hash).name == name


def test_template_socket_types() -> None:
    registry = get_security_registry()
    split = parse_node(registry.get_template("train_test_split").code)
    assert [socket.data_type for socket in split.inputs] == [SocketType.DATAFRAME]
    assert [socket.data_type for socket in split.outputs] == [
        SocketType.DATASET,
        SocketType.DATASET,
    ]

    evaluate = parse_node(registry.get_template("evaluate_metrics").code)
    assert [socket.data_type for socket in evaluate.inputs] == [
        SocketType.ML_MODEL,
        SocketType.DATASET,
    ]
    assert evaluate.outputs[0].data_type is SocketType.METRICS


def test_security_registry_hashes_and_normalizes_line_endings() -> None:
    registry = SecurityRegistry()
    assert len(registry.templates()) == 7
    for record in registry.templates():
        assert record.code_hash == code_hash(record.code)
        assert registry.is_official(record.code)
        assert registry.is_official(record.code.replace("\n", "\r\n"))


def test_modified_template_is_not_official() -> None:
    registry = SecurityRegistry()
    modified = registry.get_template("load_iris").code + "# comment\n"
    assert not registry.is_official(modified)
    assert registry.template_for_hash(code_hash(modified)) is None


def test_missing_dependencies_detection() -> None:
    assert missing_dependencies(["pytest>=7"]) == []
    assert missing_dependencies(["packaging"]) == []
    assert missing_dependencies(["definitely-not-a-real-pkg-xyz>=1.0"]) == [
        "definitely-not-a-real-pkg-xyz>=1.0"
    ]
    assert missing_dependencies(["pytest>=99999"]) == ["pytest>=99999"]
    assert missing_dependencies(["%%%bad-requirement%%%"]) == [
        "%%%bad-requirement%%%"
    ]


def test_install_dependencies_invokes_uv_pip(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("app.env_manager.subprocess.run", fake_run)
    install_dependencies(["tiny-package>=0.1"])

    assert calls["command"] == ["uv", "pip", "install", "tiny-package>=0.1"]
    assert calls["kwargs"]["check"] is True


def test_install_dependencies_failure_wrapped(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="resolution failed")

    monkeypatch.setattr("app.env_manager.subprocess.run", fake_run)

    with pytest.raises(DependencyInstallError, match="resolution failed"):
        install_dependencies(["tiny-package>=0.1"])


def test_installing_deps_event_streamed_before_started(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.executor.missing_dependencies", lambda deps: ["fake-dep>=1.0"]
    )
    installed: list[list[str]] = []
    monkeypatch.setattr("app.executor.install_dependencies", installed.append)

    events: list[ExecutionEvent] = []
    graph = WorkflowGraph(
        nodes=[GraphNode(id="load", code=_templates()["load_iris"])]
    )
    execute_graph(graph, emit=events.append)

    assert [event.type for event in events] == [
        EventType.INSTALLING_DEPS,
        EventType.NODE_STARTED,
        EventType.NODE_COMPLETED,
    ]
    assert events[0].node_id == "load"
    assert events[0].data["dependencies"] == ["fake-dep>=1.0"]
    assert installed == [["fake-dep>=1.0"]]


def test_allow_installs_false_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.executor.missing_dependencies", lambda deps: ["fake-dep>=1.0"]
    )
    graph = WorkflowGraph(
        nodes=[GraphNode(id="load", code=_templates()["load_iris"])]
    )

    with pytest.raises(DependencyInstallError, match="fake-dep"):
        execute_graph(graph, allow_installs=False)


def test_cancellation_between_nodes_keeps_completed_cache() -> None:
    stop = {"value": False}

    def emit(event: ExecutionEvent) -> None:
        if event.type is EventType.NODE_COMPLETED:
            stop["value"] = True

    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="load", code=_templates()["load_iris"]),
            GraphNode(id="split", code=_templates()["train_test_split"]),
        ],
        edges=[
            GraphEdge(
                source_node_id="load",
                source_handle="df",
                target_node_id="split",
                target_handle="df",
            )
        ],
    )

    with pytest.raises(ExecutionCancelled, match="cancelled"):
        execute_graph(graph, emit=emit, should_stop=lambda: stop["value"])

    events: list[ExecutionEvent] = []
    result = execute_graph(graph, emit=events.append)

    assert result.cached == {"load": True, "split": False}
    started = [
        event.node_id
        for event in events
        if event.type is EventType.NODE_STARTED
    ]
    assert started == ["split"]


def test_cancellation_before_first_node_runs_nothing() -> None:
    events: list[ExecutionEvent] = []

    with pytest.raises(ExecutionCancelled):
        execute_graph(
            _chain_graph(), emit=events.append, should_stop=lambda: True
        )

    assert events == []


def test_run_to_target_scopes_order_and_outputs() -> None:
    events: list[ExecutionEvent] = []
    result = execute_graph(_chain_graph(), emit=events.append, targets=["knn"])

    assert list(result.cached) == ["load", "split", "knn"]
    assert result.cached == {"load": False, "split": False, "knn": False}
    assert "evaluate" not in result.outputs_by_name
    assert "evaluate" not in result.previews
    started = [
        event.node_id for event in events if event.type is EventType.NODE_STARTED
    ]
    assert started == ["load", "split", "knn"]


def test_run_to_target_skips_untrusted_downstream_node() -> None:
    templates = _templates()
    graph = WorkflowGraph(
        nodes=[
            GraphNode(id="load", code=templates["load_iris"]),
            GraphNode(id="split", code=templates["train_test_split"]),
            GraphNode(
                id="knn",
                code=templates["knn_train"],
                widget_values={"n_neighbors": 5},
            ),
            GraphNode(
                id="evaluate",
                code=templates["evaluate_metrics"] + "\n# tampered\n",
            ),
        ],
        edges=_chain_graph().edges,
    )

    plan = validate_graph(graph, targets=["knn"])
    assert plan.untrusted == []
    assert plan.order == ["load", "split", "knn"]

    full_plan = validate_graph(graph)
    assert [node_id for node_id, _hash in full_plan.untrusted] == ["evaluate"]

    result = execute_graph(graph, targets=["knn"])
    assert result.cached == {"load": False, "split": False, "knn": False}


def test_unknown_and_empty_targets_raise() -> None:
    with pytest.raises(GraphValidationError, match="Unknown target"):
        validate_graph(_chain_graph(), targets=["ghost"])
    with pytest.raises(GraphValidationError, match="Unknown target"):
        execute_graph(_chain_graph(), targets=["ghost"])
    with pytest.raises(GraphValidationError, match="must not be empty"):
        validate_graph(_chain_graph(), targets=[])
