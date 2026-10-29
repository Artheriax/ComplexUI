"""Tests for the AST transpiler (Milestone 1 verification)."""

from __future__ import annotations

import inspect
import statistics
import time

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from app.schemas import SocketType, WidgetCondition, WidgetParam
from app.transpiler import TranspileError, parse_node, transpile_node

SPEC_EXAMPLE = '''\
# /// script
# dependencies = ["scikit-learn>=1.5", "polars>=1.0"]
# ///
from sklearn.neighbors import KNeighborsClassifier

# 1. Unassigned type annotations -> INPUT SOCKETS
train_data: DATASET

# 2. Assigned type annotations -> UI WIDGETS + OPTIONAL INPUT SOCKETS
n_neighbors: int = 5
weights: str = "uniform"

# 3. Standard Python Execution Logic
X_train, y_train = train_data
model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
model.fit(X_train, y_train)

# 4. Typed Return Statement -> OUTPUT SOCKETS
return model: ML_MODEL
'''


def _node_traceback_lineno(exc: BaseException, node_id: str) -> int:
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == f"<node:{node_id}>":
            return tb.tb_lineno
        tb = tb.tb_next
    raise AssertionError("no frame from node code found in traceback")


def test_spec_example_schema() -> None:
    schema = parse_node(SPEC_EXAMPLE, node_id="demo")

    assert schema.node_id == "demo"
    assert schema.syntax_errors == []
    assert schema.dependencies == ["scikit-learn>=1.5", "polars>=1.0"]
    assert [(s.name, s.data_type) for s in schema.inputs] == [
        ("train_data", SocketType.DATASET)
    ]
    assert [(w.name, w.data_type, w.default) for w in schema.widgets] == [
        ("n_neighbors", "int", 5),
        ("weights", "str", "uniform"),
    ]
    assert [(s.name, s.data_type) for s in schema.outputs] == [
        ("model", SocketType.ML_MODEL)
    ]


def test_spec_example_executes_with_dummy_inputs() -> None:
    result = transpile_node(SPEC_EXAMPLE, node_id="demo")

    features = np.array([[float(i), float(i)] for i in range(12)])
    labels = np.array([0] * 6 + [1] * 6)

    (model,) = result.func(train_data=(features, labels))

    assert isinstance(model, KNeighborsClassifier)
    assert model.get_params()["n_neighbors"] == 5
    assert model.get_params()["weights"] == "uniform"
    assert model.predict(np.array([[0.5, 0.5]]))[0] == 0

    (custom,) = result.func(train_data=(features, labels), n_neighbors=3)
    assert custom.get_params()["n_neighbors"] == 3


def test_dependencies_missing_header_returns_empty_list() -> None:
    schema = parse_node("x: DATASET\nreturn x: DATASET")
    assert schema.dependencies == []
    assert schema.syntax_errors == []


def test_dependencies_malformed_toml_reports_error() -> None:
    code = "# /// script\n# dependencies = [oops\n# ///\nx = 1\n"
    schema = parse_node(code)
    assert any("PEP 723" in err.message for err in schema.syntax_errors)


def test_unassigned_annotation_becomes_required_input_socket() -> None:
    code = "data: DATASET\nreturn data: DATASET"
    result = transpile_node(code, node_id="t")

    assert [s.name for s in result.schema.inputs] == ["data"]
    assert [s.name for s in result.schema.outputs] == ["data"]
    assert result.func(data=(1, 2)) == ((1, 2),)
    with pytest.raises(TypeError):
        result.func()


def test_assigned_annotation_becomes_widget_with_default() -> None:
    code = "k: int = 5\nreturn k: ANY"
    result = transpile_node(code, node_id="t")

    assert [(w.name, w.data_type, w.default) for w in result.schema.widgets] == [
        ("k", "int", 5)
    ]
    assert result.schema.inputs == []
    assert result.func() == (5,)
    assert result.func(k=7) == (7,)


def test_literal_widget_choices_round_trip() -> None:
    code = (
        "from typing import Literal\n"
        "level: Literal['low', 'medium', 'high'] = 'medium'\n"
        "return level: STRING"
    )
    result = transpile_node(code, node_id="t")

    assert result.schema.widgets == [
        WidgetParam(
            name="level",
            data_type="str",
            default="medium",
            options=["low", "medium", "high"],
        )
    ]
    assert result.func() == ("medium",)
    assert result.func(level="high") == ("high",)


def test_literal_widget_choices_support_numeric_and_qualified_annotations() -> None:
    schema = parse_node(
        "count: typing.Literal[1, 2, 3] = 2\n"
        "ratio: Literal[0.25, 0.5] = 0.5\n"
        "return count: INT, ratio: FLOAT"
    )

    assert schema.syntax_errors == []
    assert [(w.name, w.data_type, w.options) for w in schema.widgets] == [
        ("count", "int", [1, 2, 3]),
        ("ratio", "float", [0.25, 0.5]),
    ]


def test_literal_widget_accepts_string_annotations() -> None:
    schema = parse_node(
        "level: \"Literal['low', 'high']\" = 'high'\n"
        "return level: STRING"
    )

    assert schema.syntax_errors == []
    assert schema.widgets[0].options == ["low", "high"]
    assert schema.widgets[0].data_type == "str"


def test_show_if_widget_marker_round_trip() -> None:
    code = (
        "from typing import Literal\n"
        "mode: Literal['a', 'b'] = 'a'\n"
        "value: int = ShowIf(default=5, mode='b')\n"
        "return value: INT"
    )
    result = transpile_node(code, node_id="t")

    assert result.schema.syntax_errors == []
    assert result.schema.widgets == [
        WidgetParam(
            name="mode",
            data_type="str",
            default="a",
            options=["a", "b"],
        ),
        WidgetParam(
            name="value",
            data_type="int",
            default=5,
            visible_when=[
                WidgetCondition(widget="mode", values=["b"]),
            ],
        ),
    ]
    assert result.func() == (5,)
    assert result.func(mode="b", value=7) == (7,)


def test_show_if_supports_multiple_conditions_and_alternatives() -> None:
    schema = parse_node(
        "mode: Literal['a', 'b', 'c'] = 'a'\n"
        "preset: Literal['x', 'y'] = 'x'\n"
        "value: float = ShowIf(\n"
        "    default=1.0,\n"
        "    mode=['b', 'c'],\n"
        "    preset='y',\n"
        ")\n"
        "flag: bool = True\n"
        "other: int = ShowIf(default=2, flag=False)\n"
        "return value: FLOAT"
    )

    assert schema.syntax_errors == []
    assert schema.widgets[2].visible_when == [
        WidgetCondition(widget="mode", values=["b", "c"]),
        WidgetCondition(widget="preset", values=["y"]),
    ]
    assert schema.widgets[4].visible_when == [
        WidgetCondition(widget="flag", values=[False]),
    ]


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            "value: int = ShowIf(mode='x')\nreturn value: INT",
            "default",
        ),
        (
            "mode: str = 'x'\nvalue: int = ShowIf(default=1)\nreturn value: INT",
            "widget condition",
        ),
        (
            "value: int = ShowIf(default=1, missing='x')\nreturn value: INT",
            "unknown widget",
        ),
        (
            "mode: Literal['x'] = 'x'\n"
            "value: int = ShowIf(default=1, mode='y')\n"
            "return value: INT",
            "must be one of",
        ),
        (
            "value: int = ShowIf(default=1, value='x')\nreturn value: INT",
            "its own widget",
        ),
    ],
)
def test_show_if_rejects_invalid_conditions(code: str, message: str) -> None:
    schema = parse_node(code)

    assert any(message in error.message for error in schema.syntax_errors)


def test_literal_widget_rejects_default_outside_choices() -> None:
    schema = parse_node(
        "level: Literal['low', 'high'] = 'medium'\n"
        "return level: STRING"
    )

    assert any("must be one of" in err.message for err in schema.syntax_errors)


def test_literal_widget_rejects_mixed_choice_types() -> None:
    schema = parse_node(
        "level: Literal['low', 2] = 'low'\n"
        "return level: STRING"
    )

    assert any(
        "share one scalar type" in err.message
        for err in schema.syntax_errors
    )


def test_widget_literal_defaults_round_trip() -> None:
    code = (
        "flag: bool = True\n"
        "ratio: float = 1.5\n"
        "label: str = 'hello'\n"
        "items: list = [1, 2, 3]\n"
        "mapping: dict = {'a': 1}\n"
        "nothing: str = None\n"
        "return flag: ANY, ratio: ANY, label: ANY, items: ANY, "
        "mapping: ANY, nothing: ANY"
    )
    result = transpile_node(code, node_id="t")

    assert result.func() == (True, 1.5, "hello", [1, 2, 3], {"a": 1}, None)


def test_typed_single_return_is_always_a_tuple() -> None:
    code = "class FakeModel:\n    pass\n\nmodel = FakeModel()\nreturn model: ML_MODEL"
    result = transpile_node(code, node_id="t")

    assert [(s.name, s.data_type) for s in result.schema.outputs] == [
        ("model", SocketType.ML_MODEL)
    ]
    outputs = result.func()
    assert isinstance(outputs, tuple)
    assert len(outputs) == 1
    assert type(outputs[0]).__name__ == "FakeModel"


def test_multi_typed_return_preserves_order() -> None:
    code = "a: DATASET\nb: DATASET\nc = a\nd = b\nreturn d: DATASET, c: DATASET"
    result = transpile_node(code, node_id="t")

    assert [s.name for s in result.schema.outputs] == ["d", "c"]
    assert result.func(a=1, b=2) == (2, 1)


def test_multiline_helper_signatures_do_not_claim_top_level_returns() -> None:
    code = (
        "def helper(\n"
        "    value: int,\n"
        ") -> int:\n"
        "    return value + 1\n"
        "\n"
        "result = helper(41)\n"
        "return result: INT\n"
    )
    result = transpile_node(code, node_id="helper_node")

    assert result.schema.syntax_errors == []
    assert [(socket.name, socket.data_type) for socket in result.schema.outputs] == [
        ("result", SocketType.INT)
    ]
    assert result.func() == (42,)


def test_bare_return_defaults_to_any() -> None:
    code = "val = 42\nreturn val"
    result = transpile_node(code, node_id="t")

    assert [(s.name, s.data_type) for s in result.schema.outputs] == [
        ("val", SocketType.ANY)
    ]
    assert result.func() == (42,)


def test_bare_return_of_tuple_value_is_containerized() -> None:
    code = "data = (1, 2)\nreturn data"
    result = transpile_node(code, node_id="t")

    assert result.schema.outputs[0].name == "data"
    assert result.func() == ((1, 2),)


def test_bare_tuple_return_is_valid_python_and_keeps_names() -> None:
    code = "left = 1\nright = 2\nreturn left, right"
    result = transpile_node(code, node_id="t")

    assert [(s.name, s.data_type) for s in result.schema.outputs] == [
        ("left", SocketType.ANY),
        ("right", SocketType.ANY),
    ]
    assert result.func() == (1, 2)


def test_return_none_registers_no_outputs() -> None:
    code = "data: DATASET\nreturn None"
    result = transpile_node(code, node_id="t")

    assert result.schema.outputs == []
    assert result.func(data=1) is None


def test_bare_return_registers_no_outputs() -> None:
    code = "data: DATASET\nreturn"
    result = transpile_node(code, node_id="t")

    assert result.schema.outputs == []
    assert result.func(data=1) is None


def test_return_inside_nested_def_is_untouched() -> None:
    code = (
        "def helper(value):\n"
        "    return value * 2\n"
        "\n"
        "n: int = 4\n"
        "out = helper(n)\n"
        "return out: METRICS"
    )
    result = transpile_node(code, node_id="t")

    assert [(s.name, s.data_type) for s in result.schema.outputs] == [
        ("out", SocketType.METRICS)
    ]
    assert result.func() == (8,)


def test_return_inside_top_level_if_stays_conditionally_scoped() -> None:
    code = (
        "flag: bool = True\n"
        "if flag:\n"
        "    value = 10\n"
        "    return value: METRICS\n"
    )
    result = transpile_node(code, node_id="t")

    assert result.func(flag=True) == (10,)
    assert result.func(flag=False) is None


def test_syntax_error_reports_raw_line_number() -> None:
    code = "x = 1\ndef broken(:\n"
    schema = parse_node(code, node_id="t")

    assert schema.syntax_errors
    assert schema.syntax_errors[0].line == 2
    with pytest.raises(TranspileError) as excinfo:
        transpile_node(code, node_id="t")
    assert excinfo.value.schema.syntax_errors[0].line == 2


def test_traceback_line_numbers_map_to_raw_snippet() -> None:
    code = "a: DATASET\nraise RuntimeError('boom')\n"
    result = transpile_node(code, node_id="t")

    with pytest.raises(RuntimeError, match="boom") as excinfo:
        result.func(a=1)

    wrapped_line = _node_traceback_lineno(excinfo.value, "t")
    assert wrapped_line - result.line_offset == 2


def test_imports_are_hoisted_to_module_header() -> None:
    code = "import math\nk: int = 16\nvalue = math.sqrt(k)\nreturn value: METRICS"
    result = transpile_node(code, node_id="t")

    assert result.source.startswith("import math\n")
    assert result.func() == (4.0,)


def test_signature_orders_inputs_before_widgets() -> None:
    code = (
        "n_neighbors: int = 3\n"
        "data: DATASET\n"
        "first, second = data\n"
        "return first: ANY, second: ANY"
    )
    result = transpile_node(code, node_id="t")

    params = list(inspect.signature(result.func).parameters)
    assert params == ["data", "n_neighbors"]
    assert result.func(data=(1, 2)) == (1, 2)


def test_optional_inputs_support_optional_and_union_annotations() -> None:
    code = (
        "from typing import Optional\n"
        "image1: Optional[IMAGE] = None\n"
        "image2: IMAGE | None = None\n"
        "amount: int = 5\n"
        "return image1: ANY, image2: ANY, amount: INT"
    )
    result = transpile_node(code, node_id="t")

    assert result.schema.syntax_errors == []
    assert [
        (socket.name, socket.data_type, socket.optional)
        for socket in result.schema.inputs
    ] == [
        ("image1", SocketType.IMAGE, True),
        ("image2", SocketType.IMAGE, True),
    ]
    assert [widget.name for widget in result.schema.widgets] == ["amount"]
    assert list(inspect.signature(result.func).parameters) == [
        "image1",
        "image2",
        "amount",
    ]
    assert result.func() == (None, None, 5)
    assert result.func(image1="one", image2="two", amount=2) == (
        "one",
        "two",
        2,
    )


def test_optional_input_accepts_string_annotation_and_required_first() -> None:
    code = (
        "required: IMAGE\n"
        "optional: \"IMAGE | None\" = None\n"
        "return required: IMAGE, optional: ANY"
    )
    result = transpile_node(code, node_id="t")

    assert [socket.name for socket in result.schema.inputs] == [
        "required",
        "optional",
    ]
    assert result.schema.inputs[1].optional is True
    assert list(inspect.signature(result.func).parameters) == [
        "required",
        "optional",
    ]


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            "from typing import Optional\n"
            "image: Optional[IMAGE]\n"
            "return image: ANY",
            "must declare",
        ),
        (
            "from typing import Optional\n"
            "image: Optional[IMAGE] = 'x'\n"
            "return image: ANY",
            "must declare",
        ),
        (
            "image: IMAGE | STRING = None\nreturn image: ANY",
            "Socket unions",
        ),
        (
            "from typing import Optional\n"
            "image: Optional[IMAGE, STRING] = None\n"
            "return image: ANY",
            "exactly one",
        ),
    ],
)
def test_optional_input_rejects_invalid_annotations(
    code: str, message: str
) -> None:
    schema = parse_node(code)

    assert any(message in error.message for error in schema.syntax_errors)


def test_unknown_socket_type_maps_to_any() -> None:
    schema = parse_node("custom: SomethingExotic\nreturn custom: AlsoExotic")
    assert schema.inputs[0].data_type is SocketType.ANY
    assert schema.outputs[0].data_type is SocketType.ANY


def test_string_literal_annotation_is_unwrapped() -> None:
    schema = parse_node('x: "DATASET"\nreturn x: DATASET')
    assert schema.inputs[0].data_type is SocketType.DATASET


def test_non_literal_widget_default_reports_error() -> None:
    schema = parse_node("x: int = len('abc')\nreturn x: ANY")
    assert any("literal" in err.message for err in schema.syntax_errors)
    with pytest.raises(TranspileError):
        transpile_node("x: int = len('abc')\nreturn x: ANY")


def test_duplicate_declaration_reports_error() -> None:
    schema = parse_node("x: DATASET\nx: int = 5")
    assert any("Duplicate" in err.message for err in schema.syntax_errors)


def test_duplicate_output_name_reports_error() -> None:
    schema = parse_node("a = 1\nreturn a: ANY, a: ANY")
    assert any("duplicate" in err.message.lower() for err in schema.syntax_errors)


def test_unsupported_return_expression_reports_error() -> None:
    schema = parse_node("x = 1\nreturn x + 1")
    assert any("Return items" in err.message for err in schema.syntax_errors)


def test_multiline_typed_return_preserves_source_lines() -> None:
    code = 'data: DATASET\nreturn (\n    data: DATASET\n)\n'
    result = transpile_node(code, node_id="t")

    assert [(s.name, s.data_type) for s in result.schema.outputs] == [
        ("data", SocketType.DATASET)
    ]
    assert result.func((1, 2)) == ((1, 2),)
    wrapped_lines = result.source.split("\n")
    raw_lines = code.split("\n")
    assert len(wrapped_lines) == len(raw_lines) + result.line_offset


def test_multiline_string_content_is_not_reindented() -> None:
    code = 'text = """a\nb"""\nout = text.splitlines()\nreturn out: METRICS'
    result = transpile_node(code, node_id="t")

    assert result.func() == (["a", "b"],)


def test_empty_code_transpiles_to_noop_callable() -> None:
    result = transpile_node("", node_id="t")

    assert result.schema.inputs == []
    assert result.schema.widgets == []
    assert result.schema.outputs == []
    assert result.func() is None


def test_parse_node_never_raises_on_garbage() -> None:
    for garbage in ("def broken(:", "\x00", "return return", "class:"):
        schema = parse_node(garbage)
        assert isinstance(schema.syntax_errors, list)


def test_parse_performance_under_5ms() -> None:
    parse_node(SPEC_EXAMPLE, node_id="warmup")

    timings: list[float] = []
    for _ in range(100):
        start = time.perf_counter()
        parse_node(SPEC_EXAMPLE, node_id="perf")
        timings.append(time.perf_counter() - start)

    assert statistics.median(timings) < 0.005
