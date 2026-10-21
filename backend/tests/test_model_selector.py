"""ModelSelector DSL marker parsing."""

from __future__ import annotations

from app.transpiler import TranspileError, parse_node, transpile_node


def test_model_selector_marker_becomes_widget() -> None:
    schema = parse_node(
        'local_model: ModelSelector = ModelSelector(type="yolo")\n'
        "return local_model: STRING"
    )

    assert schema.syntax_errors == []
    assert [(widget.name, widget.data_type, widget.default) for widget in schema.widgets] == [
        ("local_model", "ModelSelector", "")
    ]
    assert schema.widgets[0].model_type == "yolo"


def test_model_selector_runtime_value_is_supplied_by_widget() -> None:
    result = transpile_node(
        'local_model: ModelSelector = ModelSelector(type="torchvision")\n'
        "return local_model: STRING",
        node_id="selector",
    )

    assert result.func() == ("",)
    assert result.func(local_model="models/torchvision/model.pth") == (
        "models/torchvision/model.pth",
    )


def test_model_selector_accepts_qualified_marker() -> None:
    schema = parse_node(
        'model: ModelSelector = complexui.ModelSelector(type="classification")\n'
        "return model: STRING"
    )
    assert schema.syntax_errors == []
    assert schema.widgets[0].model_type == "classification"


def test_model_selector_requires_type_and_string_literal() -> None:
    for code, message in [
        (
            "model: ModelSelector = ModelSelector()\nreturn model: STRING",
            "requires a type",
        ),
        (
            'model: ModelSelector = ModelSelector(type="yolo/escape")\n'
            "return model: STRING",
            "must match",
        ),
        (
            "model: ModelSelector = ModelSelector(type=kind)\nreturn model: STRING",
            "string literal",
        ),
        (
            "model: ModelSelector\nreturn model: STRING",
            "require",
        ),
    ]:
        schema = parse_node(code)
        assert any(message in error.message for error in schema.syntax_errors)
        try:
            transpile_node(code)
        except TranspileError:
            pass
        else:
            raise AssertionError(f"Expected TranspileError for {code!r}")
