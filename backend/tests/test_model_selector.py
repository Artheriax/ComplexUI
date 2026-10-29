"""ModelSelector DSL marker parsing."""

from __future__ import annotations

import pytest

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


def test_model_selector_records_selection_kind_and_requirement() -> None:
    schema = parse_node(
        'model: ModelSelector = ModelSelector(\n'
        '    type="diffusion", select="any", required=True\n'
        ")\n"
        "return model: STRING"
    )

    assert schema.syntax_errors == []
    widget = schema.widgets[0]
    assert widget.model_type == "diffusion"
    assert widget.model_select == "any"
    assert widget.model_required is True
    assert widget.model_include == []


def test_model_selector_records_included_model_types() -> None:
    schema = parse_node(
        'model: ModelSelector = ModelSelector(\n'
        '    type="vae", select="any", required=True,\n'
        '    include=["diffusion", "unet"]\n'
        ")\n"
        "return model: STRING"
    )

    assert schema.syntax_errors == []
    widget = schema.widgets[0]
    assert widget.model_type == "vae"
    assert widget.model_include == ["diffusion", "unet"]


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        ('ModelSelector(type="yolo", select="files")', "select must be one of"),
        ('ModelSelector(type="yolo", required="yes")', "required must be a boolean"),
        ('ModelSelector(type="yolo", other="x")', "only type, select, required and include"),
        ('ModelSelector(type="yolo", required=True, required=False)', "duplicate"),
        ('ModelSelector(select="any")', "requires a type"),
        ('ModelSelector(type="yolo", include=[])', "1-8"),
        (
            'ModelSelector(type="yolo", include=["clip", "clip"])',
            "repeats",
        ),
        ('ModelSelector(type="yolo", include=["yolo"])', "must not repeat"),
        (
            'ModelSelector(type="yolo", include=["bad/type"])',
            "include entries must match",
        ),
    ],
)
def test_model_selector_rejects_invalid_selection_options(
    marker: str, message: str
) -> None:
    schema = parse_node(f"model: ModelSelector = {marker}\nreturn model: STRING")

    assert any(message in error.message for error in schema.syntax_errors)


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
