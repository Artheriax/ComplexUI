"""Tests for rich output capture (stdout, tables, images, HTML and Plotly)."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from app.output_capture import OutputCapture
from app.schemas import (
    HtmlOutput,
    ImageOutput,
    PlotlyOutput,
    TableOutput,
    TextOutput,
)


def display(*_objects: object) -> None:
    raise AssertionError("display() should be injected by OutputCapture")


def _capture(func) -> list:
    capture = OutputCapture()
    with capture.activate(func):
        func()
    return capture.outputs


def test_stdout_and_display_are_interleaved() -> None:
    def node() -> None:
        print("before")
        display("middle")
        print("after")

    outputs = _capture(node)

    assert [output.kind for output in outputs] == ["text", "text", "text"]
    assert outputs[0] == TextOutput(source="print", text="before\n")
    assert outputs[1] == TextOutput(source="display", text="middle")
    assert outputs[2] == TextOutput(source="print", text="after\n")


def test_stdout_buffer_writes_are_captured() -> None:
    import sys

    def node() -> None:
        sys.stdout.buffer.write(b"raw bytes\n")

    outputs = _capture(node)

    assert len(outputs) == 1
    assert outputs[0] == TextOutput(source="print", text="raw bytes\n")


def test_polars_display_becomes_a_truncated_table() -> None:
    import polars as pl

    frame = pl.DataFrame({f"c{index}": list(range(40)) for index in range(35)})

    def node() -> None:
        display(frame)

    outputs = _capture(node)

    assert len(outputs) == 1
    table = outputs[0]
    assert isinstance(table, TableOutput)
    assert table.total_rows == 40
    assert table.total_cols == 35
    assert len(table.columns) == 30
    assert len(table.rows) == 20
    assert table.truncated is True


def test_pandas_display_becomes_a_truncated_table() -> None:
    import pandas as pd

    frame = pd.DataFrame({f"c{index}": range(40) for index in range(35)})

    def node() -> None:
        display(frame)

    outputs = _capture(node)

    assert len(outputs) == 1
    table = outputs[0]
    assert isinstance(table, TableOutput)
    assert table.columns == [f"c{index}" for index in range(30)]
    assert table.total_rows == 40
    assert table.total_cols == 35
    assert len(table.rows) == 20
    assert len(table.rows[0]) == 30
    assert table.truncated is True


def test_matplotlib_figures_are_captured_and_closed() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def node() -> None:
        plt.plot([1, 2, 3], [3, 2, 1])
        plt.show()

    outputs = _capture(node)

    images = [output for output in outputs if isinstance(output, ImageOutput)]
    assert len(images) == 1
    assert images[0].mime == "image/png"
    assert images[0].source == "matplotlib"
    assert base64.b64decode(images[0].data).startswith(b"\x89PNG")
    assert plt.get_fignums() == []


def test_repeated_show_calls_capture_distinct_figures() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    def node() -> None:
        for index in range(3):
            plt.plot([index, index + 1], [index + 1, index])
            plt.show()
            go.Figure(data=[go.Scatter(x=[index], y=[index])]).show()

    outputs = _capture(node)

    assert [output.kind for output in outputs] == ["image", "plotly"] * 3


def test_plotly_show_and_display_do_not_duplicate() -> None:
    import plotly.graph_objects as go

    figure = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])

    def node() -> None:
        display(figure)
        figure.show()

    outputs = _capture(node)

    plots = [output for output in outputs if isinstance(output, PlotlyOutput)]
    assert len(plots) == 1
    assert plots[0].figure["data"][0]["type"] == "scatter"


def test_mimebundle_plotly_payload_is_json_safe() -> None:
    class Marker:
        def __str__(self) -> str:
            return "converted"

    class Figure:
        def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
            return {
                "application/vnd.plotly.v1+json": {
                    "data": [
                        {
                            "type": "scatter",
                            "marker": {"color": Marker()},
                            "y": [1.0, float("nan"), float("inf")],
                        }
                    ],
                    "layout": {},
                }
            }

    outputs = _capture(lambda: display(Figure()))

    assert len(outputs) == 1
    plot = outputs[0]
    assert isinstance(plot, PlotlyOutput)
    assert plot.figure["data"][0]["marker"]["color"] == "converted"
    assert plot.figure["data"][0]["y"] == [1.0, None, None]
    assert "NaN" not in json.dumps(plot.figure)
    assert "Infinity" not in json.dumps(plot.figure)


def test_html_and_image_protocols() -> None:
    class Html:
        def _repr_html_(self) -> str:
            return "<b>bold</b>"

    class Image:
        def _repr_png_(self) -> str:
            return base64.b64encode(b"png-bytes").decode("ascii")

    class Vector:
        def _repr_svg_(self) -> str:
            return "<svg xmlns='http://www.w3.org/2000/svg'><circle r='4'/></svg>"

    def node() -> None:
        display(Html())
        display(Image())
        display(Vector())

    outputs = _capture(node)

    assert any(isinstance(output, HtmlOutput) for output in outputs)
    images = [output for output in outputs if isinstance(output, ImageOutput)]
    assert images[0].mime == "image/png"
    assert base64.b64decode(images[0].data) == b"png-bytes"
    assert images[1].mime == "image/svg+xml"
    assert base64.b64decode(images[1].data).startswith(b"<svg")


def test_plotting_modules_imported_inside_a_node_are_patched() -> None:
    script = textwrap.dedent(
        """
        from app.output_capture import OutputCapture

        def plotly_node():
            import plotly.graph_objects as go
            go.Figure(data=[go.Scatter(x=[1], y=[2])]).show()

        def matplotlib_node():
            import matplotlib.pyplot as plt
            plt.plot([1, 2, 3], [3, 2, 1])
            plt.show()

        plotly_capture = OutputCapture()
        with plotly_capture.activate(plotly_node):
            plotly_node()
        assert [output.kind for output in plotly_capture.outputs] == ["plotly"], (
            plotly_capture.outputs
        )

        matplotlib_capture = OutputCapture()
        with matplotlib_capture.activate(matplotlib_node):
            matplotlib_node()
        assert [output.kind for output in matplotlib_capture.outputs] == ["image"], (
            matplotlib_capture.outputs
        )
        print("captured")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "captured" in completed.stdout


def test_stdout_is_captured_before_a_node_error() -> None:
    def node() -> None:
        print("debug line")
        raise RuntimeError("boom")

    capture = OutputCapture()
    try:
        with capture.activate(node):
            node()
    except RuntimeError:
        pass

    assert capture.outputs[0].kind == "text"
    assert capture.outputs[0].text == "debug line\n"
