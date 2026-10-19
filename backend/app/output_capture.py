"""Capture stdout and Jupyter-style rich displays for one node execution.

Node code runs in a worker thread and is allowed to use ``print``, ``display``,
matplotlib/seaborn and Plotly.  This module turns those side effects into a
small, JSON-safe list of :mod:`app.schemas` rich-output models.  Capture is
strictly best effort: a broken repr, an oversized figure or a missing optional
library must never fail the node itself.
"""

from __future__ import annotations

import base64
import builtins
import io
import json
import os
import sys
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, time
from typing import Any

from .schemas import (
    HtmlOutput,
    ImageOutput,
    PlotlyOutput,
    RichOutput,
    TableOutput,
    TextOutput,
)

__all__ = [
    "MAX_HTML_CHARS",
    "MAX_IMAGE_BYTES",
    "MAX_NODE_OUTPUT_BYTES",
    "MAX_OUTPUTS",
    "MAX_PLOTLY_BYTES",
    "MAX_TABLE_COLS",
    "MAX_TABLE_ROWS",
    "MAX_TEXT_CHARS",
    "OutputCapture",
    "rich_output_size",
]

# A server-side IDE should never try to open a GUI window or a browser.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PLOTLY_RENDERER", "json")

MAX_OUTPUTS = 10
MAX_TEXT_CHARS = 12_000
MAX_TABLE_ROWS = 20
MAX_TABLE_COLS = 30
MAX_CELL_CHARS = 200
MAX_IMAGE_BYTES = 1_000_000
MAX_PLOTLY_BYTES = 1_500_000
MAX_HTML_CHARS = 60_000
MAX_NODE_OUTPUT_BYTES = 2_500_000

_MISSING = object()
_TRUNCATION_NOTICE = "Output truncated — the node produced more than the display limit."


class _StdoutBuffer:
    """Binary facade for code that writes to ``sys.stdout.buffer``."""

    def __init__(self, capture: OutputCapture) -> None:
        self._capture = capture

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        self._capture._write_stdout(bytes(data).decode("utf-8", "replace"))
        return len(data)

    def flush(self) -> None:
        return None


class _Stdout(io.TextIOBase):
    """Text stream that retains a bounded stdout buffer for the active node."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, capture: OutputCapture) -> None:
        super().__init__()
        self._capture = capture
        self.buffer = _StdoutBuffer(capture)

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._capture._write_stdout(text)
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno is unavailable while capturing output")


class OutputCapture:
    """Collect stdout, tables, images, HTML and Plotly figures for a node."""

    def __init__(self) -> None:
        self.outputs: list[RichOutput] = []
        self._pending: list[str] = []
        self._pending_chars = 0
        self._pending_truncated = False
        self._used_bytes = 0
        self._limit_notice_added = False
        self._seen_figures: weakref.WeakValueDictionary[int, Any] = (
            weakref.WeakValueDictionary()
        )
        self._seen_plotly: weakref.WeakValueDictionary[int, Any] = (
            weakref.WeakValueDictionary()
        )
        self._stdout = _Stdout(self)
        self._saved_stdout: Any = None
        self._saved_display: Any = _MISSING
        self._saved_globals: dict[str, Any] | None = None
        self._saved_import: Any = None
        self._patches: list[tuple[Any, str, Any]] = []
        self._patched: set[tuple[int, str]] = set()

    # -- public lifecycle -------------------------------------------------
    @contextmanager
    def activate(self, func: Callable[..., Any]) -> Iterator[OutputCapture]:
        """Redirect stdout/display and patch show() for the duration of *func*."""
        self._saved_stdout = sys.stdout
        sys.stdout = self._stdout
        globals_dict = getattr(func, "__globals__", None)
        if isinstance(globals_dict, dict):
            self._saved_globals = globals_dict
            self._saved_display = globals_dict.get("display", _MISSING)
            globals_dict["display"] = self.display
        self._patch_show_methods()
        self._install_import_hook()
        try:
            yield self
        finally:
            try:
                self.flush_text()
                self.capture_matplotlib_figures()
            finally:
                self._restore_import_hook()
                self._restore_show_methods()
                sys.stdout = self._saved_stdout
                if self._saved_globals is not None:
                    if self._saved_display is _MISSING:
                        self._saved_globals.pop("display", None)
                    else:
                        self._saved_globals["display"] = self._saved_display
                    self._saved_globals = None

    def display(self, *objects: Any, **kwargs: Any) -> None:
        """Jupyter-compatible display hook injected into the node namespace."""
        del kwargs
        self.flush_text()
        for obj in objects:
            self._add_object(obj)

    # -- stdout -----------------------------------------------------------
    def _write_stdout(self, text: str) -> None:
        if not text:
            return
        remaining = MAX_TEXT_CHARS - self._pending_chars
        if remaining > 0:
            self._pending.append(text[:remaining])
            self._pending_chars += min(len(text), remaining)
        if len(text) > max(remaining, 0):
            self._pending_truncated = True

    def flush_text(self) -> None:
        """Turn buffered stdout into a text output before another display."""
        if not self._pending:
            return
        text = "".join(self._pending)
        truncated = self._pending_truncated
        self._pending.clear()
        self._pending_chars = 0
        self._pending_truncated = False
        text, clipped = self._fit_text(text)
        if not text and not truncated:
            return
        self._append(
            TextOutput(
                source="print",
                text=text,
                truncated=truncated or clipped,
            )
        )

    # -- display protocol -------------------------------------------------
    def _add_object(self, obj: Any) -> None:
        try:
            if isinstance(obj, str):
                self._add_text(obj, "display")
                return
            if obj is None:
                self._add_text("None", "display")
                return
            table = self._as_table(obj)
            if table is not None:
                self._append(table)
                return
            if self._is_matplotlib_figure(obj):
                self._add_figure(obj, "matplotlib")
                return
            if self._is_pil_image(obj):
                self._add_pil_image(obj)
                return
            if self._is_plotly_figure(obj):
                self._add_plotly(obj, "plotly")
                return
            if self._add_mime_bundle(obj):
                return
            if self._add_repr_image(obj, "_repr_png_", "image/png"):
                return
            if self._add_repr_image(obj, "_repr_jpeg_", "image/jpeg"):
                return
            if self._add_repr_image(obj, "_repr_svg_", "image/svg+xml"):
                return
            html = self._call_repr(obj, "_repr_html_")
            if isinstance(html, str) and html.strip():
                self._add_html(html)
                return
            self._add_text(repr(obj), "display")
        except Exception:
            # A broken rich representation must not break the node.
            self._add_text(repr(obj), "display")

    def _add_mime_bundle(self, obj: Any) -> bool:
        method = getattr(obj, "_repr_mimebundle_", None)
        if not callable(method):
            return False
        bundle: Any = None
        try:
            bundle = method(include=None, exclude=None)
        except TypeError:
            try:
                bundle = method()
            except Exception:
                return False
        except Exception:
            return False
        if isinstance(bundle, tuple):
            bundle = bundle[0]
        if not isinstance(bundle, dict):
            return False
        plotly = bundle.get("application/vnd.plotly.v1+json")
        if isinstance(plotly, dict):
            self._add_plotly_payload(plotly, "plotly")
            return True
        for mime in ("image/png", "image/jpeg", "image/svg+xml"):
            value = bundle.get(mime)
            if isinstance(value, (bytes, str)):
                self._add_encoded_image(value, mime)
                return True
        html = bundle.get("text/html")
        if isinstance(html, str) and html.strip():
            self._add_html(html)
            return True
        plain = bundle.get("text/plain")
        if isinstance(plain, str) and plain.strip():
            self._add_text(plain, "display")
            return True
        return False

    def _add_repr_image(self, obj: Any, method_name: str, mime: str) -> bool:
        value = self._call_repr(obj, method_name)
        if not isinstance(value, (bytes, str)) or not value:
            return False
        self._add_encoded_image(value, mime)
        return True

    @staticmethod
    def _call_repr(obj: Any, method_name: str) -> Any:
        method = getattr(obj, method_name, None)
        if not callable(method):
            return None
        try:
            return method()
        except Exception:
            return None

    # -- tables -----------------------------------------------------------
    def _as_table(self, obj: Any) -> TableOutput | None:
        try:
            if all(hasattr(obj, name) for name in ("to_dicts", "height", "width")):
                total_rows = int(obj.height)
                total_cols = int(obj.width)
                columns = [str(name) for name in obj.columns][:MAX_TABLE_COLS]
                head = obj.head(MAX_TABLE_ROWS)
                raw_rows = head.to_dicts()[:MAX_TABLE_ROWS]
                rows = [
                    [self._cell(row.get(column)) for column in columns]
                    for row in raw_rows
                ]
            elif all(hasattr(obj, name) for name in ("to_dict", "shape", "columns")):
                shape = obj.shape
                total_rows = int(shape[0])
                total_cols = int(shape[1])
                head = obj.head(MAX_TABLE_ROWS)
                if hasattr(head, "iloc"):
                    head = head.iloc[:, :MAX_TABLE_COLS]
                original_columns = list(head.columns)
                columns = [str(name) for name in original_columns]
                records = head.to_dict("records")[:MAX_TABLE_ROWS]
                rows = [
                    [self._cell(record.get(name)) for name in original_columns]
                    for record in records
                ]
            else:
                return None
        except Exception:
            return None
        return TableOutput(
            columns=columns,
            rows=rows,
            total_rows=total_rows,
            total_cols=total_cols,
            truncated=total_rows > len(rows) or total_cols > len(columns),
        )

    @staticmethod
    def _cell(value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return repr(value)
            return value
        if isinstance(value, str):
            return value[:MAX_CELL_CHARS]
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                return OutputCapture._cell(value.item())
            except Exception:
                return repr(value)[:MAX_CELL_CHARS]
        text = repr(value)
        return text[:MAX_CELL_CHARS]

    # -- images -----------------------------------------------------------
    @staticmethod
    def _is_matplotlib_figure(obj: Any) -> bool:
        return type(obj).__module__.startswith("matplotlib.") and callable(
            getattr(obj, "savefig", None)
        )

    @staticmethod
    def _is_pil_image(obj: Any) -> bool:
        return type(obj).__module__.startswith("PIL.") and callable(
            getattr(obj, "save", None)
        )

    def _add_figure(self, figure: Any, source: str) -> bool:
        if id(figure) in self._seen_figures:
            return True
        self._seen_figures[id(figure)] = figure
        data: bytes | None = None
        for dpi in (100, 75, 50, 35):
            buffer = io.BytesIO()
            try:
                figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
            except Exception:
                return False
            candidate = buffer.getvalue()
            data = candidate
            if len(candidate) <= MAX_IMAGE_BYTES:
                break
        if not data or len(data) > MAX_IMAGE_BYTES:
            self._mark_limit()
            return False
        self._append(self._image_output(data, "image/png", source))
        return True

    def _add_pil_image(self, image: Any) -> bool:
        buffer = io.BytesIO()
        try:
            image.save(buffer, format="PNG")
        except Exception:
            return False
        return self._add_image_bytes(buffer.getvalue(), "image/png", "display")

    def _add_encoded_image(self, value: bytes | str, mime: str) -> bool:
        if isinstance(value, str) and mime == "image/svg+xml":
            data = value.encode("utf-8")
            encoded = base64.b64encode(data).decode("ascii")
        elif isinstance(value, str):
            raw = "".join(value.split())
            try:
                data = base64.b64decode(raw, validate=True)
                encoded = raw
            except Exception:
                data = raw.encode("utf-8")
                encoded = base64.b64encode(data).decode("ascii")
        else:
            data = value
            encoded = base64.b64encode(value).decode("ascii")
        if len(data) > MAX_IMAGE_BYTES:
            self._mark_limit()
            return False
        self._append(
            ImageOutput(
                data=encoded,
                mime=mime,
                source="display",
                alt="Plot output" if mime.startswith("image/") else "Image output",
            )
        )
        return True

    def _add_image_bytes(self, data: bytes, mime: str, source: str) -> bool:
        if len(data) > MAX_IMAGE_BYTES:
            self._mark_limit()
            return False
        self._append(self._image_output(data, mime, source))
        return True

    @staticmethod
    def _image_output(data: bytes, mime: str, source: str) -> ImageOutput:
        return ImageOutput(
            mime=mime,
            data=base64.b64encode(data).decode("ascii"),
            source=source,
            alt="Plot output",
        )

    def capture_matplotlib_figures(self) -> None:
        """Snapshot every open Pyplot figure and close it afterwards."""
        self.flush_text()
        pyplot = sys.modules.get("matplotlib.pyplot")
        if pyplot is None:
            return
        try:
            for number in list(pyplot.get_fignums()):
                try:
                    figure = pyplot.figure(number)
                except Exception:
                    continue
                self._add_figure(figure, "matplotlib")
        finally:
            try:
                pyplot.close("all")
            except Exception:
                pass

    # -- plotly -----------------------------------------------------------
    @staticmethod
    def _is_plotly_figure(obj: Any) -> bool:
        module = type(obj).__module__
        return module.startswith("plotly.") and callable(getattr(obj, "to_json", None))

    def _add_plotly(self, figure: Any, source: str) -> bool:
        if id(figure) in self._seen_plotly:
            return True
        self._seen_plotly[id(figure)] = figure
        try:
            payload = json.loads(figure.to_json())
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        return self._add_plotly_payload(payload, source)

    def _add_plotly_payload(self, payload: dict[str, Any], source: str) -> bool:
        try:
            safe_payload = json.loads(
                json.dumps(payload, default=str),
                parse_constant=lambda _value: None,
            )
        except Exception:
            return False
        if not isinstance(safe_payload, dict) or not isinstance(
            safe_payload.get("data"), list
        ):
            return False
        encoded = json.dumps(safe_payload)
        if len(encoded.encode("utf-8")) > MAX_PLOTLY_BYTES:
            self._mark_limit()
            return False
        self._append(PlotlyOutput(figure=safe_payload, source=source))
        return True

    # -- show patches -----------------------------------------------------
    def _install_import_hook(self) -> None:
        """Patch plotting modules that the node imports after activation."""
        self._saved_import = builtins.__import__

        def import_hook(
            name: str,
            globals: dict[str, Any] | None = None,
            locals: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            module = self._saved_import(name, globals, locals, fromlist, level)
            if name.startswith(("plotly", "matplotlib")):
                self._patch_show_methods()
            return module

        builtins.__import__ = import_hook

    def _restore_import_hook(self) -> None:
        if self._saved_import is not None:
            builtins.__import__ = self._saved_import
            self._saved_import = None

    def _patch_show_methods(self) -> None:
        pyplot = sys.modules.get("matplotlib.pyplot")
        if pyplot is not None and callable(getattr(pyplot, "show", None)):
            self._patch(pyplot, "show", lambda *args, **kwargs: self.capture_matplotlib_figures())

        basedatatypes = sys.modules.get("plotly.basedatatypes")
        figure_class = getattr(basedatatypes, "BaseFigure", None)
        if figure_class is not None and callable(getattr(figure_class, "show", None)):
            self._patch(
                figure_class,
                "show",
                lambda figure, *args, **kwargs: self._capture_plotly_show(figure),
            )

        plotly_io = sys.modules.get("plotly.io")
        renderers = getattr(plotly_io, "renderers", None)
        if renderers is not None and callable(getattr(renderers, "show", None)):
            original_show = renderers.show

            def renderer_show(figure: Any, *args: Any, **kwargs: Any) -> Any:
                if self._is_plotly_figure(figure):
                    self._capture_plotly_show(figure)
                    return None
                return original_show(figure, *args, **kwargs)

            self._patch(renderers, "show", renderer_show)

    def _capture_plotly_show(self, figure: Any) -> None:
        self.flush_text()
        self._add_plotly(figure, "plotly")

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        key = (id(owner), name)
        if key in self._patched:
            return
        try:
            original = getattr(owner, name)
            setattr(owner, name, replacement)
        except Exception:
            return
        self._patched.add(key)
        self._patches.append((owner, name, original))

    def _restore_show_methods(self) -> None:
        for owner, name, original in reversed(self._patches):
            try:
                setattr(owner, name, original)
            except Exception:
                pass
        self._patches.clear()
        self._patched.clear()

    # -- storage helpers --------------------------------------------------
    def _add_text(self, text: str, source: str) -> None:
        text, clipped = self._fit_text(text)
        if not text and not clipped:
            return
        self._append(TextOutput(source=source, text=text, truncated=clipped))

    def _add_html(self, html: str) -> None:
        clipped, truncated = self._clip_text(html, MAX_HTML_CHARS)
        self._append(HtmlOutput(html=clipped, truncated=truncated))

    @staticmethod
    def _clip_text(text: str, limit: int) -> tuple[str, bool]:
        if len(text) <= limit:
            return text, False
        return text[:limit], True

    def _fit_text(self, text: str) -> tuple[str, bool]:
        char_limit = min(MAX_TEXT_CHARS, max(MAX_NODE_OUTPUT_BYTES - self._used_bytes, 0))
        if len(text) <= char_limit:
            return text, False
        if char_limit <= 0:
            self._mark_limit()
            return "", True
        return text[:char_limit], True

    def _append(self, output: RichOutput) -> None:
        cost = rich_output_size(output)
        if len(self.outputs) >= MAX_OUTPUTS or self._used_bytes + cost > MAX_NODE_OUTPUT_BYTES:
            self._mark_limit()
            return
        self.outputs.append(output)
        self._used_bytes += cost

    def _mark_limit(self) -> None:
        if self._limit_notice_added:
            return
        self._limit_notice_added = True
        self.outputs.append(TextOutput(source="system", text=_TRUNCATION_NOTICE, truncated=True))


def rich_output_size(output: RichOutput) -> int:
    """Estimate the wire size of one rich output."""
    try:
        payload: Sequence[Any] = [output.model_dump(mode="json")]
        return len(json.dumps(payload, default=str).encode("utf-8"))
    except Exception:
        return MAX_NODE_OUTPUT_BYTES
