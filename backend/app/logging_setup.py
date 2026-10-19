"""Console and structured logging for the ComplexUI backend.

Call :func:`configure_logging` once at startup; modules then use
``logging.getLogger("complexui.<area>")`` so every step of a parse, approval,
install or execution is narrated on the console.

Set ``COMPLEXUI_LOG_FORMAT=json`` for newline-delimited JSON application logs
(one object per line) suitable for shipping to a log collector.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import IO, Any

__all__ = [
    "ConsoleFormatter",
    "JsonFormatter",
    "LogFormat",
    "configure_logging",
]

_DIM = "\x1b[2m"
_RESET = "\x1b[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\x1b[90m",
    logging.INFO: "\x1b[36m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[1;31m",
}
_LEVEL_WIDTH = 8
_HANDLER_MARKER = "_complexui_console_handler"
_FORMAT_ENV = "COMPLEXUI_LOG_FORMAT"
_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_RESERVED_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


class LogFormat(StrEnum):
    """Supported application log output formats."""

    TEXT = "text"
    JSON = "json"


class ConsoleFormatter(logging.Formatter):
    """Compact ``HH:MM:SS.mmm LEVEL logger message`` output with ANSI colors."""

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = f"{self.formatTime(record, '%H:%M:%S')}.{int(record.msecs):03d}"
        level = record.levelname.ljust(_LEVEL_WIDTH)
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"
        if not self._color:
            return f"{timestamp} {level} {record.name} {message}"
        level_color = _LEVEL_COLORS.get(record.levelno, "")
        return (
            f"{_DIM}{timestamp}{_RESET} "
            f"{level_color}{level}{_RESET} "
            f"{_DIM}{record.name}{_RESET} {message}"
        )


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object, including ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _iso_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )


def _iso_timestamp(created: float) -> str:
    return (
        datetime.fromtimestamp(created, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _should_color(
    explicit: bool | None, stream: IO[str] | None = None
) -> bool:
    if explicit is not None:
        return explicit
    if "NO_COLOR" in os.environ:
        return False
    force_color = os.environ.get("FORCE_COLOR")
    if force_color is not None:
        return force_color.strip().lower() not in _FALSE_VALUES
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    target = stream if stream is not None else sys.stderr
    return bool(getattr(target, "isatty", lambda: False)())


def _coerce_level(level: str | int) -> int:
    if isinstance(level, bool):
        raise ValueError("Log level must be a name or number, not a boolean")
    if isinstance(level, int):
        if level < 0:
            raise ValueError(f"Log level must be non-negative, got {level}")
        return level
    text = level.strip().upper()
    if text.isdigit():
        return int(text)
    name = _LEVEL_ALIASES.get(text, text)
    levels = logging.getLevelNamesMapping()
    if name not in levels:
        raise ValueError(
            f"Unknown log level {level!r}; use DEBUG, INFO, WARNING, ERROR, "
            "CRITICAL or a numeric level"
        )
    return levels[name]


def _coerce_format(log_format: str | LogFormat | None) -> LogFormat:
    raw = (
        log_format
        if log_format is not None
        else os.environ.get(_FORMAT_ENV, LogFormat.TEXT.value)
    )
    try:
        return LogFormat(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"Unknown log format {raw!r}; use 'text' or 'json'"
        ) from exc


def configure_logging(
    level: str | int = "INFO",
    *,
    color: bool | None = None,
    log_format: str | LogFormat | None = None,
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Configure the ``complexui`` logger tree and return it.

    Repeated calls update the level and replace the managed handler, so
    callers can switch between text/JSON output or targets without stacking
    handlers. The format defaults to ``COMPLEXUI_LOG_FORMAT`` (``text`` or
    ``json``).
    """
    logger = logging.getLogger("complexui")
    logger.setLevel(_coerce_level(level))
    selected_format = _coerce_format(log_format)
    target = stream if stream is not None else sys.stderr
    formatter: logging.Formatter
    if selected_format is LogFormat.JSON:
        formatter = JsonFormatter()
    else:
        formatter = ConsoleFormatter(color=_should_color(color, target))

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(target)
    handler.setFormatter(formatter)
    setattr(handler, _HANDLER_MARKER, True)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
