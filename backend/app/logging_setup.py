"""Console and structured logging for the ComplexUI backend.

Call :func:`configure_logging` once at startup; modules then use
``logging.getLogger("complexui.<area>")`` so every step of a parse, approval,
install or execution is narrated on the console. Daily plain-text logs are
written under ``backend/workspace/logs`` by default; set ``COMPLEXUI_LOG_DIR``
to relocate them or to ``0`` to disable file logging.

Set ``COMPLEXUI_LOG_FORMAT=json`` for newline-delimited JSON console logs (one
object per line) suitable for shipping to a log collector. Files stay in the
human-readable text format.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

from .workspace import workspace_root

__all__ = [
    "ConsoleFormatter",
    "DailyTextFileHandler",
    "DEFAULT_LOG_DIR",
    "FileFormatter",
    "JsonFormatter",
    "LogFormat",
    "PollingAccessFilter",
    "configure_logging",
    "install_access_log_filter",
    "resolve_log_dir",
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
_FILE_HANDLER_MARKER = "_complexui_file_handler"
_FORMAT_ENV = "COMPLEXUI_LOG_FORMAT"
_LOG_DIR_ENV = "COMPLEXUI_LOG_DIR"
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
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "workspace" / "logs"
_POLLING_ACCESS_PATHS = frozenset(
    {"/api/system/metrics", "/api/status", "/api/health"}
)


class PollingAccessFilter(logging.Filter):
    """Drop successful access-log records for high-frequency polling routes."""

    def __init__(self, paths: tuple[str, ...] = tuple(_POLLING_ACCESS_PATHS)) -> None:
        super().__init__()
        self.paths = frozenset(paths)

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        method, path, status = args[1], args[2], args[4]
        if (
            not isinstance(method, str)
            or not isinstance(path, str)
            or not isinstance(status, (int, str))
        ):
            return True
        try:
            status_code = int(status)
        except (TypeError, ValueError):
            return True
        return not (
            method.upper() == "GET"
            and status_code < 400
            and path.split("?", 1)[0] in self.paths
        )


def install_access_log_filter() -> logging.Logger:
    """Install the polling access filter on Uvicorn's access logger."""
    access_logger = logging.getLogger("uvicorn.access")
    for existing in list(access_logger.filters):
        if isinstance(existing, PollingAccessFilter):
            access_logger.removeFilter(existing)
    access_logger.addFilter(PollingAccessFilter())
    return access_logger


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


class FileFormatter(logging.Formatter):
    """Plain timestamped output for persistent ``.txt`` logs (never ANSI)."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}."
            f"{int(record.msecs):03d}"
        )
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"
        return (
            f"{timestamp} {record.levelname.ljust(_LEVEL_WIDTH)} "
            f"{record.name} {message}"
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


class DailyTextFileHandler(logging.Handler):
    """Append to ``<prefix>-YYYY-MM-DD.txt`` and prune old daily files."""

    def __init__(
        self,
        directory: str | Path,
        *,
        prefix: str = "complexui",
        retention_days: int = 14,
    ) -> None:
        super().__init__()
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        self._directory = Path(directory).expanduser()
        self._prefix = prefix
        self._retention_days = retention_days
        self._stream: IO[str] | None = None
        self._day: str | None = None
        self._directory.mkdir(parents=True, exist_ok=True)
        self._prune()

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def current_path(self) -> Path | None:
        if self._day is None:
            return None
        return self._directory / f"{self._prefix}-{self._day}.txt"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream()
            stream = self._stream
            if stream is None:
                return
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._close_stream()
        super().close()

    def _ensure_stream(self) -> None:
        day = datetime.now().astimezone().date().isoformat()
        if self._stream is not None and self._day == day:
            return
        self._close_stream()
        path = self._directory / f"{self._prefix}-{day}.txt"
        self._stream = path.open("a", encoding="utf-8")
        self._day = day
        self._prune()

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _prune(self) -> None:
        cutoff = time.time() - self._retention_days * 86_400
        for path in self._directory.glob(f"{self._prefix}-*.txt"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                # A log file locked by another process must not break logging.
                continue


def resolve_log_dir() -> Path | None:
    """Return the configured log directory, or ``None`` when disabled.

    Logs follow ``COMPLEXUI_WORKSPACE`` unless ``COMPLEXUI_LOG_DIR`` overrides
    them explicitly.
    """
    raw = os.environ.get(_LOG_DIR_ENV)
    if raw is None:
        return workspace_root() / "logs"
    value = raw.strip()
    if not value or value.lower() in _FALSE_VALUES:
        return None
    return Path(value).expanduser()


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
    log_dir: str | Path | None = None,
    retention_days: int = 14,
    file_prefix: str = "complexui",
) -> logging.Logger:
    """Configure the ``complexui`` logger tree and return it.

    Repeated calls update the level and replace managed console/file handlers,
    so callers can switch between text/JSON output or targets without stacking
    handlers. The console format defaults to ``COMPLEXUI_LOG_FORMAT`` (``text``
    or ``json``). Passing ``log_dir`` additionally writes daily plain-text files
    named ``<file_prefix>-YYYY-MM-DD.txt`` and prunes files older than
    ``retention_days``.
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
        if getattr(handler, _HANDLER_MARKER, False) or getattr(
            handler, _FILE_HANDLER_MARKER, False
        ):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(target)
    handler.setFormatter(formatter)
    setattr(handler, _HANDLER_MARKER, True)
    logger.addHandler(handler)
    logger.propagate = False

    if log_dir is not None:
        try:
            file_handler = DailyTextFileHandler(
                log_dir,
                prefix=file_prefix,
                retention_days=retention_days,
            )
        except (OSError, ValueError) as exc:
            logger.warning("file logging disabled: %s", exc)
        else:
            file_handler.setFormatter(FileFormatter())
            setattr(file_handler, _FILE_HANDLER_MARKER, True)
            logger.addHandler(file_handler)

    return logger
