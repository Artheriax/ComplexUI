"""Console logging for the ComplexUI backend.

Call :func:`configure_logging` once at startup; modules then use
``logging.getLogger("complexui.<area>")`` so every step of a parse, approval,
install or execution is narrated on the console.
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["configure_logging"]

_DIM = "\x1b[2m"
_RESET = "\x1b[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\x1b[90m",
    logging.INFO: "\x1b[36m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[1;31m",
}


class ConsoleFormatter(logging.Formatter):
    """Compact ``HH:MM:SS LEVEL logger message`` formatter with ANSI colors."""

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        level = record.levelname.ljust(7)
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if not self._color:
            return f"{timestamp} {level} {record.name} {message}"
        level_color = _LEVEL_COLORS.get(record.levelno, "")
        return (
            f"{_DIM}{timestamp}{_RESET} "
            f"{level_color}{level}{_RESET} "
            f"{_DIM}{record.name}{_RESET} {message}"
        )


def _should_color(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stderr.isatty()


def configure_logging(
    level: str | int = "INFO", *, color: bool | None = None
) -> logging.Logger:
    """Configure the ``complexui`` logger tree (idempotent).

    Repeated calls update the level without stacking handlers.
    """
    logger = logging.getLogger("complexui")
    handler_configured = getattr(logger, "_complexui_configured", False)
    if str(level).isdigit():
        logger.setLevel(int(level))
    else:
        logger.setLevel(str(level).upper())
    if not handler_configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ConsoleFormatter(color=_should_color(color)))
        logger.addHandler(handler)
        logger.propagate = False
        logger._complexui_configured = True  # type: ignore[attr-defined]
    return logger
