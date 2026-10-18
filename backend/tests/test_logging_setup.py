"""Tests for the console logging setup."""

from __future__ import annotations

import logging

from app.logging_setup import ConsoleFormatter, _should_color, configure_logging


def _record(
    level: int = logging.INFO, message: str = "hello", name: str = "complexui.test"
) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, message, None, None)


def test_formatter_plain_output() -> None:
    formatter = ConsoleFormatter(color=False)

    output = formatter.format(_record(logging.WARNING, "careful"))

    assert "WARNING" in output
    assert "complexui.test" in output
    assert "careful" in output
    assert "\x1b[" not in output


def test_formatter_color_output() -> None:
    formatter = ConsoleFormatter(color=True)

    output = formatter.format(_record())

    assert "\x1b[" in output
    assert "hello" in output


def test_formatter_includes_exception_text() -> None:
    formatter = ConsoleFormatter(color=False)
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "complexui.test",
            logging.ERROR,
            __file__,
            1,
            "failed",
            None,
            __import__("sys").exc_info(),
        )
    output = formatter.format(record)
    assert "ValueError: boom" in output


def test_configure_logging_is_idempotent_and_updates_level() -> None:
    logger = configure_logging("INFO", color=False)

    def console_handlers() -> list[logging.Handler]:
        return [
            handler
            for handler in logger.handlers
            if isinstance(handler.formatter, ConsoleFormatter)
        ]

    assert len(console_handlers()) == 1

    again = configure_logging("DEBUG", color=False)

    assert again is logger
    assert len(console_handlers()) == 1
    assert logger.level == logging.DEBUG


def test_should_color_prefers_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert _should_color(True) is True
    assert _should_color(False) is False


def test_should_color_respects_environment(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _should_color(None) is True

    monkeypatch.delenv("FORCE_COLOR")
    monkeypatch.setenv("NO_COLOR", "1")
    assert _should_color(None) is False
