"""Tests for the console logging setup."""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest

from app.logging_setup import (
    ConsoleFormatter,
    JsonFormatter,
    _should_color,
    configure_logging,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    configure_logging("INFO", color=False, log_format="text")


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


def test_configure_logging_accepts_aliases_and_numeric_levels() -> None:
    logger = configure_logging("warn", color=False, stream=io.StringIO())

    assert logger.level == logging.WARNING

    configure_logging("10", color=False, stream=io.StringIO())

    assert logger.level == logging.DEBUG


def test_configure_logging_switches_between_json_and_text() -> None:
    stream = io.StringIO()
    logger = configure_logging(
        "DEBUG", color=False, log_format="json", stream=stream
    )

    logger.info("structured", extra={"run_id": "run-1", "node_id": "n1"})
    logger.handlers[0].flush()

    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "INFO"
    assert payload["message"] == "structured"
    assert payload["run_id"] == "run-1"
    assert payload["node_id"] == "n1"
    assert payload["timestamp"].endswith("Z")

    configure_logging("INFO", color=False, log_format="text", stream=stream)
    logger.debug("hidden")
    logger.info("plain")
    logger.handlers[0].flush()

    lines = stream.getvalue().splitlines()
    assert lines[-1].endswith("plain")
    assert "hidden" not in lines[-1]


def test_configure_logging_rejects_unknown_level_and_format() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("VERBOSE")

    with pytest.raises(ValueError, match="Unknown log format"):
        configure_logging("INFO", log_format="xml")


def test_json_formatter_emits_structured_fields() -> None:
    record = _record(logging.WARNING, "careful")
    record.run_id = "run-1"
    record.node_id = "n1"

    payload = json.loads(JsonFormatter().format(record))

    assert payload == {
        "timestamp": payload["timestamp"],
        "level": "WARNING",
        "logger": "complexui.test",
        "message": "careful",
        "run_id": "run-1",
        "node_id": "n1",
    }


def test_json_formatter_includes_exception_text() -> None:
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
            sys.exc_info(),
        )

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_execution_events_log_structured_context() -> None:
    from app.main import _log_event
    from app.schemas import EventType, ExecutionEvent

    stream = io.StringIO()
    logger = configure_logging("DEBUG", log_format="json", stream=stream)

    _log_event(
        ExecutionEvent(
            type=EventType.NODE_ERROR,
            node_id="n1",
            data={
                "run_id": "run-1",
                "error_type": "ValueError",
                "line": 3,
                "message": "boom",
            },
        )
    )
    logger.handlers[0].flush()

    payload = json.loads(stream.getvalue().splitlines()[-1])
    assert payload["level"] == "ERROR"
    assert payload["run_id"] == "run-1"
    assert payload["node_id"] == "n1"
    assert payload["error_type"] == "ValueError"
    assert payload["line"] == 3
    assert payload["message"] == "  [error] n1: ValueError (line 3): boom"


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


def test_should_color_respects_force_color_disable(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "0")

    assert _should_color(None, io.StringIO()) is False


def test_should_color_respects_empty_no_color_and_dumb_term(monkeypatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "")
    assert _should_color(None, io.StringIO()) is False

    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert _should_color(None, io.StringIO()) is False
