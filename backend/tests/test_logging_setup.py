"""Tests for the console logging setup."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

from app.logging_setup import (
    ConsoleFormatter,
    DailyTextFileHandler,
    FileFormatter,
    JsonFormatter,
    PollingAccessFilter,
    _should_color,
    configure_logging,
    install_access_log_filter,
    prune_run_logs,
    resolve_log_dir,
    run_log_path,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    configure_logging("INFO", color=False, log_format="text")


def _record(
    level: int = logging.INFO, message: str = "hello", name: str = "complexui.test"
) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, message, None, None)


def test_polling_access_filter_drops_successful_polls_only() -> None:
    access_filter = PollingAccessFilter()
    record = _record(name="uvicorn.access")
    record.args = ("127.0.0.1", "GET", "/api/system/metrics", "HTTP/1.1", 200)
    assert not access_filter.filter(record)

    record.args = ("127.0.0.1", "GET", "/api/system/metrics", "HTTP/1.1", 500)
    assert access_filter.filter(record)

    record.args = ("127.0.0.1", "POST", "/api/system/metrics", "HTTP/1.1", 200)
    assert access_filter.filter(record)

    record.args = ("127.0.0.1", "GET", "/api/execute", "HTTP/1.1", 200)
    assert access_filter.filter(record)


def test_install_access_log_filter_is_idempotent() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    install_access_log_filter()
    install_access_log_filter()
    try:
        installed = [
            item for item in access_logger.filters if isinstance(item, PollingAccessFilter)
        ]
        assert len(installed) == 1
    finally:
        for item in list(access_logger.filters):
            if isinstance(item, PollingAccessFilter):
                access_logger.removeFilter(item)


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


def test_file_formatter_is_plain_and_timestamped() -> None:
    output = FileFormatter().format(_record(logging.INFO, "persisted"))

    assert "persisted" in output
    assert "complexui.test" in output
    assert "\x1b[" not in output
    assert output[:4].isdigit()


def test_configure_logging_writes_daily_text_file(tmp_path: Path) -> None:
    logger = configure_logging(
        "DEBUG", color=False, log_dir=tmp_path, retention_days=3
    )

    logger.info("persisted message", extra={"run_id": "run-1"})
    handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, DailyTextFileHandler)
    ]
    assert len(handlers) == 1
    handler = handlers[0]
    assert handler.current_path is not None
    handler.flush()

    files = list(tmp_path.glob("complexui-*.txt"))
    assert files == [handler.current_path]
    content = files[0].read_text(encoding="utf-8")
    assert "persisted message" in content
    assert "run-1" not in content
    assert "\x1b[" not in content


def test_configure_logging_replaces_managed_file_handler(tmp_path: Path) -> None:
    logger = configure_logging("INFO", color=False, log_dir=tmp_path)

    configure_logging(
        "INFO", color=False, log_dir=tmp_path / "nested", retention_days=7
    )

    file_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, DailyTextFileHandler)
    ]
    assert len(file_handlers) == 1
    assert file_handlers[0].directory == tmp_path / "nested"


def test_daily_file_handler_prunes_old_logs(tmp_path: Path) -> None:
    old = tmp_path / "complexui-2000-01-01.txt"
    current = tmp_path / "complexui-2099-01-01.txt"
    old.write_text("old", encoding="utf-8")
    current.write_text("current", encoding="utf-8")
    old_time = time.time() - 3 * 86_400
    old.touch()
    os.utime(old, (old_time, old_time))

    DailyTextFileHandler(tmp_path, retention_days=1)

    assert not old.exists()
    assert current.exists()


def test_daily_file_handler_rejects_invalid_retention(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retention_days"):
        DailyTextFileHandler(tmp_path, retention_days=0)


def test_resolve_log_dir_honors_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COMPLEXUI_LOG_DIR", str(tmp_path))
    assert resolve_log_dir() == tmp_path

    monkeypatch.setenv("COMPLEXUI_LOG_DIR", "off")
    assert resolve_log_dir() is None

    monkeypatch.delenv("COMPLEXUI_LOG_DIR")
    assert resolve_log_dir() is not None


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
    from app.logging_setup import log_execution_event
    from app.schemas import EventType, ExecutionEvent

    stream = io.StringIO()
    logger = configure_logging("DEBUG", log_format="json", stream=stream)

    log_execution_event(
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


def test_per_run_log_captures_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.logging_setup import log_execution_event
    from app.main import _close_run_log, _open_run_log
    from app.schemas import EventType, ExecutionEvent

    monkeypatch.setenv("COMPLEXUI_LOG_DIR", str(tmp_path))
    run_logger, handler = _open_run_log("run-test")
    assert run_logger is not None
    assert handler is not None
    log_execution_event(
        ExecutionEvent(
            type=EventType.EXECUTION_STARTED,
            node_id="",
            data={"run_id": "run-test", "order": ["a", "b"]},
        ),
        target=run_logger,
    )
    log_execution_event(
        ExecutionEvent(
            type=EventType.EXECUTION_COMPLETED,
            node_id="",
            data={"run_id": "run-test", "duration_ms": 12.5, "cached": {}},
        ),
        target=run_logger,
    )
    handler.flush()
    path = tmp_path / "runs" / "run-test.txt"
    content = path.read_text(encoding="utf-8")
    _close_run_log(run_logger, handler)

    assert "run run-test started: 2 node(s)" in content
    assert "run run-test completed in 12.5ms" in content
    assert run_logger.handlers == []


def test_open_run_log_counts_current_file_against_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.main import _close_run_log, _open_run_log

    monkeypatch.setenv("COMPLEXUI_LOG_DIR", str(tmp_path))
    runs = tmp_path / "runs"
    runs.mkdir()
    for index in range(55):
        (runs / f"old-{index}.txt").write_text(str(index), encoding="utf-8")

    run_logger, handler = _open_run_log("new-run")
    try:
        assert run_logger is not None
        assert handler is not None
        assert (runs / "new-run.txt").is_file()
        assert len(list(runs.glob("*.txt"))) == 50
    finally:
        _close_run_log(run_logger, handler)


def test_prune_run_logs_keeps_newest_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    for index in range(4):
        path = runs / f"run-{index}.txt"
        path.write_text(str(index), encoding="utf-8")
        stat = path.stat()
        os.utime(path, (stat.st_atime, stat.st_mtime + index))

    prune_run_logs(tmp_path, keep=2)

    assert sorted(path.name for path in runs.glob("*.txt")) == [
        "run-2.txt",
        "run-3.txt",
    ]


def test_prune_run_logs_rejects_negative_retention(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep"):
        prune_run_logs(tmp_path, keep=-1)


def test_run_log_path_stays_under_run_directory(tmp_path: Path) -> None:
    assert run_log_path(tmp_path, "abc").parent == tmp_path / "runs"


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
