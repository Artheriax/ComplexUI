"""Tests for the shared durable-write helpers."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from app.storage import normalize_code, utc_timestamp, write_json_atomic, write_text_atomic


def test_utc_timestamp_is_utc_iso8601() -> None:
    plain = utc_timestamp()
    precise = utc_timestamp(milliseconds=True)

    assert plain.endswith("Z")
    assert precise.endswith("Z")
    assert "." not in plain
    assert "." in precise


def test_normalize_code_rewrites_windows_line_endings() -> None:
    assert normalize_code("a\r\nb\rc\n") == "a\nb\nc\n"


def test_write_text_atomic_replaces_contents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "note.txt"
    target.parent.mkdir()
    target.write_text("old", encoding="utf-8")

    write_text_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert list(target.parent.iterdir()) == [target]


def test_write_json_atomic_round_trips_unicode(tmp_path: Path) -> None:
    target = tmp_path / "record.json"

    write_json_atomic(target, {"title": "café", "values": [1, 2]})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "title": "café",
        "values": [1, 2],
    }
    assert target.read_text(encoding="utf-8").endswith("\n")


def test_write_text_atomic_cleans_up_after_failure(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "record.json"

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("app.storage.os.replace", fail_replace)

    try:
        write_text_atomic(target, "payload")
    except OSError:
        pass
    else:  # pragma: no cover - the helper must propagate write failures
        raise AssertionError("expected write_text_atomic to propagate the failure")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_text_atomic_keeps_concurrent_writes_whole(tmp_path: Path) -> None:
    target = tmp_path / "shared.txt"
    payloads = [f"payload-{index}" * 50 for index in range(8)]

    threads = [
        threading.Thread(target=write_text_atomic, args=(target, payload))
        for payload in payloads
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert target.read_text(encoding="utf-8") in payloads
