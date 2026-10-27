"""Small dependency-free helpers for durable local file writes.

Workspace projects, run records and user templates all need the same safety
property: a crash or full disk must never leave a half-written JSON document or
Python module behind. Writes go to a sibling temporary file and are moved into
place with :func:`os.replace`, which is atomic on the same filesystem.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

__all__ = ["normalize_code", "utc_timestamp", "write_json_atomic", "write_text_atomic"]

_REPLACE_ATTEMPTS = 5


def _replace_atomically(source: Path, target: Path) -> None:
    """Move ``source`` onto ``target``, retrying Windows sharing violations."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            # Windows refuses to replace a file another thread/process has open
            # without FILE_SHARE_DELETE; a short retry usually clears the race.
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.01 * (attempt + 1))


def normalize_code(code: str) -> str:
    """Normalize line endings so hashes and stored code are platform independent."""
    return code.replace("\r\n", "\n").replace("\r", "\n")


def utc_timestamp(*, milliseconds: bool = False) -> str:
    """Return the current UTC time as an ISO-8601 ``...Z`` string."""
    timespec = "milliseconds" if milliseconds else "seconds"
    return datetime.now(UTC).isoformat(timespec=timespec).replace("+00:00", "Z")


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text atomically, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        _replace_atomically(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: object) -> None:
    """Serialize ``payload`` as pretty UTF-8 JSON and write it atomically."""
    write_text_atomic(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
