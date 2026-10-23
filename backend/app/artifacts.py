"""Allowlisted media paths produced by completed runs.

Save Media can write outside the outputs directory. The browser may only
preview, copy, open or reveal those files when a completed run recorded the
path, so the API never becomes an arbitrary filesystem reader or opener.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from .datasets import IMAGE_FORMATS, VIDEO_FORMATS
from .run_history import get_run_history_store

__all__ = [
    "MAX_TRACKED_ARTIFACTS",
    "OutputArtifactRegistry",
    "get_output_artifact_registry",
]

MAX_TRACKED_ARTIFACTS = 512
_MEDIA_SUFFIXES = {*IMAGE_FORMATS, *VIDEO_FORMATS}


def _resolved_media_path(value: object) -> Path | None:
    if isinstance(value, Path):
        path = value.expanduser()
    elif isinstance(value, str) and value.strip():
        path = Path(value.strip()).expanduser()
    else:
        return None
    if not path.is_absolute():
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_file() or resolved.suffix.lower() not in _MEDIA_SUFFIXES:
        return None
    return resolved


def _artifact_key(path: Path) -> str:
    return os.path.normcase(str(path))


class OutputArtifactRegistry:
    """Bounded run-produced artifact paths with retained-history fallback."""

    def __init__(self, limit: int = MAX_TRACKED_ARTIFACTS) -> None:
        self._limit = max(1, limit)
        self._paths: OrderedDict[str, Path] = OrderedDict()
        self._lock = threading.Lock()

    def register(self, value: object) -> None:
        resolved = _resolved_media_path(value)
        if resolved is None:
            return
        key = _artifact_key(resolved)
        with self._lock:
            self._paths[key] = resolved
            self._paths.move_to_end(key)
            while len(self._paths) > self._limit:
                self._paths.popitem(last=False)

    def resolve(self, value: str) -> Path | None:
        resolved = _resolved_media_path(value)
        if resolved is None:
            return None
        key = _artifact_key(resolved)
        with self._lock:
            cached = self._paths.get(key)
            if cached is not None:
                self._paths.move_to_end(key)
                return cached
        for recorded in get_run_history_store().saved_paths():
            candidate = _resolved_media_path(recorded)
            if candidate is not None and _artifact_key(candidate) == key:
                self.register(candidate)
                return candidate
        return None


@lru_cache(maxsize=1)
def get_output_artifact_registry() -> OutputArtifactRegistry:
    """Return the process-wide run artifact registry."""
    return OutputArtifactRegistry()
