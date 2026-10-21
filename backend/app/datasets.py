"""Workspace-backed dataset uploads with bounded table previews.

Uploaded files live under ``<workspace>/data/uploads`` and are referenced from
``File`` widgets by their workspace-relative path. File names are sanitized,
collisions get a numeric suffix, and every write is streamed through a size cap
before being atomically moved into place.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import polars as pl

from .schemas import (
    DEFAULT_DATASET_PREVIEW_ROWS,
    MAX_DATASET_BYTES,
    MAX_DATASET_NAME_CHARS,
    MAX_DATASET_PREVIEW_ROWS,
    DatasetPreview,
    DatasetSummary,
)
from .workspace import workspace_root

__all__ = [
    "DatasetError",
    "DatasetNotFound",
    "DatasetPreviewError",
    "DatasetStore",
    "DatasetTooLarge",
    "InvalidDataset",
    "MAX_UPLOAD_REQUEST_BYTES",
    "SUPPORTED_DATASET_SUFFIXES",
    "get_dataset_store",
]

FORMAT_BY_SUFFIX: dict[str, str] = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".parquet": "Parquet",
    ".xlsx": "Excel",
    ".xlsm": "Excel",
    ".ndjson": "NDJSON",
    ".jsonl": "NDJSON",
}
SUPPORTED_DATASET_SUFFIXES = tuple(FORMAT_BY_SUFFIX)
MAX_UPLOAD_REQUEST_BYTES = MAX_DATASET_BYTES + (2 * 1024 * 1024)
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_RELATIVE_UPLOAD_DIR = Path("data") / "uploads"
_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class DatasetError(Exception):
    """Base class for dataset storage and preview failures."""


class DatasetNotFound(DatasetError):
    """The requested dataset does not exist."""


class InvalidDataset(DatasetError):
    """An upload name, extension, or contents is not supported."""


class DatasetTooLarge(DatasetError):
    """The uploaded file exceeds the workspace dataset size cap."""


class DatasetPreviewError(DatasetError):
    """A stored dataset could not be parsed for preview."""


def _safe_name(raw: str) -> tuple[str, str]:
    name = Path(raw.replace("\\", "/")).name.strip().strip(".")
    cleaned = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name).strip(" .")
    if not cleaned:
        cleaned = "dataset.csv"
    suffix = Path(cleaned).suffix.lower()
    if suffix not in FORMAT_BY_SUFFIX:
        supported = ", ".join(SUPPORTED_DATASET_SUFFIXES)
        raise InvalidDataset(
            f"Unsupported dataset type '{suffix or '<none>'}'; use {supported}"
        )
    stem = cleaned[: -len(suffix)].strip(" .") or "dataset"
    if stem.split(".", 1)[0].casefold() in _RESERVED_NAMES:
        stem = f"dataset-{stem}"
    budget = MAX_DATASET_NAME_CHARS - len(suffix)
    return f"{stem[:budget]}{suffix}", FORMAT_BY_SUFFIX[suffix]


def _json_cell(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)[:2_000]
    return str(value)


def _modified_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
        timespec="seconds"
    )


class DatasetStore:
    """Thread-safe storage for workspace dataset uploads."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or workspace_root()).expanduser()
        self.uploads_dir = self.root / _RELATIVE_UPLOAD_DIR
        self._lock = threading.Lock()

    def _path_for(self, file_id: str) -> Path:
        if (
            not file_id
            or len(file_id) > MAX_DATASET_NAME_CHARS
            or file_id != Path(file_id).name
            or Path(file_id).suffix.lower() not in FORMAT_BY_SUFFIX
        ):
            raise InvalidDataset(f"Invalid dataset id: {file_id!r}")
        path = self.uploads_dir / file_id
        if not path.is_file() or path.is_symlink():
            raise DatasetNotFound(f"Dataset not found: {file_id}")
        return path

    def _summary(self, path: Path) -> DatasetSummary:
        suffix = path.suffix.lower()
        return DatasetSummary(
            id=path.name,
            name=path.name,
            path=(_RELATIVE_UPLOAD_DIR / path.name).as_posix(),
            format=FORMAT_BY_SUFFIX[suffix],
            size_bytes=path.stat().st_size,
            modified_at=_modified_at(path),
        )

    def _unique_name(self, candidate: str) -> str:
        path = self.uploads_dir / candidate
        if not path.exists():
            return candidate
        stem = path.stem
        suffix = path.suffix
        index = 2
        while True:
            marker = f"-{index}"
            budget = MAX_DATASET_NAME_CHARS - len(suffix) - len(marker)
            name = f"{stem[:budget]}{marker}{suffix}"
            if not (self.uploads_dir / name).exists():
                return name
            index += 1

    def list_datasets(self) -> list[DatasetSummary]:
        """List supported upload files, newest first."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        summaries: list[DatasetSummary] = []
        for path in self.uploads_dir.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix.lower() not in FORMAT_BY_SUFFIX:
                continue
            try:
                summaries.append(self._summary(path))
            except OSError:
                continue
        return sorted(
            summaries,
            key=lambda item: (item.modified_at, item.name.lower()),
            reverse=True,
        )

    def get_dataset(self, file_id: str) -> DatasetSummary:
        """Return metadata for one dataset."""
        return self._summary(self._path_for(file_id))

    def save(self, filename: str, source: BinaryIO) -> DatasetSummary:
        """Stream an upload into a uniquely named workspace file."""
        cleaned, _format = _safe_name(filename)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            final_name = self._unique_name(cleaned)
            destination = self.uploads_dir / final_name
            temporary = self.uploads_dir / f".{final_name}.upload-{uuid.uuid4().hex}.tmp"
            total = 0
            try:
                with temporary.open("wb") as handle:
                    while chunk := source.read(_UPLOAD_CHUNK_BYTES):
                        total += len(chunk)
                        if total > MAX_DATASET_BYTES:
                            raise DatasetTooLarge(
                                "Dataset exceeds the "
                                f"{MAX_DATASET_BYTES // (1024 * 1024)} MiB limit"
                            )
                        handle.write(chunk)
                if total == 0:
                    raise InvalidDataset("Dataset file is empty")
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            return self._summary(destination)

    def delete(self, file_id: str) -> DatasetSummary:
        """Remove one dataset and return its final metadata."""
        with self._lock:
            path = self._path_for(file_id)
            summary = self._summary(path)
            path.unlink()
            return summary

    def preview(
        self,
        file_id: str,
        *,
        limit: int = DEFAULT_DATASET_PREVIEW_ROWS,
    ) -> DatasetPreview:
        """Parse the first ``limit`` rows of a stored dataset."""
        bounded = max(1, min(limit, MAX_DATASET_PREVIEW_ROWS))
        path = self._path_for(file_id)
        try:
            frame = self._read_frame(path, bounded + 1)
        except FileNotFoundError as exc:
            raise DatasetNotFound(f"Dataset not found: {file_id}") from exc
        except Exception as exc:
            raise DatasetPreviewError(
                f"Could not preview {file_id}: {exc}"
            ) from exc
        truncated = frame.height > bounded
        preview = frame.head(bounded)
        columns = list(preview.columns)
        rows = [
            [_json_cell(row[column]) for column in columns]
            for row in preview.to_dicts()
        ]
        return DatasetPreview(
            dataset=self._summary(path),
            columns=columns,
            rows=rows,
            truncated=truncated,
            row_limit=bounded,
        )

    def _read_frame(self, path: Path, rows: int) -> pl.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pl.read_csv(path, n_rows=rows)
        if suffix == ".tsv":
            return pl.read_csv(path, separator="\t", n_rows=rows)
        if suffix == ".parquet":
            return pl.scan_parquet(path).head(rows).collect()
        if suffix in {".ndjson", ".jsonl"}:
            return pl.scan_ndjson(path).head(rows).collect()
        if suffix in {".xlsx", ".xlsm"}:
            frame = pl.read_excel(path, sheet_id=1)
            if not isinstance(frame, pl.DataFrame):
                raise DatasetPreviewError("Excel workbook has no readable sheet")
            return frame.head(rows)
        raise InvalidDataset(f"Unsupported dataset type: {suffix}")


@lru_cache(maxsize=1)
def get_dataset_store() -> DatasetStore:
    """Return the process-wide dataset store."""
    return DatasetStore()
