"""Workspace-backed asset uploads, table previews and generated outputs.

Uploaded files live under the repo-root ``inputs`` directory and are referenced
from ``File``/``ImageFile``/``VideoFile`` widgets by their ``inputs/...`` path.
Generated files live under the repo-root ``outputs`` directory and are addressed by
``OutputFile`` widgets. File names are sanitized, collisions get a numeric
suffix, and every upload is streamed through a per-kind size cap before being
atomically moved into place.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import re
import threading
import uuid
from collections.abc import Iterator
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
    MAX_MEDIA_BYTES,
    MAX_OUTPUT_PATH_CHARS,
    AssetKind,
    DatasetPreview,
    DatasetSummary,
    OutputSummary,
)

__all__ = [
    "DatasetError",
    "DatasetNotFound",
    "DatasetPreviewError",
    "DatasetStore",
    "DatasetTooLarge",
    "InvalidAssetPath",
    "InvalidDataset",
    "MAX_UPLOAD_REQUEST_BYTES",
    "OutputStore",
    "SUPPORTED_DATASET_SUFFIXES",
    "SUPPORTED_IMAGE_SUFFIXES",
    "SUPPORTED_VIDEO_SUFFIXES",
    "get_dataset_store",
    "get_output_store",
]

TABLE_FORMATS: dict[str, str] = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".parquet": "Parquet",
    ".xlsx": "Excel",
    ".xlsm": "Excel",
    ".ndjson": "NDJSON",
    ".jsonl": "NDJSON",
}
IMAGE_FORMATS: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WebP",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
VIDEO_FORMATS: dict[str, str] = {
    ".mp4": "MP4",
    ".webm": "WebM",
    ".mov": "QuickTime",
    ".avi": "AVI",
    ".mkv": "Matroska",
    ".m4v": "M4V",
}
ASSET_FORMATS: dict[str, tuple[str, AssetKind]] = {
    **{suffix: (fmt, AssetKind.TABLE) for suffix, fmt in TABLE_FORMATS.items()},
    **{suffix: (fmt, AssetKind.IMAGE) for suffix, fmt in IMAGE_FORMATS.items()},
    **{suffix: (fmt, AssetKind.VIDEO) for suffix, fmt in VIDEO_FORMATS.items()},
}
SUPPORTED_DATASET_SUFFIXES = tuple(TABLE_FORMATS)
SUPPORTED_IMAGE_SUFFIXES = tuple(IMAGE_FORMATS)
SUPPORTED_VIDEO_SUFFIXES = tuple(VIDEO_FORMATS)
MAX_UPLOAD_REQUEST_BYTES = MAX_MEDIA_BYTES + (2 * 1024 * 1024)
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_RELATIVE_INPUT_DIR = Path("inputs")
_DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / _RELATIVE_INPUT_DIR
_INPUT_DIR_ENV = "COMPLEXUI_INPUT_DIR"
_RELATIVE_OUTPUT_DIR = Path("outputs")
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / _RELATIVE_OUTPUT_DIR
_OUTPUT_DIR_ENV = "COMPLEXUI_OUTPUT_DIR"
_OUTPUT_SUFFIXES = {
    *IMAGE_FORMATS,
    *VIDEO_FORMATS,
    ".json",
    ".csv",
    ".txt",
}
_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,158}")
_MAX_OUTPUT_DEPTH = 8
_MAX_OUTPUT_SCAN_DIRECTORIES = 2_000
_MAX_OUTPUT_SCAN_FILES = 5_000
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
    """A stored asset could not be parsed for preview."""


class InvalidAssetPath(DatasetError):
    """An output path is invalid or escapes the workspace outputs directory."""


def _safe_name(raw: str) -> tuple[str, str, AssetKind]:
    name = Path(raw.replace("\\", "/")).name.strip().strip(".")
    cleaned = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name).strip(" .")
    if not cleaned:
        cleaned = "dataset.csv"
    suffix = Path(cleaned).suffix.lower()
    asset = ASSET_FORMATS.get(suffix)
    if asset is None:
        supported = ", ".join(sorted(ASSET_FORMATS))
        raise InvalidDataset(f"Unsupported asset type '{suffix or '<none>'}'; use {supported}")
    fmt, kind = asset
    stem = cleaned[: -len(suffix)].strip(" .") or "asset"
    if stem.split(".", 1)[0].casefold() in _RESERVED_NAMES:
        stem = f"asset-{stem}"
    budget = MAX_DATASET_NAME_CHARS - len(suffix)
    return f"{stem[:budget]}{suffix}", fmt, kind


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


def default_input_dir() -> Path:
    """Return the configured uploaded-input directory."""
    configured = os.environ.get(_INPUT_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return _DEFAULT_INPUT_DIR


class DatasetStore:
    """Thread-safe storage for repo-root input uploads."""

    def __init__(self, inputs_dir: Path | None = None) -> None:
        self.uploads_dir = (inputs_dir or default_input_dir()).expanduser()
        self._lock = threading.Lock()

    def _path_for(self, file_id: str) -> Path:
        if (
            not file_id
            or len(file_id) > MAX_DATASET_NAME_CHARS
            or file_id != Path(file_id).name
            or Path(file_id).suffix.lower() not in ASSET_FORMATS
        ):
            raise InvalidDataset(f"Invalid asset id: {file_id!r}")
        path = self.uploads_dir / file_id
        if not path.is_file() or path.is_symlink():
            raise DatasetNotFound(f"Asset not found: {file_id}")
        return path

    def _summary(self, path: Path) -> DatasetSummary:
        suffix = path.suffix.lower()
        fmt, kind = ASSET_FORMATS[suffix]
        return DatasetSummary(
            id=path.name,
            name=path.name,
            path=(_RELATIVE_INPUT_DIR / path.name).as_posix(),
            format=fmt,
            kind=kind,
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

    def resolve_path(self, value: str) -> Path:
        """Resolve an ``inputs/...`` value to a stored upload."""
        raw = value.strip().replace("\\", "/")
        if raw.startswith("inputs/"):
            raw = raw[len("inputs/") :]
        return self._path_for(raw)

    def list_datasets(self) -> list[DatasetSummary]:
        """List supported upload files, newest first."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        summaries: list[DatasetSummary] = []
        for path in self.uploads_dir.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix.lower() not in ASSET_FORMATS:
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
        cleaned, _format, kind = _safe_name(filename)
        limit = MAX_DATASET_BYTES if kind is AssetKind.TABLE else MAX_MEDIA_BYTES
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
                        if total > limit:
                            raise DatasetTooLarge(
                                f"{kind.value.title()} asset exceeds the "
                                f"{limit // (1024 * 1024)} MiB limit"
                            )
                        handle.write(chunk)
                if total == 0:
                    raise InvalidDataset("Asset file is empty")
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            return self._summary(destination)

    def raw_path(self, file_id: str) -> Path:
        """Return a validated upload path for raw media serving."""
        path = self._path_for(file_id)
        if path.suffix.lower() not in {*IMAGE_FORMATS, *VIDEO_FORMATS}:
            raise InvalidDataset(f"Asset is not a media file: {file_id}")
        return path

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
        if ASSET_FORMATS[path.suffix.lower()][1] is not AssetKind.TABLE:
            raise DatasetPreviewError(f"Only tabular assets have table previews: {file_id}")
        try:
            frame = self._read_frame(path, bounded + 1)
        except FileNotFoundError as exc:
            raise DatasetNotFound(f"Asset not found: {file_id}") from exc
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


def default_output_dir() -> Path:
    """Return the configured generated-output directory."""
    configured = os.environ.get(_OUTPUT_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return _DEFAULT_OUTPUT_DIR


class OutputStore:
    """Resolve and serve node-generated files under the repo-root ``outputs``."""

    def __init__(self, outputs_dir: Path | None = None) -> None:
        self.outputs_dir = (outputs_dir or default_output_dir()).expanduser()

    def _validate_relative_name(self, value: str) -> str:
        candidate = value.strip().replace("\\", "/")
        prefix = f"{_RELATIVE_OUTPUT_DIR.as_posix()}/"
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
        if not candidate or len(candidate) > MAX_OUTPUT_PATH_CHARS:
            raise InvalidAssetPath(f"Invalid output path: {value!r}")
        path = Path(candidate)
        if path.is_absolute():
            raise InvalidAssetPath(f"Invalid output path: {value!r}")
        parts = [part for part in candidate.split("/") if part not in {"", "."}]
        if (
            not parts
            or len(parts) > _MAX_OUTPUT_DEPTH + 1
            or any(part == ".." for part in parts)
        ):
            raise InvalidAssetPath(f"Invalid output path: {value!r}")
        for part in parts[:-1]:
            if not _OUTPUT_NAME.fullmatch(part):
                raise InvalidAssetPath(f"Invalid output path: {value!r}")
        name = parts[-1]
        if (
            len(name) > MAX_DATASET_NAME_CHARS
            or not _OUTPUT_NAME.fullmatch(name)
            or Path(name).suffix.lower() not in _OUTPUT_SUFFIXES
        ):
            raise InvalidAssetPath(f"Invalid output path: {value!r}")
        return "/".join(parts)

    def resolve_path(self, value: str) -> Path:
        """Resolve an ``OutputFile`` widget value and create its parent directory."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        root = self.outputs_dir.resolve()
        raw = value.strip()
        path = Path(raw).expanduser()
        if path.is_absolute():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise InvalidAssetPath("Output paths must stay inside outputs.")
            relative = self._validate_relative_name(
                resolved.relative_to(root).as_posix()
            )
        else:
            relative = self._validate_relative_name(raw)
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise InvalidAssetPath("Output paths must stay inside outputs.")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def media_path(self, file_id: str) -> Path:
        """Return a validated generated image/video for raw serving."""
        relative = self._validate_relative_name(file_id)
        if Path(relative).suffix.lower() not in {*IMAGE_FORMATS, *VIDEO_FORMATS}:
            raise InvalidAssetPath(f"Output is not a media file: {file_id}")
        root = self.outputs_dir.resolve()
        candidate = root / relative
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise InvalidAssetPath(f"Invalid output path: {file_id!r}")
        if candidate.is_symlink() or candidate.parent.is_symlink():
            raise DatasetNotFound(f"Output not found: {file_id}")
        if not path.is_file():
            raise DatasetNotFound(f"Output not found: {file_id}")
        return path

    def relative_path(self, value: str) -> str | None:
        """Return an ``outputs/...`` path suitable for a raw output URL."""
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value.strip()).expanduser()
        candidates = (
            [path.resolve()]
            if path.is_absolute()
            else [
                (self.outputs_dir.parent / path).resolve(),
                (self.outputs_dir / path).resolve(),
            ]
        )
        for resolved in candidates:
            try:
                relative = resolved.relative_to(self.outputs_dir.resolve())
            except (OSError, ValueError):
                continue
            return (_RELATIVE_OUTPUT_DIR / relative).as_posix()
        return None

    def _summary(self, path: Path) -> OutputSummary:
        root = self.outputs_dir.resolve()
        relative = path.resolve().relative_to(root).as_posix()
        suffix = path.suffix.lower()
        fmt, kind = ASSET_FORMATS[suffix]
        return OutputSummary(
            id=relative,
            name=path.name,
            path=(_RELATIVE_OUTPUT_DIR / relative).as_posix(),
            format=fmt,
            kind=kind,
            size_bytes=path.stat().st_size,
            modified_at=_modified_at(path),
        )

    def _iter_media_files(self) -> Iterator[Path]:
        stack: list[tuple[Path, int]] = [(self.outputs_dir, 0)]
        directories = 0
        files = 0
        while stack and directories < _MAX_OUTPUT_SCAN_DIRECTORIES:
            directory, depth = stack.pop()
            directories += 1
            try:
                entries = tuple(directory.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if depth < _MAX_OUTPUT_DEPTH:
                        stack.append((entry, depth + 1))
                    continue
                if not entry.is_file():
                    continue
                suffix = entry.suffix.lower()
                if suffix not in {*IMAGE_FORMATS, *VIDEO_FORMATS}:
                    continue
                if (
                    len(entry.name) > MAX_DATASET_NAME_CHARS
                    or not _OUTPUT_NAME.fullmatch(entry.name)
                ):
                    continue
                yield entry
                files += 1
                if files >= _MAX_OUTPUT_SCAN_FILES:
                    return

    def list_outputs(self, limit: int = 200) -> list[OutputSummary]:
        """List generated images and videos, newest first."""
        if not self.outputs_dir.is_dir():
            return []
        bounded = max(1, min(limit, 500))

        def modified(path: Path) -> int:
            try:
                return path.stat().st_mtime_ns
            except OSError:
                return -1

        candidates = heapq.nlargest(
            bounded,
            self._iter_media_files(),
            key=modified,
        )
        summaries: list[OutputSummary] = []
        for path in candidates:
            try:
                summaries.append(self._summary(path))
            except (OSError, ValueError):
                continue
        return summaries

    def delete_output(self, file_id: str) -> OutputSummary:
        """Delete one generated image or video and return its metadata."""
        path = self.media_path(file_id)
        summary = self._summary(path)
        path.unlink()
        return summary


@lru_cache(maxsize=1)
def get_dataset_store() -> DatasetStore:
    """Return the process-wide dataset store."""
    return DatasetStore()


@lru_cache(maxsize=1)
def get_output_store() -> OutputStore:
    """Return the process-wide generated-output store."""
    return OutputStore()
