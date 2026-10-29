"""Repo-root local model library.

The model library is intentionally separate from the workspace: models are
shared by every project and are selected by repo-relative paths such as
``models/yolo/custom.pt`` or ``models/diffusion/my-flux-repo``. The backend only
scans metadata; model files are loaded by node code at execution time.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from .memory_estimate import estimate_model_memory
from .quantization import detect_quantization, read_parameter_count
from .schemas import (
    MAX_DATASET_NAME_CHARS,
    MODEL_SELECT_KINDS,
    ModelSelectKind,
    ModelSummary,
    ModelTypeSummary,
    QuantizationInfo,
)

__all__ = [
    "DEFAULT_MODEL_TYPES",
    "MAX_MODEL_FOLDER_ENTRIES",
    "MODELS_ENV",
    "MODEL_LIBRARY_PREFIX",
    "MODEL_SUFFIXES",
    "ModelError",
    "InvalidModelPath",
    "ModelNotFound",
    "ModelStore",
    "default_models_root",
    "get_model_store",
]

MODELS_ENV = "COMPLEXUI_MODELS_DIR"
MODEL_LIBRARY_PREFIX = "models"
MAX_MODEL_FOLDER_ENTRIES = 512
DEFAULT_MODEL_TYPES: tuple[str, ...] = (
    "yolo",
    "torchvision",
    "classification",
    "diffusion",
    "unet",
    "vae",
    "clip",
)
MODEL_FORMATS: dict[str, str] = {
    ".pt": "PyTorch",
    ".pth": "PyTorch",
    ".onnx": "ONNX",
    ".safetensors": "SafeTensors",
}
MODEL_SUFFIXES = frozenset(MODEL_FORMATS)
_MODEL_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,158}")
_MODEL_FOLDER_MARKERS = frozenset(
    {"model_index.json", "modular_model_index.json", "config.json"}
)
_MODEL_COMPONENT_DIRS = frozenset(
    {
        "scheduler",
        "text_conditioner",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "transformer",
        "unet",
        "vae",
    }
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelError(Exception):
    """Base class for local model library failures."""


class InvalidModelPath(ModelError):
    """A model type or path is malformed or escapes the library root."""


class ModelNotFound(ModelError):
    """A requested model file or folder does not exist."""


@dataclass(frozen=True, slots=True)
class _FolderStats:
    size_bytes: int
    mtime_ns: int
    fingerprint: str
    executable: bool


def default_models_root() -> Path:
    """Return the configured model root, defaulting to ``<repo>/models``."""
    configured = os.environ.get(MODELS_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else _PROJECT_ROOT / path
    return _PROJECT_ROOT / MODEL_LIBRARY_PREFIX


def _modified_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
        timespec="seconds"
    )


def _valid_model_name(name: str) -> bool:
    return (
        bool(name)
        and len(name) <= MAX_DATASET_NAME_CHARS
        and name not in {".", ".."}
        and _MODEL_NAME.fullmatch(name) is not None
    )


def _is_link(path: Path) -> bool:
    """True for symlinks and Windows junctions."""
    return path.is_symlink() or path.is_junction()


def _directory_has_model_content(path: Path) -> bool:
    """Return whether a folder looks like a diffusers repo or component."""
    try:
        entries = path.iterdir()
    except OSError:
        return False
    for entry in entries:
        if _is_link(entry) or entry.name.startswith((".", "__")):
            continue
        try:
            if entry.is_file() and (
                entry.name in _MODEL_FOLDER_MARKERS
                or entry.suffix.lower() in MODEL_SUFFIXES
            ):
                return True
            if entry.is_dir() and entry.name in _MODEL_COMPONENT_DIRS:
                return True
        except OSError:
            continue
    return False


def _folder_stats(path: Path) -> _FolderStats:
    """Return a bounded recursive size/manifest fingerprint for a folder."""
    try:
        root_stat = path.stat()
    except OSError:
        return _FolderStats(0, 0, "missing", False)
    manifest: list[str] = [f"root:{root_stat.st_mtime_ns}"]
    total_size = 0
    executable = False
    visited = 0
    pending: list[tuple[Path, str]] = [(path, "")]
    while pending and visited < MAX_MODEL_FOLDER_ENTRIES:
        directory, prefix = pending.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        child_directories: list[tuple[Path, str]] = []
        for entry in entries:
            if visited >= MAX_MODEL_FOLDER_ENTRIES:
                break
            if entry.name.startswith((".", "__")):
                continue
            try:
                if _is_link(entry):
                    continue
                stat = entry.stat()
            except OSError:
                continue
            relative = f"{prefix}{entry.name}"
            try:
                if entry.is_dir():
                    manifest.append(f"d:{relative}:{stat.st_mtime_ns}")
                    visited += 1
                    child_directories.append((entry, f"{relative}/"))
                elif entry.is_file():
                    manifest.append(
                        f"f:{relative}:{stat.st_size}:{stat.st_mtime_ns}"
                    )
                    total_size += stat.st_size
                    visited += 1
                    if entry.suffix.lower() == ".pt":
                        executable = True
            except OSError:
                continue
        pending.extend(reversed(child_directories))
    digest = hashlib.sha256("\n".join(manifest).encode("utf-8")).hexdigest()
    return _FolderStats(
        size_bytes=total_size,
        mtime_ns=root_stat.st_mtime_ns,
        fingerprint=f"folder:{digest}",
        executable=executable,
    )


@lru_cache(maxsize=1024)
def _cached_quantization(
    path: str, stamp: str, model_type: str
) -> QuantizationInfo | None:
    """Cache header scans by path and content stamp (bounded, metadata only)."""
    del stamp  # part of the cache key only
    try:
        return detect_quantization(Path(path), model_type)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1024)
def _cached_parameter_count(path: str, stamp: str) -> int | None:
    del stamp  # part of the cache key only
    try:
        return read_parameter_count(Path(path))
    except (OSError, ValueError):
        return None


def _validate_select(select: str) -> ModelSelectKind:
    if select == "file":
        return "file"
    if select == "folder":
        return "folder"
    if select == "any":
        return "any"
    raise InvalidModelPath(
        f"Model selection kind must be one of {list(MODEL_SELECT_KINDS)!r}"
    )


class ModelStore:
    """Metadata-only view of the repo model folder."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_models_root()).expanduser()
        self._ensure_root()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for model_type in DEFAULT_MODEL_TYPES:
            directory = self.root / model_type
            if directory.exists() and _is_link(directory):
                raise InvalidModelPath(
                    f"Model type directory may not be a symlink: {directory.name}"
                )
            directory.mkdir(parents=True, exist_ok=True)

    def ensure(self) -> None:
        """Recreate the model root and default type folders if needed."""
        self._ensure_root()

    @staticmethod
    def _validate_type(model_type: str) -> str:
        candidate = model_type.strip()
        if (
            not candidate
            or len(candidate) > 64
            or candidate in {".", ".."}
            or not _MODEL_TYPE.fullmatch(candidate)
        ):
            raise InvalidModelPath(f"Invalid model type: {model_type!r}")
        return candidate

    def _type_dir(self, model_type: str, *, create: bool = False) -> Path:
        candidate = self._validate_type(model_type)
        directory = self.root / candidate
        if directory.exists() and (not directory.is_dir() or _is_link(directory)):
            raise InvalidModelPath(f"Invalid model type directory: {candidate}")
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _iter_models(directory: Path) -> list[tuple[Path, bool]]:
        """Return eligible files and diffusers-style folders in one type."""
        try:
            entries = sorted(
                directory.iterdir(), key=lambda item: item.name.lower()
            )
        except OSError:
            return []
        models: list[tuple[Path, bool]] = []
        for path in entries:
            if _is_link(path) or not _valid_model_name(path.name):
                continue
            if path.is_file():
                if path.suffix.lower() in MODEL_SUFFIXES:
                    models.append((path, False))
            elif path.is_dir() and _directory_has_model_content(path):
                models.append((path, True))
        return models

    def list_types(self) -> list[ModelTypeSummary]:
        """List default and custom model type folders with model counts."""
        self.ensure()
        names = set(DEFAULT_MODEL_TYPES)
        for path in self.root.iterdir():
            if path.is_dir() and not _is_link(path):
                try:
                    names.add(self._validate_type(path.name))
                except InvalidModelPath:
                    continue
        ordered = sorted(
            names,
            key=lambda name: (
                DEFAULT_MODEL_TYPES.index(name)
                if name in DEFAULT_MODEL_TYPES
                else len(DEFAULT_MODEL_TYPES),
                name.lower(),
            ),
        )
        result: list[ModelTypeSummary] = []
        for name in ordered:
            directory = self._type_dir(name)
            count = len(self._iter_models(directory))
            result.append(
                ModelTypeSummary(
                    name=name,
                    path=f"{MODEL_LIBRARY_PREFIX}/{name}",
                    count=count,
                    default=name in DEFAULT_MODEL_TYPES,
                )
            )
        return result

    def _summary(self, model_type: str, path: Path) -> ModelSummary:
        if path.is_dir():
            stats = _folder_stats(path)
            quantization = _cached_quantization(
                str(path), stats.fingerprint, model_type
            )
            parameters = (
                None
                if quantization is not None
                else _cached_parameter_count(str(path), stats.fingerprint)
            )
            memory = estimate_model_memory(
                size_bytes=stats.size_bytes,
                quantization=quantization,
                parameter_count=parameters,
                is_directory=True,
            )
            return ModelSummary(
                id=f"{model_type}/{path.name}",
                name=path.name,
                path=f"{MODEL_LIBRARY_PREFIX}/{model_type}/{path.name}",
                model_type=model_type,
                format="Folder",
                size_bytes=stats.size_bytes,
                modified_at=_modified_at(path),
                mtime_ns=stats.mtime_ns,
                fingerprint=stats.fingerprint,
                executable=stats.executable,
                is_directory=True,
                parameter_count=parameters,
                quantization=quantization,
                memory=memory,
            )
        stat = path.stat()
        suffix = path.suffix.lower()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
        quantization = _cached_quantization(str(path), stamp, model_type)
        parameters = (
            None
            if quantization is not None
            else _cached_parameter_count(str(path), stamp)
        )
        memory = estimate_model_memory(
            size_bytes=stat.st_size,
            quantization=quantization,
            parameter_count=parameters,
        )
        return ModelSummary(
            id=f"{model_type}/{path.name}",
            name=path.name,
            path=f"{MODEL_LIBRARY_PREFIX}/{model_type}/{path.name}",
            model_type=model_type,
            format=MODEL_FORMATS[suffix],
            size_bytes=stat.st_size,
            modified_at=_modified_at(path),
            mtime_ns=stat.st_mtime_ns,
            fingerprint=stamp,
            executable=suffix == ".pt",
            parameter_count=parameters,
            quantization=quantization,
            memory=memory,
        )

    def list_models(self, model_type: str | None = None) -> list[ModelSummary]:
        """List supported files and folders, optionally scoped to one type."""
        self.ensure()
        types = [self._validate_type(model_type)] if model_type else [
            item.name for item in self.list_types()
        ]
        summaries: list[ModelSummary] = []
        for current in types:
            directory = self._type_dir(current)
            if not directory.is_dir():
                continue
            for path, _ in self._iter_models(directory):
                summaries.append(self._summary(current, path))
        return sorted(
            summaries,
            key=lambda item: (item.model_type.lower(), item.name.lower()),
        )

    def _locate(self, model_type: str, relative_parts: list[str]) -> Path:
        """Locate one existing file or folder inside a model-type directory."""
        directory = self._type_dir(model_type)
        target = directory
        for part in relative_parts:
            target = target / part
            if _is_link(target):
                raise ModelNotFound(
                    f"Model not found: "
                    f"{MODEL_LIBRARY_PREFIX}/{model_type}/{'/'.join(relative_parts)}"
                )
        if not target.exists():
            raise ModelNotFound(
                f"Model not found: "
                f"{MODEL_LIBRARY_PREFIX}/{model_type}/{'/'.join(relative_parts)}"
            )
        return target

    def _validate_target(
        self,
        model_type: str,
        target: Path,
        relative_parts: list[str],
        kind: ModelSelectKind,
    ) -> Path:
        if kind == "file" and not target.is_file():
            raise InvalidModelPath(
                f"Model '{target.name}' is a folder; this selector requires a file"
            )
        if kind == "folder" and not target.is_dir():
            raise InvalidModelPath(
                f"Model '{target.name}' is a file; this selector requires a folder"
            )
        directory = self._type_dir(model_type)
        if not target.resolve().is_relative_to(directory.resolve()):
            raise InvalidModelPath("Model paths must stay inside their type folder")
        return target

    def _resolve_selection(
        self,
        model_type: str,
        value: str,
        *,
        select: str = "any",
        include: Sequence[str] = (),
    ) -> tuple[str, Path]:
        """Resolve a selection and return its actual type and path."""
        expected = self._validate_type(model_type)
        kind = _validate_select(select)
        allowed: list[str] = [expected]
        for extra in include:
            actual = self._validate_type(extra)
            if actual not in allowed:
                allowed.append(actual)
        if len(allowed) > 9:
            raise InvalidModelPath("Model selectors support at most 8 include types")
        raw = value.strip().replace("\\", "/")
        if not raw:
            raise ModelNotFound("No local model was selected")
        parts = [part for part in raw.split("/") if part]
        if parts and parts[0] == MODEL_LIBRARY_PREFIX:
            parts = parts[1:]
        if not parts:
            raise InvalidModelPath("Model paths must be models/<type>/<name>")
        if len(parts) == 1:
            relative_parts = parts
            if any(not _valid_model_name(part) for part in relative_parts):
                raise InvalidModelPath(
                    f"Invalid model name: {'/'.join(relative_parts)!r}"
                )
            last_error: ModelError | None = None
            for candidate_type in allowed:
                try:
                    target = self._locate(candidate_type, relative_parts)
                except ModelNotFound as exc:
                    last_error = exc
                    continue
                return (
                    candidate_type,
                    self._validate_target(
                        candidate_type,
                        target,
                        relative_parts,
                        kind,
                    ),
                )
            suffix = Path(relative_parts[0]).suffix.lower()
            if kind != "folder" and suffix and suffix not in MODEL_SUFFIXES:
                raise InvalidModelPath(f"Invalid model file: {relative_parts[0]!r}")
            raise last_error or ModelNotFound("No local model was selected")
        actual_type = self._validate_type(parts[0])
        relative_parts = parts[1:]
        if actual_type not in allowed:
            raise InvalidModelPath(
                f"Model type '{actual_type}' does not match selector types "
                f"{allowed!r}"
            )
        if any(not _valid_model_name(part) for part in relative_parts):
            raise InvalidModelPath(
                f"Invalid model name: {'/'.join(relative_parts)!r}"
            )
        target = self._locate(actual_type, relative_parts)
        return (
            actual_type,
            self._validate_target(actual_type, target, relative_parts, kind),
        )

    def resolve(
        self,
        model_type: str,
        value: str,
        *,
        select: str = "any",
        include: Sequence[str] = (),
    ) -> Path:
        """Resolve a selected repo-relative model path to a file or folder."""
        _, path = self._resolve_selection(
            model_type,
            value,
            select=select,
            include=include,
        )
        return path

    def describe(
        self,
        model_type: str,
        value: str,
        *,
        select: str = "any",
        include: Sequence[str] = (),
    ) -> ModelSummary:
        """Return metadata for a selected model file or folder."""
        actual_type, path = self._resolve_selection(
            model_type,
            value,
            select=select,
            include=include,
        )
        return self._summary(actual_type, path)


@lru_cache(maxsize=1)
def get_model_store() -> ModelStore:
    """Return the process-wide local model store."""
    return ModelStore()
