"""Repo-root local model library.

The model library is intentionally separate from the workspace: models are
shared by every project and are selected by repo-relative paths such as
``models/yolo/custom.pt``. The backend only scans metadata; model files are
loaded by node code at execution time.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from .schemas import (
    MAX_DATASET_NAME_CHARS,
    ModelSummary,
    ModelTypeSummary,
)

__all__ = [
    "DEFAULT_MODEL_TYPES",
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
DEFAULT_MODEL_TYPES: tuple[str, ...] = (
    "yolo",
    "torchvision",
    "classification",
    "diffusion",
    "unet",
    "vae",
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
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelError(Exception):
    """Base class for local model library failures."""


class InvalidModelPath(ModelError):
    """A model type or path is malformed or escapes the library root."""


class ModelNotFound(ModelError):
    """A requested model file does not exist."""


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


class ModelStore:
    """Metadata-only view of the repo model folder."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_models_root()).expanduser()
        self._ensure_root()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for model_type in DEFAULT_MODEL_TYPES:
            directory = self.root / model_type
            if directory.exists() and directory.is_symlink():
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
        if directory.exists() and (not directory.is_dir() or directory.is_symlink()):
            raise InvalidModelPath(f"Invalid model type directory: {candidate}")
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def list_types(self) -> list[ModelTypeSummary]:
        """List default and custom model type folders with file counts."""
        self.ensure()
        names = set(DEFAULT_MODEL_TYPES)
        for path in self.root.iterdir():
            if path.is_dir() and not path.is_symlink():
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
            count = sum(
                1
                for path in directory.iterdir()
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in MODEL_SUFFIXES
            )
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
        stat = path.stat()
        suffix = path.suffix.lower()
        return ModelSummary(
            id=f"{model_type}/{path.name}",
            name=path.name,
            path=f"{MODEL_LIBRARY_PREFIX}/{model_type}/{path.name}",
            model_type=model_type,
            format=MODEL_FORMATS[suffix],
            size_bytes=stat.st_size,
            modified_at=_modified_at(path),
            mtime_ns=stat.st_mtime_ns,
            fingerprint=f"{stat.st_size}:{stat.st_mtime_ns}",
            executable=suffix == ".pt",
        )

    def list_models(self, model_type: str | None = None) -> list[ModelSummary]:
        """List supported files, optionally scoped to one model type."""
        self.ensure()
        types = [self._validate_type(model_type)] if model_type else [
            item.name for item in self.list_types()
        ]
        summaries: list[ModelSummary] = []
        for current in types:
            directory = self._type_dir(current)
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.suffix.lower() not in MODEL_SUFFIXES
                    or not _MODEL_NAME.fullmatch(path.name)
                    or len(path.name) > MAX_DATASET_NAME_CHARS
                ):
                    continue
                summaries.append(self._summary(current, path))
        return sorted(
            summaries,
            key=lambda item: (item.model_type.lower(), item.name.lower()),
        )

    def resolve(self, model_type: str, value: str) -> Path:
        """Resolve a selected repo-relative model path to an absolute file."""
        expected = self._validate_type(model_type)
        raw = value.strip().replace("\\", "/")
        if not raw:
            raise ModelNotFound("No local model was selected")
        parts = [part for part in raw.split("/") if part]
        if parts and parts[0] == MODEL_LIBRARY_PREFIX:
            parts = parts[1:]
        if len(parts) == 1:
            actual_type = expected
            name = parts[0]
        elif len(parts) == 2:
            actual_type = self._validate_type(parts[0])
            name = parts[1]
        else:
            raise InvalidModelPath("Model paths must be models/<type>/<file>")
        if actual_type != expected:
            raise InvalidModelPath(
                f"Model type '{actual_type}' does not match selector type '{expected}'"
            )
        if (
            not name
            or name in {".", ".."}
            or name != Path(name).name
            or not _MODEL_NAME.fullmatch(name)
            or len(name) > MAX_DATASET_NAME_CHARS
            or Path(name).suffix.lower() not in MODEL_SUFFIXES
        ):
            raise InvalidModelPath(f"Invalid model file: {name!r}")
        directory = self._type_dir(expected)
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ModelNotFound(
                f"Model not found: {MODEL_LIBRARY_PREFIX}/{expected}/{name}"
            )
        if path.resolve().parent != directory.resolve():
            raise InvalidModelPath("Model paths must stay inside their type folder")
        return path

    def describe(self, model_type: str, value: str) -> ModelSummary:
        """Return metadata for a selected model path."""
        path = self.resolve(model_type, value)
        return self._summary(self._validate_type(model_type), path)


@lru_cache(maxsize=1)
def get_model_store() -> ModelStore:
    """Return the process-wide local model store."""
    return ModelStore()
