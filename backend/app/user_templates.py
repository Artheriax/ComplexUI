"""User-created node templates stored in the writable workspace.

User templates are explicit local extensions: saving one through the API
records its code under the workspace and makes that exact code trusted by the
security registry. They never mutate the bundled templates directory.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .schemas import MAX_NODE_CODE_CHARS
from .workspace import workspace_root

__all__ = [
    "InvalidUserTemplate",
    "UserTemplateConflict",
    "UserTemplateRecord",
    "UserTemplateStore",
    "get_user_template_store",
]

USER_TEMPLATES_ENV = "COMPLEXUI_USER_TEMPLATES_DIR"
_NAME = re.compile(r"[a-z0-9][a-z0-9_]{0,79}")


class UserTemplateError(RuntimeError):
    """Base class for user template failures."""


class InvalidUserTemplate(UserTemplateError):
    """The template name, metadata or code is invalid."""


class UserTemplateConflict(UserTemplateError):
    """A template with the same name already exists."""


@dataclass(frozen=True, slots=True)
class UserTemplateRecord:
    name: str
    title: str
    category: str
    description: str
    code: str
    path: Path


def default_user_templates_dir() -> Path:
    configured = os.environ.get(USER_TEMPLATES_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return workspace_root().expanduser() / "templates"


class UserTemplateStore:
    """Read and write user templates below one writable directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_user_templates_dir()).expanduser()

    def _ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_name(name: str) -> str:
        candidate = name.strip()
        if not _NAME.fullmatch(candidate):
            raise InvalidUserTemplate(
                "Template names must use lowercase letters, digits and underscores"
            )
        return candidate

    def _metadata_path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def _code_path(self, name: str) -> Path:
        return self.root / f"{name}.py"

    def _metadata(self, name: str) -> dict[str, Any]:
        path = self._metadata_path(name)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def list_templates(self) -> list[UserTemplateRecord]:
        if not self.root.is_dir():
            return []
        records: list[UserTemplateRecord] = []
        for path in sorted(self.root.glob("*.py")):
            name = path.stem
            if not _NAME.fullmatch(name):
                continue
            try:
                code = path.read_text(encoding="utf-8")
            except OSError:
                continue
            metadata = self._metadata(name)
            records.append(
                UserTemplateRecord(
                    name=name,
                    title=str(metadata.get("title") or name.replace("_", " ").title()),
                    category=str(metadata.get("category") or "Custom"),
                    description=str(metadata.get("description") or ""),
                    code=code,
                    path=path,
                )
            )
        return records

    def get(self, name: str) -> UserTemplateRecord:
        candidate = self._validate_name(name)
        path = self._code_path(candidate)
        try:
            code = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UserTemplateError(f"Template not found: {candidate}") from exc
        metadata = self._metadata(candidate)
        return UserTemplateRecord(
            name=candidate,
            title=str(metadata.get("title") or candidate.replace("_", " ").title()),
            category=str(metadata.get("category") or "Custom"),
            description=str(metadata.get("description") or ""),
            code=code,
            path=path,
        )

    def save(
        self,
        *,
        name: str,
        title: str,
        category: str,
        description: str,
        code: str,
        overwrite: bool = False,
    ) -> UserTemplateRecord:
        candidate = self._validate_name(name)
        clean_title = title.strip()
        clean_category = category.strip() or "Custom"
        clean_description = description.strip()
        if not clean_title:
            raise InvalidUserTemplate("Template title is required")
        if len(code) > MAX_NODE_CODE_CHARS:
            raise InvalidUserTemplate("Template code is too large")
        if not code.strip():
            raise InvalidUserTemplate("Template code is required")
        self._ensure()
        code_path = self._code_path(candidate)
        if code_path.exists() and not overwrite:
            raise UserTemplateConflict(f"Template '{candidate}' already exists")
        normalized = code.replace("\r\n", "\n").replace("\r", "\n")
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.root,
            prefix=f".{candidate}-",
            suffix=".py",
            delete=False,
        ) as handle:
            handle.write(normalized)
            temp_code = Path(handle.name)
        temp_code.replace(code_path)
        metadata = {
            "title": clean_title,
            "category": clean_category,
            "description": clean_description,
        }
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.root,
            prefix=f".{candidate}-",
            suffix=".json",
            delete=False,
        ) as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temp_metadata = Path(handle.name)
        temp_metadata.replace(self._metadata_path(candidate))
        return UserTemplateRecord(
            name=candidate,
            title=clean_title,
            category=clean_category,
            description=clean_description,
            code=normalized,
            path=code_path,
        )

    def delete(self, name: str) -> None:
        candidate = self._validate_name(name)
        code_path = self._code_path(candidate)
        if not code_path.is_file():
            raise UserTemplateError(f"Template not found: {candidate}")
        code_path.unlink()
        try:
            self._metadata_path(candidate).unlink()
        except FileNotFoundError:
            pass


@lru_cache(maxsize=1)
def get_user_template_store() -> UserTemplateStore:
    return UserTemplateStore()
