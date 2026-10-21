"""Template hashing and trust verification.

The backend only executes node code that either matches an official template
byte-for-byte (after line-ending normalization) or was explicitly approved by
the user for the current session via ``POST /api/approve``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

__all__ = [
    "DEFAULT_TEMPLATES_DIR",
    "SecurityRegistry",
    "TemplateRecord",
    "UntrustedNodeError",
    "code_hash",
    "get_security_registry",
    "normalize_code",
]


def normalize_code(code: str) -> str:
    """Normalize line endings so hashes are platform independent."""
    return code.replace("\r\n", "\n").replace("\r", "\n")


def code_hash(code: str) -> str:
    """SHA-256 hex digest of node code."""
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TemplateRecord:
    name: str
    code: str
    code_hash: str


class UntrustedNodeError(Exception):
    """Raised when a node with non-official, non-approved code is executed."""

    def __init__(self, node_id: str, code_hash_value: str) -> None:
        super().__init__(
            f"Node '{node_id}' contains untrusted code "
            f"(sha256:{code_hash_value[:12]}...); approve it before execution"
        )
        self.node_id = node_id
        self.code_hash = code_hash_value


class SecurityRegistry:
    """Loads and hashes official templates; tracks session approvals."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self._templates_by_name: dict[str, TemplateRecord] = {}
        self._template_hashes: dict[str, str] = {}
        self._approvals: set[tuple[str, str]] = set()
        self._load_templates()

    def _load_templates(self) -> None:
        if not self.templates_dir.is_dir():
            return
        for path in sorted(self.templates_dir.glob("*.py")):
            code = path.read_text(encoding="utf-8")
            record = TemplateRecord(name=path.stem, code=code, code_hash=code_hash(code))
            self._templates_by_name[record.name] = record
            self._template_hashes[record.code_hash] = record.name

    def templates(self) -> list[TemplateRecord]:
        return list(self._templates_by_name.values())

    def get_template(self, name: str) -> TemplateRecord:
        return self._templates_by_name[name]

    def template_for_hash(self, hash_value: str) -> TemplateRecord | None:
        name = self._template_hashes.get(hash_value)
        return self._templates_by_name[name] if name is not None else None

    def is_official(self, code: str) -> bool:
        return code_hash(code) in self._template_hashes

    def approve(self, node_id: str, hash_value: str) -> None:
        self._approvals.add((node_id, hash_value))

    def is_approved(self, node_id: str, hash_value: str) -> bool:
        return (node_id, hash_value) in self._approvals

    def approved_pairs(self) -> list[tuple[str, str]]:
        return sorted(self._approvals)

    def is_trusted(self, node_id: str, code: str) -> bool:
        value = code_hash(code)
        return value in self._template_hashes or self.is_approved(node_id, value)


@lru_cache(maxsize=1)
def get_security_registry() -> SecurityRegistry:
    """Process-wide registry loaded from the bundled templates directory."""
    return SecurityRegistry()
