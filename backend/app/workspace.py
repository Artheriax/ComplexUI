"""File-backed workspace storage for named projects and recovery sessions.

The workspace root is ``backend/workspace`` by default and can be relocated
with ``COMPLEXUI_WORKSPACE``. Project documents use the same JSON format as
frontend import/export so saved and downloaded workflows remain portable.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import unicodedata
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from .schemas import (
    MAX_PROJECT_SLUG_CHARS,
    PROJECT_VERSION_RETENTION,
    ProjectDetail,
    ProjectSummary,
    ProjectVersionDetail,
    ProjectVersionInfo,
    SavedWorkflowDocument,
    WorkspaceSession,
    WorkspaceSessionRequest,
)
from .storage import utc_timestamp, write_json_atomic

_DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"
_WORKSPACE_ENV = "COMPLEXUI_WORKSPACE"
_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_VERSION_PATTERN = re.compile(r"[0-9A-Za-z-]+")
_PROJECT_SUFFIX = ".cuigraph.json"


class WorkspaceError(Exception):
    """Base class for workspace storage failures."""


class ProjectNotFound(WorkspaceError):
    """The requested project or version does not exist."""


class ProjectAlreadyExists(WorkspaceError):
    """Creation would overwrite an existing project without permission."""


class InvalidProjectSlug(WorkspaceError):
    """A slug is malformed or could escape the projects directory."""


class InvalidProjectDocument(WorkspaceError):
    """A stored workflow file is not a valid project document."""


def workspace_root() -> Path:
    """Return the configured workspace root without creating it."""
    configured = os.environ.get(_WORKSPACE_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return _DEFAULT_WORKSPACE


def slugify_project_name(name: str) -> str:
    """Build a filesystem-safe slug from a human-readable project name."""
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    slug = slug[:MAX_PROJECT_SLUG_CHARS].strip("-")
    return slug or "project"


def _validate_slug(slug: str) -> str:
    if not slug or len(slug) > MAX_PROJECT_SLUG_CHARS or not _SLUG_PATTERN.fullmatch(slug):
        raise InvalidProjectSlug(f"Invalid project slug: {slug!r}")
    return slug


def _validate_version(version: str) -> str:
    if not version or len(version) > 80 or not _VERSION_PATTERN.fullmatch(version):
        raise InvalidProjectSlug(f"Invalid project version: {version!r}")
    return version


def _timestamp() -> str:
    return utc_timestamp()


def _version_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid4().hex[:6]}"


class ProjectStore:
    """Thread-safe project, version-snapshot and session persistence."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or workspace_root()).expanduser()
        self.projects_dir = self.root / "workflows"
        self.versions_dir = self.root / ".versions"
        self.trash_dir = self.root / ".trash"
        self.session_path = self.root / "session.json"
        self._lock = threading.Lock()

    def _project_path(self, slug: str) -> Path:
        return self.projects_dir / f"{_validate_slug(slug)}{_PROJECT_SUFFIX}"

    def _versions_path(self, slug: str) -> Path:
        return self.versions_dir / _validate_slug(slug)

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        write_json_atomic(path, payload)

    def _read_document(self, path: Path) -> SavedWorkflowDocument:
        if not path.is_file():
            raise ProjectNotFound(f"Project document not found: {path.name}")
        try:
            return SavedWorkflowDocument.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            message = exc.errors()[0].get("msg", "validation failed")
            raise InvalidProjectDocument(
                f"Invalid workflow document {path.name}: {message}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError(f"Could not read {path.name}: {exc}") from exc

    def _document(
        self, name: str, workflow: SavedWorkflowDocument
    ) -> SavedWorkflowDocument:
        return workflow.model_copy(update={"name": name, "saved_at": _timestamp()})

    def _version_count(self, slug: str) -> int:
        versions_dir = self._versions_path(slug)
        if not versions_dir.is_dir():
            return 0
        return len(list(versions_dir.glob(f"*{_PROJECT_SUFFIX}")))

    def _summary(self, slug: str, workflow: SavedWorkflowDocument, path: Path) -> ProjectSummary:
        saved_at = workflow.saved_at
        if not saved_at:
            saved_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
                timespec="seconds"
            )
        return ProjectSummary(
            slug=slug,
            name=workflow.name or slug,
            saved_at=saved_at,
            node_count=len(workflow.nodes),
            edge_count=len(workflow.edges),
            version_count=self._version_count(slug),
        )

    def list_projects(self) -> list[ProjectSummary]:
        """List valid project documents, newest save first."""
        if not self.projects_dir.is_dir():
            return []
        summaries: list[ProjectSummary] = []
        for path in self.projects_dir.glob(f"*{_PROJECT_SUFFIX}"):
            slug = path.name[: -len(_PROJECT_SUFFIX)]
            try:
                workflow = self._read_document(path)
                summaries.append(self._summary(slug, workflow, path))
            except (
                InvalidProjectDocument,
                InvalidProjectSlug,
                ProjectNotFound,
                OSError,
            ):
                continue
        return sorted(
            summaries,
            key=lambda item: (item.saved_at, item.name.lower()),
            reverse=True,
        )

    def get_project(self, slug: str) -> ProjectDetail:
        path = self._project_path(slug)
        workflow = self._read_document(path)
        summary = self._summary(slug, workflow, path)
        return ProjectDetail(**summary.model_dump(), workflow=workflow)

    def _snapshot(self, slug: str, path: Path) -> None:
        if not path.is_file():
            return
        versions_dir = self._versions_path(slug)
        versions_dir.mkdir(parents=True, exist_ok=True)
        destination = versions_dir / f"{_version_id()}{_PROJECT_SUFFIX}"
        shutil.copy2(path, destination)
        versions = sorted(
            versions_dir.glob(f"*{_PROJECT_SUFFIX}"),
            key=lambda candidate: candidate.name,
            reverse=True,
        )
        for stale in versions[PROJECT_VERSION_RETENTION:]:
            stale.unlink(missing_ok=True)

    def create_project(
        self,
        name: str,
        workflow: SavedWorkflowDocument,
        *,
        overwrite: bool = False,
    ) -> tuple[ProjectSummary, bool]:
        """Create a project and return ``(summary, created)``."""
        cleaned = name.strip()
        slug = slugify_project_name(cleaned)
        path = self._project_path(slug)
        with self._lock:
            exists = path.is_file()
            if exists and not overwrite:
                raise ProjectAlreadyExists(
                    f"A project named '{cleaned}' already exists"
                )
            if exists:
                self._snapshot(slug, path)
            document = self._document(cleaned, workflow)
            self._write_json(
                path, document.model_dump(mode="json", by_alias=True)
            )
            return self._summary(slug, document, path), not exists

    def update_project(
        self, slug: str, name: str, workflow: SavedWorkflowDocument
    ) -> ProjectSummary:
        """Overwrite an existing project, snapshotting its previous contents."""
        cleaned = name.strip()
        path = self._project_path(slug)
        with self._lock:
            if not path.is_file():
                raise ProjectNotFound(f"Project not found: {slug}")
            self._snapshot(slug, path)
            document = self._document(cleaned, workflow)
            self._write_json(
                path, document.model_dump(mode="json", by_alias=True)
            )
            return self._summary(slug, document, path)

    def delete_project(self, slug: str) -> None:
        """Move a project and its snapshots into the workspace trash."""
        path = self._project_path(slug)
        versions_dir = self._versions_path(slug)
        with self._lock:
            if not path.is_file() and not versions_dir.is_dir():
                raise ProjectNotFound(f"Project not found: {slug}")
            target = self.trash_dir / f"{slug}-{_version_id()}"
            target.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                shutil.move(str(path), str(target / path.name))
            if versions_dir.is_dir():
                shutil.move(str(versions_dir), str(target / "versions"))

    def list_versions(self, slug: str) -> list[ProjectVersionInfo]:
        """List retained snapshots, newest first."""
        versions_dir = self._versions_path(slug)
        if not versions_dir.is_dir():
            raise ProjectNotFound(f"No versions found for project: {slug}")
        versions: list[ProjectVersionInfo] = []
        for path in versions_dir.glob(f"*{_PROJECT_SUFFIX}"):
            version = path.name[: -len(_PROJECT_SUFFIX)]
            try:
                workflow = self._read_document(path)
            except (InvalidProjectDocument, ProjectNotFound, OSError):
                continue
            versions.append(
                ProjectVersionInfo(
                    version=version,
                    saved_at=workflow.saved_at,
                    node_count=len(workflow.nodes),
                    edge_count=len(workflow.edges),
                    size_bytes=path.stat().st_size,
                )
            )
        return sorted(versions, key=lambda item: item.version, reverse=True)

    def get_version(self, slug: str, version: str) -> ProjectVersionDetail:
        """Load one retained project snapshot."""
        safe_version = _validate_version(version)
        path = self._versions_path(slug) / f"{safe_version}{_PROJECT_SUFFIX}"
        workflow = self._read_document(path)
        return ProjectVersionDetail(
            version=safe_version,
            saved_at=workflow.saved_at,
            node_count=len(workflow.nodes),
            edge_count=len(workflow.edges),
            size_bytes=path.stat().st_size,
            workflow=workflow,
        )

    def save_session(self, request: WorkspaceSessionRequest) -> WorkspaceSession:
        """Persist the latest canvas for recovery after an abrupt close."""
        name = (request.project_name or request.workflow.name or "").strip() or None
        workflow = request.workflow.model_copy(update={"name": name})
        session = WorkspaceSession(
            saved_at=_timestamp(),
            project_slug=request.project_slug,
            project_name=name,
            workflow=workflow,
        )
        with self._lock:
            self._write_json(
                self.session_path,
                session.model_dump(mode="json", by_alias=True),
            )
        return session

    def load_session(self) -> WorkspaceSession | None:
        """Return the recovery session, or ``None`` when unavailable/corrupt."""
        if not self.session_path.is_file():
            return None
        try:
            return WorkspaceSession.model_validate_json(
                self.session_path.read_text(encoding="utf-8")
            )
        except (ValidationError, OSError):
            return None

    def clear_session(self) -> None:
        """Remove the recovery session after the user discards it."""
        with self._lock:
            self.session_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def get_project_store() -> ProjectStore:
    """Return the process-wide project store."""
    return ProjectStore()
