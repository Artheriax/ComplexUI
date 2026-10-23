"""Platform file-action helpers for generated outputs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import os_actions


def _capture_popen(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[list[str]],
) -> None:
    def fake_popen(command: list[str], **_: object) -> None:
        calls.append(command)

    monkeypatch.setattr(os_actions.subprocess, "Popen", fake_popen)


def test_open_path_uses_startfile_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        os,
        "startfile",
        lambda value: opened.append(value),
        raising=False,
    )
    path = tmp_path / "result.png"

    os_actions.open_path(path)

    assert opened == [str(path)]


def test_reveal_path_uses_explorer_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    _capture_popen(monkeypatch, calls)
    path = tmp_path / "result.png"

    os_actions.reveal_path(path)

    assert calls == [["explorer", f"/select,{path}"]]


def test_open_and_reveal_on_macos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "darwin")
    _capture_popen(monkeypatch, calls)
    path = tmp_path / "result.png"

    os_actions.open_path(path)
    os_actions.reveal_path(path)

    assert calls == [["open", str(path)], ["open", "-R", str(path)]]


def test_open_and_reveal_on_linux(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "linux")
    _capture_popen(monkeypatch, calls)
    path = tmp_path / "result.png"

    os_actions.open_path(path)
    os_actions.reveal_path(path)

    assert calls == [["xdg-open", str(path)], ["xdg-open", str(tmp_path)]]


def test_path_action_errors_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def failing_popen(command: list[str], **_: object) -> None:
        raise OSError("no opener")

    monkeypatch.setattr(os_actions.subprocess, "Popen", failing_popen)

    with pytest.raises(os_actions.PathActionError):
        os_actions.open_path(tmp_path / "result.png")


def test_startfile_errors_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def failing_startfile(value: str) -> None:
        raise OSError(f"cannot open {value}")

    monkeypatch.setattr(os, "startfile", failing_startfile, raising=False)

    with pytest.raises(os_actions.PathActionError):
        os_actions.open_path(tmp_path / "result.png")


def test_missing_startfile_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delattr(os, "startfile", raising=False)

    with pytest.raises(os_actions.PathActionError):
        os_actions.open_path(tmp_path / "result.png")


def test_subprocess_error_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def failing_popen(command: list[str], **_: object) -> None:
        raise subprocess.SubprocessError("broken")

    monkeypatch.setattr(os_actions.subprocess, "Popen", failing_popen)

    with pytest.raises(os_actions.PathActionError):
        os_actions.reveal_path(tmp_path / "result.png")


def _completed(path: str, code: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["dialog"], code, path, stderr)


def test_pick_folder_uses_native_windows_dialog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return _completed(str(tmp_path))

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os_actions.subprocess, "run", fake_run)

    selected = os_actions.pick_folder(str(tmp_path))

    assert selected == str(tmp_path.resolve())
    assert calls[0][0][0] == "powershell.exe"
    assert "-STA" in calls[0][0]
    environment = calls[0][1]["env"]
    assert isinstance(environment, dict)
    assert environment[os_actions._FOLDER_INITIAL_ENV] == str(tmp_path)


def test_pick_folder_cancel_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        os_actions.subprocess,
        "run",
        lambda command, **kwargs: _completed(""),
    )

    assert os_actions.pick_folder() is None


def test_pick_folder_linux_zenity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return _completed(str(tmp_path))

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os_actions.shutil, "which", lambda name: "zenity" if name == "zenity" else None)
    monkeypatch.setattr(os_actions.subprocess, "run", fake_run)

    assert os_actions.pick_folder() == str(tmp_path.resolve())
    assert "--directory" in calls[0]


def test_pick_folder_requires_a_dialog_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os_actions.shutil, "which", lambda name: None)

    with pytest.raises(os_actions.PathActionError, match="zenity or kdialog"):
        os_actions.pick_folder()


def test_pick_folder_rejects_a_busy_dialog() -> None:
    with os_actions._FOLDER_DIALOG_LOCK:
        with pytest.raises(os_actions.FolderDialogBusy):
            os_actions.pick_folder()


def test_pick_folder_wraps_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os_actions.shutil, "which", lambda name: "zenity" if name == "zenity" else None)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(os_actions.subprocess, "run", fake_run)

    with pytest.raises(os_actions.PathActionError, match="timed out"):
        os_actions.pick_folder()


def test_pick_folder_rejects_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        os_actions.subprocess,
        "run",
        lambda command, **kwargs: _completed(str(target)),
    )

    with pytest.raises(os_actions.PathActionError, match="not a directory"):
        os_actions.pick_folder()
