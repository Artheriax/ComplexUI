"""Best-effort platform actions for generated output files."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

__all__ = [
    "FolderDialogBusy",
    "PathActionError",
    "open_path",
    "pick_folder",
    "reveal_path",
]

_FOLDER_DIALOG_TIMEOUT_SECONDS = 300
_FOLDER_DIALOG_LOCK = threading.Lock()
_FOLDER_INITIAL_ENV = "COMPLEXUI_PICK_FOLDER_INITIAL"
_WINDOWS_FOLDER_SCRIPT = """
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose a folder for Save Media'
$dialog.ShowNewFolderButton = $true
$initial = $env:COMPLEXUI_PICK_FOLDER_INITIAL
if ($initial -and (Test-Path -LiteralPath $initial -PathType Container)) {
    $dialog.SelectedPath = $initial
}
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($dialog.SelectedPath)
}
"""
_MACOS_FOLDER_SCRIPT = """
set initialPath to system attribute "COMPLEXUI_PICK_FOLDER_INITIAL"
try
    if initialPath is not "" then
        set chosen to choose folder with prompt "Choose a folder for Save Media" default location (POSIX file initialPath)
    else
        set chosen to choose folder with prompt "Choose a folder for Save Media"
    end if
    return POSIX path of chosen
on error number -128
    return ""
end try
"""


class PathActionError(Exception):
    """The requested file action is unavailable on this platform."""


class FolderDialogBusy(PathActionError):
    """Another native folder dialog is already open."""


def _dialog_environment(initial: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    if initial:
        environment[_FOLDER_INITIAL_ENV] = initial
    else:
        environment.pop(_FOLDER_INITIAL_ENV, None)
    return environment


def _run_dialog(
    command: list[str],
    *,
    initial: str | None,
    cancel_codes: set[int] | None = None,
) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed commands, path passed via env
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_FOLDER_DIALOG_TIMEOUT_SECONDS,
            env=_dialog_environment(initial),
        )
    except subprocess.TimeoutExpired as exc:
        raise PathActionError("The folder dialog timed out.") from exc
    except OSError as exc:
        raise PathActionError(f"Could not open the folder dialog: {exc}") from exc
    if completed.returncode != 0:
        if (
            completed.returncode in (cancel_codes or set())
            and not completed.stdout.strip()
            and not completed.stderr.strip()
        ):
            return ""
        message = completed.stderr.strip() or "The folder dialog was closed."
        raise PathActionError(message)
    return completed.stdout.strip()


def _pick_folder_windows(initial: str | None) -> str | None:
    output = _run_dialog(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", _WINDOWS_FOLDER_SCRIPT],
        initial=initial,
    )
    return output or None


def _pick_folder_macos(initial: str | None) -> str | None:
    output = _run_dialog(
        ["osascript", "-e", _MACOS_FOLDER_SCRIPT],
        initial=initial,
    )
    return output or None


def _pick_folder_linux(initial: str | None) -> str | None:
    if shutil.which("zenity"):
        command = [
            "zenity",
            "--file-selection",
            "--directory",
            "--title=Choose a folder for Save Media",
        ]
        if initial:
            command.append(f"--filename={initial.rstrip(os.sep)}{os.sep}")
        output = _run_dialog(command, initial=initial, cancel_codes={1})
        return output or None
    if shutil.which("kdialog"):
        command = [
            "kdialog",
            "--getexistingdirectory",
            initial or str(Path.home()),
            "--title",
            "Choose a folder for Save Media",
        ]
        output = _run_dialog(command, initial=initial, cancel_codes={1})
        return output or None
    raise PathActionError(
        "No native folder dialog is available. Install zenity or kdialog, "
        "or type the folder path."
    )


def pick_folder(initial: str | None = None) -> str | None:
    """Open a native folder dialog and return the selected directory."""
    if not _FOLDER_DIALOG_LOCK.acquire(blocking=False):
        raise FolderDialogBusy("A folder dialog is already open.")
    try:
        hint = initial.strip() if isinstance(initial, str) else ""
        if hint and not Path(hint).expanduser().is_dir():
            hint = ""
        if sys.platform == "win32":
            selected = _pick_folder_windows(hint or None)
        elif sys.platform == "darwin":
            selected = _pick_folder_macos(hint or None)
        else:
            selected = _pick_folder_linux(hint or None)
        if not selected:
            return None
        path = Path(selected).expanduser()
        if not path.is_dir():
            raise PathActionError("The selected path is not a directory.")
        return str(path.resolve())
    finally:
        _FOLDER_DIALOG_LOCK.release()


def _run(command: list[str]) -> None:
    try:
        subprocess.Popen(  # noqa: S603 - fixed commands with a validated path
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PathActionError(f"Could not launch the file manager: {exc}") from exc


def open_path(path: Path) -> None:
    """Open a file with the platform's default application."""
    if sys.platform == "win32":
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise PathActionError("Opening files is not supported on this platform.")
        try:
            startfile(str(path))
        except OSError as exc:
            raise PathActionError(f"Could not open the file: {exc}") from exc
        return
    if sys.platform == "darwin":
        _run(["open", str(path)])
        return
    _run(["xdg-open", str(path)])


def reveal_path(path: Path) -> None:
    """Show a file in the platform's file manager."""
    if sys.platform == "win32":
        _run(["explorer", f"/select,{path}"])
        return
    if sys.platform == "darwin":
        _run(["open", "-R", str(path)])
        return
    _run(["xdg-open", str(path.parent)])
