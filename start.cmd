@echo off
setlocal
pushd "%~dp0"

rem Keep Python's console output UTF-8 so the launcher banner renders on all
rem Windows codepages.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

where uv >nul 2>nul
if errorlevel 1 (
    echo [start.cmd] 'uv' was not found on PATH.
    echo [start.cmd] Install it from https://docs.astral.sh/uv/ and run this file again.
    popd
    exit /b 2
)

where node >nul 2>nul
if errorlevel 1 (
    echo [start.cmd] 'node' was not found on PATH.
    echo [start.cmd] Install Node.js 20.19+ or 22.12+ from https://nodejs.org/ and run this file again.
    popd
    exit /b 2
)

rem --no-dev keeps the first launch fast: serving the UI does not need the
rem optional ML/test toolchain. `check` mode still syncs dev dependencies.
uv run --directory backend --no-dev python launch.py %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
