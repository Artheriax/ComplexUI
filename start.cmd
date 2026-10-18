@echo off
setlocal
pushd "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [start.cmd] 'uv' was not found on PATH.
    echo [start.cmd] Install it from https://docs.astral.sh/uv/ and retry.
    popd
    exit /b 2
)

uv run --directory backend python launch.py %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
