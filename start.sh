#!/usr/bin/env sh
# ComplexUI launcher for macOS and Linux (the Windows equivalent is start.cmd).
set -u
cd "$(dirname "$0")"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if ! command -v uv >/dev/null 2>&1; then
    echo "[start.sh] 'uv' was not found on PATH." >&2
    echo "[start.sh] Install it from https://docs.astral.sh/uv/ and run this file again." >&2
    exit 2
fi

if ! command -v node >/dev/null 2>&1; then
    echo "[start.sh] 'node' was not found on PATH." >&2
    echo "[start.sh] Install Node.js 20.19+ or 22.12+ from https://nodejs.org/ and run this file again." >&2
    exit 2
fi

# --no-dev keeps the first launch fast: serving the UI does not need the
# optional ML/test toolchain. `check` mode still syncs dev dependencies.
exec uv run --directory backend --no-dev python launch.py "$@"
