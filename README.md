# ComplexUI

Visual, code-first node-graph IDE for Machine Learning (Scikit-Learn, Polars,
PyTorch). Every node is live, editable Python; the backend parses each node's
code into typed sockets and widgets, executes the DAG in topological order with
content-addressable caching, and streams telemetry over WebSocket. Untrusted
code must pass a diff-based approval gate before it runs.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12+) — backend runtime and tooling
- Node.js 20+ with npm — frontend build tooling

## Quickstart

```cmd
start.cmd
```

That launches **dev mode** (backend with auto-reload on `http://127.0.0.1:8000`,
Vite dev server on `http://localhost:5173`) and opens nothing by default; visit
the UI URL printed at the end of startup. `Ctrl+C` stops both processes cleanly.

On a fresh clone the launcher bootstraps everything by itself: `uv` creates and
syncs `backend/.venv`, and `npm install` runs automatically in `frontend/` the
first time it is needed — no manual setup beyond having `uv` and Node.js
installed.

Other modes via the launcher (`start.cmd <mode>` or `uv run python launch.py <mode>`
from `backend/`):

| Mode | What it does |
| --- | --- |
| `dev` (default) | backend + Vite dev server, both with hot reload |
| `backend` | backend only |
| `frontend` | Vite dev server only |
| `prod` | builds the frontend and serves everything from the backend on one port |
| `check` | full verification gate: backend pytest, frontend typecheck, vitest, build |

Useful flags: `--backend-port`, `--frontend-port`, `--host`, `--log-level`,
`--no-reload`, `--no-color`, `--open` (e.g. `start.cmd dev --open`).

Console logs are colorized and prefixed `[backend]` / `[frontend]`; every step —
node starts, cache hits, timings, dependency installs, cancellations — is
narrated in the terminal. Set `COMPLEXUI_LOG_LEVEL=DEBUG` for more detail, or
`NO_COLOR=1` / `--no-color` for plain output.

## Layout

- `backend/` — FastAPI service, node-code transpiler, DAG executor, template
  registry, and the `launch.py` launcher (see `backend/README.md` for the API).
- `frontend/` — React 19 + TypeScript + XYFlow canvas, Monaco code editor,
  Zustand store, and execution telemetry UI (see `frontend/README.md`).
