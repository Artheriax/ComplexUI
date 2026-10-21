# ComplexUI

Visual, code-first node-graph IDE for Machine Learning (Scikit-Learn, XGBoost,
LightGBM, Polars, PyTorch). Every node is live, editable Python; the backend
parses each node's code into typed sockets and widgets, executes the DAG in
topological order with content-addressable caching, and streams telemetry over
WebSocket. Untrusted code must pass a diff-based approval gate before it runs.
Each run executes in a disposable child process with configurable node/run
timeouts and a best-effort memory limit, so hangs, crashes and memory blowups do
not take down the server.

Node code can use `print()` and Jupyter-style `display()`: stdout, DataFrames,
HTML, images, matplotlib/seaborn figures and Plotly figures stream back as rich
outputs, rendered inline on the node and in full in the Execution panel.
Workflow outputs stay out of `.cuigraph.json`; terminal runs instead keep bounded
previews and rich outputs in workspace run-history records, alongside statuses,
timings and errors.

The bundled catalog ships **64 node templates** (56 production nodes across 9
categories plus 8 example/synthetic loaders that stay hidden and power the
bundled workflows) and **12 ready-to-run example workflows**. Browser-style
workflow tabs keep multiple graphs open at once, each with its own undo history,
execution output and project identity; open tabs are restored after a reload.
the left rail expands **Nodes**, **Workflows**, **Templates**, and **Datasets** drawers: drag a template, open a bundled workflow or saved
workflow in a new tab, or upload a dataset for the `File` widgets used by the
CSV/Parquet/Excel/NDJSON loaders. Int/Float/Bool/String **Value** nodes wire
directly into widget parameters; numeric values have `<`/`>` step buttons
driven by an editable `step` field. Widgets annotated with
`typing.Literal[...]` render fixed-choice dropdowns, and their
defaults/overrides/wired values are validated before execution. Datasets can be
previewed and deleted from the sidebar. Press `Ctrl+S` to name and save the
active workflow; later saves are silent, and overwrites retain up to 30 version
snapshots. Selected nodes can be turned into reusable nested subgraphs with
automatic boundary ports; definitions have one shared Inputs and Outputs
manager pair, open in a nested canvas, and can be instantiated from the Nodes
library. Managers provide Add input/output buttons plus draggable + handles for
creating additional typed ports; dropping a + handle on empty canvas creates a
compatible node already wired. Removing the last instance also removes the
definition. The right-click **File** submenu downloads the workflow as a
`.cuigraph.json`, a runnable PEP 723 Python script, or a Jupyter notebook, and
also contains Save, Import and Clear; exports reuse the same transpiler and
topological order as the executor, with nested instances flattened to
path-scoped ids (`sub:<path>/<id>`).

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
| `check` | full verification gate: Ruff, backend pytest, ESLint, frontend typecheck, vitest, build |

Browser E2E is a separate gate: install Chromium once with
`cd frontend && npx playwright install chromium`, then run `npm run e2e`.
GitHub Actions runs backend, frontend, and Playwright checks on every push and
PR; `.github/workflows/ml-overlay.yml` additionally exercises torch, xgboost,
and lightgbm weekly or on demand.

Useful flags: `--backend-port`, `--frontend-port`, `--host`, `--log-level`,
`--log-format` (`text` or `json`), `--log-dir`, `--no-log-file`, `--no-reload`,
`--no-color`, `--open` (e.g. `start.cmd dev --open`).

By default the backend binds loopback and accepts only loopback `Host`/`Origin`
headers. Binding beyond loopback requires `--allow-network`, which disables that
protection: the API has no authentication and executes Python code. Use
`COMPLEXUI_ALLOWED_HOSTS` for explicit custom hostnames instead of wildcards.

Console logs are colorized and prefixed `[backend]` / `[frontend]`; every step —
node starts, cache hits, timings, dependency installs, cancellations — is
narrated in the terminal. Set `COMPLEXUI_LOG_LEVEL=DEBUG` for more detail,
`COMPLEXUI_LOG_FORMAT=json` (or `--log-format json`) for structured application
logs, or `NO_COLOR=1` / `--no-color` for plain output.

Daily plain-text logs are written to `backend/workspace/logs/` and retained for
14 days; each execution also gets `backend/workspace/logs/runs/<run_id>.txt`.
Set `COMPLEXUI_LOG_DIR` to another directory or `0` to disable file logs, or use
launcher flags `--log-dir PATH` / `--no-log-file`. Independently of logging, the
50 newest terminal runs are retained as JSON records in `workspace/runs/` and
are browsable from the Execution panel.

Execution isolation uses `COMPLEXUI_NODE_TIMEOUT_SECONDS` (default 600),
`COMPLEXUI_RUN_TIMEOUT_SECONDS` (default 3600), and
`COMPLEXUI_MAX_MEMORY_MB` (default 4096); set any to `0` to disable that guard.
This is crash/resource containment, not a filesystem or network sandbox: node
code still runs with the backend user's permissions. `File` widgets are the
exception and may only resolve files inside the workspace.

The default workspace is `backend/workspace`; set `COMPLEXUI_WORKSPACE` to move
it. Named projects live in `workflows/<slug>.cuigraph.json`, including reusable
subgraph definitions and nested canvases, overwrite snapshots in
`.versions/<slug>/`, close/visibility recovery data in `session.json`, run
history in `runs/`, and uploaded datasets in `data/uploads/`. Deleted projects
move to `.trash/`. Logs
follow the workspace unless `COMPLEXUI_LOG_DIR` is set explicitly. `File`
widgets store workspace-relative paths such as `data/uploads/iris.csv`;
exported scripts and notebooks therefore expect to run from the workspace root
(or need those paths adjusted). `File` widgets reject absolute paths and
symlink escapes outside the workspace.

## Layout

- `backend/` — FastAPI service, node-code transpiler, DAG executor, template
  registry, workflow catalog, and the `launch.py` launcher (see
  `backend/README.md` for the API).
- `frontend/` — React 19 + TypeScript + XYFlow canvas, Monaco code editor,
  Zustand stores, multi-window workflow tabs, categorized node/workflow
  libraries, dataset upload and preview UI, undo/redo, multi-node clipboard,
  selection alignment, resizable frame groups, reusable nested subgraphs,
  execution telemetry and run-history UI (see `frontend/README.md`).

