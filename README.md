# ComplexUI

Visual, code-first node-graph IDE for Machine Learning (Scikit-Learn, XGBoost,
LightGBM, Polars, PyTorch, torchvision, Ultralytics). Every node is live, editable Python; the backend
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

The vision path adds typed `IMAGE`, `VIDEO`, `MEDIA`, model, detection, mask and
classification sockets. Uploaded images and videos stream inline (with HTTP
range support) from repo-root `inputs/`, generated images/videos are served from
repo-root `outputs/`, and image, video, model, detection, mask and
classification previews render directly on nodes. torchvision covers image
classification, detection and segmentation plus sampled-frame video
classification; Ultralytics YOLO covers image/video detection and segmentation.
Video inference is monolithic per backend/task and writes browser-playable H.264
outputs under repo-root `outputs/`.

The repo-root **`models/`** library is mandatory and ships with `yolo/`,
`torchvision/`, and `classification/` folders. Custom type folders are discovered
too. Node code can declare a local picker with
`local_model: ModelSelector = ModelSelector(type="yolo")`; the widget lists files
from `models/yolo`, while an empty selection keeps the existing built-in YOLO or
torchvision model as fallback. Selections are repo-relative, block execution when
stale, and contribute their size/mtime fingerprint to cache invalidation. Local
formats are `.pt`, `.pth`, `.onnx`, and `.safetensors`; support is
backend-specific. Models Library is a read-only Assets tab and auto-refreshes
while open (press `R` to refresh immediately). Only add trusted model files:
loading `.pt`/pickle models is not a sandbox, so SafeTensors/ONNX are preferred.

The **System** settings tab controls a draggable, background-less system monitor
showing CPU/GPU utilization, VRAM/RAM usage, and GPU temperature. NVIDIA GPU
metrics come from NVML. The same tab shows read-only runtime diagnostics.
**Execution** settings control whether the Execution panel opens on run, whether
Canvas Clear asks for confirmation, and display the active node/run timeouts and
memory limit. The monitor position, visibility, refresh interval, and both
execution toggles persist in browser storage.

Nodes can stream coarse step progress through `NODE_PROGRESS`; the node footer
and Execution panel show the percentage and message, and **Stop** cancels
cooperatively inside a running node when it calls `raise_if_cancelled()`.
Save Media has an inline folder/filename editor with a native Browse dialog,
`%date%`/`%time%`/`%counter%`/`%model%`/`%width%`/`%height%`/`%batch%` tokens,
a live path preview, format/quality controls and batch indices. It defaults to
`outputs/<YYYY-MM-DD>/`, but users can pick any absolute folder; outputs files
appear in the Assets **Outputs** tab, while external files are reached through
the node's preview/open/copy/show-folder actions. External actions are limited
to files recorded by a completed run.

The development environment pins `torch`/`torchvision` to the cu130 index on
Windows and the CPU index on Linux; macOS resolves the platform's PyPI wheels.
Node PEP 723 installs follow `COMPLEXUI_ACCELERATOR` (`auto`, `cpu`, or `cuda`)
or the launcher's `--accelerator` flag. `cuda` uses the cu130 wheel index
(`COMPLEXUI_TORCH_CUDA_INDEX` overrides it), and `auto` detects an NVIDIA GPU
through NVML. Switching between CPU and CUDA torch builds reinstalls the torch
family, and large wheel downloads use `COMPLEXUI_INSTALL_TIMEOUT_SECONDS`
(default 1200).

The bundled catalog ships **115 node templates** (98 production nodes across
11 categories plus 17 hidden example and legacy aliases) and **27 ready-to-run
example workflows**. Image batches use `IMAGE_BATCH` sockets: any node with a
scalar `IMAGE` input runs once per image, while `MEDIA`/`IMAGE_BATCH` inputs
receive the whole list. `Load Images From Folder` feeds a bounded folder batch;
`Resize Image` and `Resize Latent` expose mode-specific dimensions, side,
megapixel and multiplier controls, and `Preview Mask` is batch-aware. Resolution
and Empty Latent Image switch between explicit dimensions and megapixel presets
from Preview through UHD, with a custom megapixel field. The five-input `Switch`
passes one named input through. Browser-style
workflow tabs keep multiple graphs open at once, each with its own undo history,
execution output and project identity; open tabs are restored after a reload.
The left rail expands **Nodes**, **Workflows**, **Templates**, and **Assets** drawers: drag a template, open a bundled workflow or saved
workflow in a new tab, or upload a table, image or video for the
`File`/`ImageFile`/`VideoFile` widgets used by loaders, or inspect the repo
`models/` library under the Models tab. Int/Float/Bool/String **Value** nodes wire
directly into widget parameters; numeric values have `<`/`>` step buttons
driven by an editable `step` field. Widgets annotated with
`typing.Literal[...]` render fixed-choice dropdowns, and their
defaults/overrides/wired values are validated before execution. Table uploads
can be previewed, and every asset can be searched or deleted from the sidebar. Press `Ctrl+S` to name and save the
active workflow; later saves are silent, and overwrites retain up to 30 version
snapshots. Selected nodes can be turned into reusable nested subgraphs with
automatic boundary ports; definitions have one shared Inputs and Outputs
manager pair, open in a nested canvas, and can be instantiated from the Nodes
library. Managers provide Add input/output buttons plus draggable + handles for
creating additional typed ports; dropping a + handle on empty canvas creates a
compatible node already wired. Removing the last instance also removes the
definition. Drag a connected input dot to retarget its wire to another
compatible input; the source output stays fixed and the wire body itself is not
draggable. Empty drops delete the wire and open node search; choosing a node
creates a new target and wire, while invalid drops restore the wire.
`Shift`-dragging an output moves its whole wire bundle to another
same-typed output (invalid/cancelled drops restore it; empty drops open node
search).
Incompatible or occupied handles grey out while dragging. Runnable nodes and
subgraph instances are resizable from the bottom-right; sizes persist, parameter
controls stay fixed, and media previews expand to fill the node body. The
right-click **File** submenu downloads the workflow as a
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
| `check` | full verification gate: Ruff, basedpyright, backend pytest with coverage, ESLint, frontend typecheck, vitest, build |

Browser E2E is a separate gate: install Chromium once with
`cd frontend && npx playwright install chromium`, then run `npm run e2e`.
GitHub Actions runs backend, frontend, and Playwright checks on every push and
PR; `.github/workflows/ml-overlay.yml` additionally exercises torch,
torchvision, Ultralytics, xgboost, and lightgbm weekly or on demand.

The Ultralytics node templates depend on the separately installed
`ultralytics` package, which is **AGPL-3.0**. They are isolated from the
torchvision templates so workflows that do not use YOLO do not install it.

Useful flags: `--backend-port`, `--frontend-port`, `--host`, `--accelerator`
(`auto`, `cpu`, or `cuda`), `--log-level`, `--log-format` (`text` or `json`),
`--log-dir`, `--no-log-file`, `--no-reload`, `--no-color`, `--open` (e.g.
`start.cmd dev --open --accelerator cuda`).

By default the backend binds loopback and accepts only loopback `Host`/`Origin`
headers. Binding beyond loopback requires `--allow-network`, which disables that
protection: the API has no authentication and executes Python code. Use
`COMPLEXUI_ALLOWED_HOSTS` for explicit custom hostnames instead of wildcards.

Console logs are colorized and prefixed `[backend]` / `[frontend]`; every step —
node starts, cache hits, timings, dependency installs, cancellations — is
narrated in the terminal. Startup prints a diagnostic banner with git/runtime
versions, paths, hardware and active log destinations. During a run, node
progress is shown as one updating line at the bottom of the console; successful
polling requests for `/api/system/metrics`, `/api/status` and `/api/health` are
filtered from the access log. Set `COMPLEXUI_LOG_LEVEL=DEBUG` for more detail,
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
`COMPLEXUI_MAX_MEMORY_MB` (default 12288); set any to `0` to disable that guard.
This is crash/resource containment, not a filesystem or network sandbox: node
code still runs with the backend user's permissions. `File` widgets are the
exception and may only resolve files inside the workspace.

The default workspace is `backend/workspace`; set `COMPLEXUI_WORKSPACE` to move
it. Named projects live in `workflows/<slug>.cuigraph.json`, including reusable
subgraph definitions and nested canvases, overwrite snapshots in
`.versions/<slug>/`, close/visibility recovery data in `session.json`, run
history in `runs/`, uploaded assets in repo-root `inputs/`
(`COMPLEXUI_INPUT_DIR` overrides it), and generated media in repo-root
`outputs/` (`COMPLEXUI_OUTPUT_DIR` overrides it). Deleted projects
move to `.trash/`. Logs
follow the workspace unless `COMPLEXUI_LOG_DIR` is set explicitly. `File`
widgets store input paths such as `inputs/iris.csv` and can
read generated files through `outputs/<folder>/<name>`; exported scripts and notebooks
therefore expect those paths to remain available. File access rejects paths
outside the workspace, inputs, and generated-output roots.

The shared model library is repo-root `models/`, not workspace-relative; set
`COMPLEXUI_MODELS_DIR` to use another location. `ModelSelector` widget values
are stored as `models/<type>/<file>` and resolve against that root.

## Layout

- `backend/` — FastAPI service, node-code transpiler, DAG executor, template
  registry, workflow catalog, and the `launch.py` launcher (see
  `backend/README.md` for the API).
- `frontend/` — React 19 + TypeScript + XYFlow canvas, Monaco code editor,
  Zustand stores, multi-window workflow tabs, categorized node/workflow
  libraries, dataset upload and preview UI, undo/redo, multi-node clipboard,
  selection alignment, resizable frame groups, reusable nested subgraphs,
  execution telemetry and run-history UI (see `frontend/README.md`).


