# ComplexUI Backend

FastAPI service and node-code toolchain for ComplexUI.

- `app/schemas.py` — Pydantic v2 models for node schemas, workflow graphs, and execution telemetry events.
- `app/transpiler.py` — node-code DSL transpiler (PEP 723 extraction, typed
  returns, input/widget/output extraction, function wrapping). Assigned
  `typing.Literal[...]` annotations become fixed-choice widgets: the schema
  carries an `options` list, and defaults, workflow overrides and wired values
  are validated against it.
- `app/executor.py` — topological DAG executor with a bounded
  content-addressable LRU cache (64 outputs / 32 MiB), trust gating,
  cancellation checks and WebSocket-ready event streaming. `File` widgets are
  resolved strictly inside the workspace before node code runs; `ModelSelector`
  widgets resolve inside the repo model root and are fingerprinted into the
  cache key.
- `app/execution_host.py` — disposable child-process execution, cache/approval
  transfer, wall-clock and memory guards, process-tree termination and recovery.
- `app/execution_worker.py` — child-process entry point that runs the executor
  and streams events/cache snapshots back to the server process.
- `app/datasets.py` — repo-root table/image/video uploads with sanitized,
  collision-safe names, atomic streaming writes (100 MiB tables, 500 MiB
  media), listing/deletion, bounded Polars table previews, raw media paths, and
  generated-file resolution under repo-root `outputs/`.
- `app/models.py` — metadata-only scanner for the repo-root `models/` library.
  It creates `yolo/`, `torchvision/`, and `classification/`, discovers custom
  type folders, validates paths/extensions/symlinks, and returns size + mtime
  fingerprints without importing model files. `COMPLEXUI_MODELS_DIR` overrides
  the root.
- `app/transpiler.py` recognizes `local_model: ModelSelector =
  ModelSelector(type="yolo")` as a model widget and records `model_type` on the
  schema.
- `app/exporter.py` — standalone PEP 723 Python and nbformat 4.5 notebook
  generation using the same transpiler/DAG plan (scoped exports and deduped
  dependencies included).
- `app/output_capture.py` — best-effort capture of `print`, `display`, tables,
  images, HTML and Plotly figure side effects for each node.
- `app/env_manager.py` — PEP 723 dependency checks and a cancellable
  `uv pip install` runner (10-minute timeout; failures emit `NODE_ERROR`).
- `app/security.py` — SHA-256 template registry and session approval tracking;
  flattened nested nodes may supply an `approval_id` to retain node-scoped trust.
- `app/network_security.py` — loopback-first `Host`/WebSocket `Origin`
  validation and a request-size guard with path-prefix overrides (used to allow
  larger uploads at `/api/uploads`).
- `app/template_catalog.py` — display metadata (title, category, description,
  palette visibility) for the official node catalog. Example dataset loaders
  (iris/wine/breast-cancer/diabetes/digits), the synthetic classification,
  messy, image and video generators, and the synthetic tensor dataset are
  flagged `hidden`: they ship for the bundled workflows and older canvases but
  stay out of the node library.
- `app/workflow_catalog.py` — example workflow models, socket/DAG validation
  and `WorkflowGraph` materialisation.
- `app/workspace.py` — file-backed named projects, 30 retained version
  snapshots, workspace trash, and close/visibility recovery sessions.
- `app/main.py` — FastAPI server: REST endpoints plus `/ws/execution` streaming.
- `app/templates/` — 84 official ML node templates across 10 categories:
  scalar values (int/float/bool/string), data loaders (sklearn sets,
  synthetic/messy generators, CSV/Parquet/Excel/NDJSON), cleaning, EDA,
  preprocessing, imbalanced-data samplers, models (scikit-learn +
  XGBoost/LightGBM), evaluation, the PyTorch tensor path, and computer vision
  (torchvision image classification/detection/segmentation, video
  classification, YOLO image/video detection and segmentation, image/video
  loaders, media preview and media saving). Optional dependencies (`torch`, `torchvision`,
  `ultralytics`, `opencv-python`, `imageio-ffmpeg`, `safetensors`,
  `onnxruntime`, `xgboost`, `lightgbm`, `fastexcel`, `imbalanced-learn`) are
  declared in PEP 723 headers and installed on demand. Generated videos are
  encoded as browser-playable H.264 MP4/MOV/MKV through the bundled
  `imageio-ffmpeg` encoder.
  Scalar outputs can also target any widget parameter (for example
  `float_value.result -> train_test_split.test_size`); the executor overrides
  that keyword argument after applying stored widget values.
- `app/workflows/` — 21 example workflow definitions (JSON) served by the API
  and validated at load time, including offline vision image/video pipelines.

Ultralytics (YOLO) templates are kept in a separate node family from the
torchvision templates so non-YOLO vision workflows do not install it.
The `ultralytics` package is **AGPL-3.0**; review its license before
distributing generated applications or network services that use YOLO nodes.

## Local models

The model library lives at the repo root, not inside the active workspace:

```text
models/
├── yolo/             # .pt / .onnx YOLO checkpoints
├── torchvision/      # .pt / .pth / .safetensors detection/segmentation state dicts
└── classification/   # .pt / .pth / .safetensors / .onnx classifiers
```

Custom folders are discovered automatically. `COMPLEXUI_MODELS_DIR` can point
elsewhere (for tests, shared caches, or deployments). Model binaries are ignored
by git; only the folder placeholders and documentation are tracked.

Code-first selector syntax:

```python
local_model: ModelSelector = ModelSelector(type="yolo")
```

The marker is static DSL metadata and is never imported or executed. It becomes
a STRING-compatible model widget whose value is
`models/<type>/<file>`. At execution the executor resolves it to an absolute
path. Empty means use the predefined fallback. Missing, mismatched, or
unsupported selections fail during `validate_graph` before a run starts. The
selected file's `size:mtime_ns` fingerprint is part of the node cache key, and
exported scripts receive the resolved absolute model path.

Supported loading by template family:

| Family | Formats | Notes |
| --- | --- | --- |
| YOLO image/video | `.pt`, `.onnx` | Loaded directly by Ultralytics |
| torchvision classifier | `.pt`, `.pth`, `.safetensors`, `.onnx` | State dict via `weights_only=True` / SafeTensors, or ONNX Runtime; architecture widget still supplies preprocessing/metadata |
| torchvision detector/segmenter | `.pt`, `.pth`, `.safetensors` | State dict only; torchvision ONNX detection/segmentation is rejected (YOLO is the ONNX path) |

Model files are executable input. `.pt` may contain pickle payloads; loading a
YOLO `.pt` is not a sandbox even though workflow execution has resource
isolation. Only add trusted files, and prefer SafeTensors or ONNX where
possible. The API/UI only scans metadata and never imports models while listing.

Note: `train_test_split` now exposes `target_column` and `stratify` widgets.
Canvases saved against the old template keep their old node code, so after
upgrading those split nodes no longer match the official hash and need a
one-time approval (or replacement). Newly dragged split nodes use the new
template.

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/parse` | POST | Parse node code into a `NodeSchema` (never executes code). |
| `/api/templates` | GET | Official templates with code, hash, category, description, `hidden` flag and socket schema. Hidden templates are still served (workflows depend on them); the UI keeps them out of the palette. |
| `/api/workflows` | GET | Bundled example workflows (nodes, edges, widget values, positions), ordered by category. |
| `/api/health` | GET | Liveness probe (`status`, `version`, template/workflow counts, `models_root`) used by the launcher. |
| `/api/models/types` | GET | List default and custom model type folders with file counts. |
| `/api/models?type=<type>` | GET | List metadata for supported model files (optionally scoped to one type); never loads model contents. |
| `/api/status` | GET | Execution status (`idle`/`running` + `run_id`); clients call it after a WebSocket reconnect to resync. |
| `/api/approve` | POST | Approve custom/modified node code for this session. |
| `/api/export` | POST | Body `{"graph": ..., "format": "python" \| "notebook", "targets": [...] \| null}`. Returns `{filename, media_type, content}` for a standalone Python script or notebook; validates wiring/syntax but does not execute or require trust approval. |
| `/api/projects` | GET | List named projects (slug, display name, save time, node/edge/version counts). |
| `/api/projects` | POST | Body `{"name", "workflow", "overwrite": false}`. Create a project; `409` if the slug exists unless `overwrite` is true. |
| `/api/projects/{slug}` | GET | Load a project document. |
| `/api/projects/{slug}` | PUT | Save over a project; snapshots the previous document first (30 retained). |
| `/api/projects/{slug}` | DELETE | Move a project and its snapshots to `workspace/.trash/`. |
| `/api/projects/{slug}/versions` | GET | List retained version snapshots, newest first. |
| `/api/projects/{slug}/versions/{version}` | GET | Load one snapshot. |
| `/api/session` | GET | Return the latest recovery session or `null`. |
| `/api/session` | PUT | Persist the canvas on page hide/close (`workflow`, `project_slug`, `project_name`). |
| `/api/uploads` | GET | List uploaded assets (`id`, `name`, `inputs/...` path, format, `kind`, size, modified time), newest first. |
| `/api/uploads` | POST | Multipart `file` upload. Accepts CSV, TSV, Parquet, XLSX/XLSM, NDJSON/JSONL (100 MiB), and PNG/JPEG/WebP/GIF/BMP/TIFF or MP4/WebM/MOV/AVI/MKV/M4V media (500 MiB); sanitizes names and suffixes collisions. |
| `/api/uploads/{file_id}` | GET | Return metadata for one uploaded asset. |
| `/api/uploads/{file_id}/raw` | GET | Serve an uploaded image/video inline with range support and `nosniff`. |
| `/api/outputs/{file_id}/raw` | GET | Serve a generated image/video from repo-root `outputs/` inline with range support. |
| `/api/uploads/{file_id}/preview` | GET | Return up to `limit` table rows (default 20, max 100) plus `columns`, `rows`, and `truncated`; `422` for media. |
| `/api/uploads/{file_id}` | DELETE | Delete one uploaded asset from `inputs/`. |
| `/api/execute` | POST | Body `{"graph": ..., "targets": [...] \| null, "node_labels": {...} \| null}`. Pre-flights (422 invalid / 403 untrusted / 409 busy), optionally scopes execution to the ancestors of `targets`, then runs the graph; returns `202` with a `run_id`. Labels are retained for run-history display. Flattened subgraph nodes may carry `approval_id` so session approvals remain scoped to the original node id. |
| `/api/cancel` | POST | Ask the running execution to stop between nodes or during a dependency install (`202`; `409` when idle). Completed nodes stay cached. |
| `/api/runs` | GET | List the newest retained terminal run summaries (`limit` 1-200, default 50). |
| `/api/runs/{run_id}` | GET | Load one terminal run with order, labels, per-node statuses/timings/outputs and failure details. |
| `/api/runs/{run_id}` | DELETE | Delete one retained run record. |
| `/ws/execution` | WS | Streams `EXECUTION_STARTED`, per-node `NODE_*` (including `rich_outputs`), and `EXECUTION_COMPLETED`/`EXECUTION_FAILED`/`EXECUTION_CANCELLED` events as JSON. |

Requests enforce node-code, graph-size, and target-count caps; advertised
request bodies larger than 16 MiB are rejected with `413`, except
`/api/uploads`, which accepts up to 502 MiB so a 500 MiB media asset plus
multipart overhead fits. Upload contents are also capped per asset kind while
streaming. The server binds
loopback by default and accepts only loopback `Host` headers and WebSocket
`Origin`s; set `COMPLEXUI_ALLOWED_HOSTS` (comma-separated) for custom hostnames
or use the launcher's `--allow-network` flag, which disables the guard and warns.
The API has no authentication and executes node code, so non-loopback exposure
is only appropriate on trusted networks.

Cancellation takes effect between nodes and while a dependency install runs.
Node code that is already executing is not interrupted by the cancel endpoint;
the per-node timeout terminates the isolated execution process instead.

## Execution isolation

Each run executes in a disposable child process. The parent keeps the HTTP and
WebSocket loop responsive, transfers the content-addressable cache and session
approvals into the child, and updates the parent cache only from a completed,
failed or cancelled terminal snapshot. `NODE_*` telemetry is forwarded live.
If the child crashes, the parent emits a `WorkerCrashed` terminal event and the
next run starts a fresh process. Cache entries whose values cannot be pickled
are intentionally not transferred (the next run recomputes them).

Defaults and environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMPLEXUI_NODE_TIMEOUT_SECONDS` | `600` | Maximum wall time from `NODE_STARTED` to completion; `0` disables |
| `COMPLEXUI_RUN_TIMEOUT_SECONDS` | `3600` | Maximum wall time for one run including installs; `0` disables |
| `COMPLEXUI_MAX_MEMORY_MB` | `4096` | Best-effort worker RSS limit; `0` disables |
| `COMPLEXUI_INPUT_DIR` | repo-root `inputs/` | Uploaded-input directory |
| `COMPLEXUI_OUTPUT_DIR` | repo-root `outputs/` | Generated-output directory |

On Unix the worker also applies an address-space limit; the parent monitors
resident memory on all platforms. Timeout, memory-limit and crash failures are
persisted in run history with `NodeTimeout`, `RunTimeout`, `MemoryLimit` or
`WorkerCrashed` error types. This isolation contains crashes and runaway
processes — it is **not** a filesystem/network security sandbox. Node code still
runs with the backend user's permissions, so only execute trusted code and keep
the API loopback-only unless the network is trusted. `File` widgets specifically
reject paths that resolve outside the workspace (including absolute paths and
symlink escapes).

## Workspace

The workspace root defaults to `backend/workspace`; set `COMPLEXUI_WORKSPACE`
to relocate it. Project documents use the same `.cuigraph.json` format as
frontend import/export:

```text
workspace/
├── workflows/<slug>.cuigraph.json   # current project documents
├── .versions/<slug>/<timestamp>.cuigraph.json
├── .trash/<slug>-<timestamp>/      # deleted project + its snapshots
├── runs/<run_id>.json              # 50 newest terminal run records
├── session.json                    # last close/visibility recovery
└── logs/                           # daily + per-run plain-text logs
```

Uploaded tables, images and videos live in the repo-root `inputs/` directory;
generated images, videos and artifacts live in repo-root `outputs/`. Set
`COMPLEXUI_INPUT_DIR` or `COMPLEXUI_OUTPUT_DIR` to relocate them.

Writes are atomic, slugs are restricted to `[a-z0-9-]`, and project/version
reads revalidate the document before returning it. Overwrites keep the 30 newest
snapshots. Project documents may include canvas frame nodes (`isFrame`, `width`,
`height`) and child links (`parentId`); frames are UI-only, cannot carry edges,
must be valid parents, and are excluded from execution and exported artifacts.
They may also include reusable `subgraphs`: each definition has a unique id, a
shared input/output port contract, its own nodes/edges (including edges to the
reserved `__subgraph_input__` / `__subgraph_output__` managers), and manager
positions. Instance nodes store `isSubgraph` plus `subgraphId`. Validation
rejects missing/cyclic definitions, invalid manager edges, conflicting frame/
instance nodes, and unknown instance ports. The frontend flattens nested
instances before execution/export and scopes ids as `sub:<path>/<id>`.

Terminal runs are saved before their terminal WebSocket event is emitted, so the
history API is immediately consistent with the UI. The 50 newest records are
retained; each stores the scoped node order, labels, targets, per-node status and
timings, previews, errors, and at most 5 MiB of rich outputs per run (excess
payloads are omitted with a truncation flag). History write failures are logged
and never change the execution result. Run-record retention is independent of
the 14-day plain-text log retention. File logs follow `COMPLEXUI_WORKSPACE`
unless `COMPLEXUI_LOG_DIR` overrides them. File widgets store paths like
`inputs/iris.csv`; execution resolves them against the repo-root inputs
directory. Legacy workspace-relative paths remain readable. They may also read
generated files through `outputs/<name>`, while `OutputFile` names write under
repo-root `outputs/`. File access rejects absolute paths and symlink escapes
outside the workspace, inputs, and outputs roots. Exports retain those paths,
so keep the input and generated-output directories available (or adjust the
paths).

## Rich outputs

Each node runs with a capture context that collects:

- `print()`/stdout as text (flushed before the next `display()` call).
- `display(value)` for Polars/Pandas-style tables, matplotlib/seaborn figures,
  PIL images, Plotly figures, HTML, and `_repr_mimebundle_` / `_repr_png_` /
  `_repr_jpeg_` / `_repr_svg_` / `_repr_html_` objects.
- matplotlib figures created through `pyplot`, including `plt.show()` and
  figures left open when the node finishes.
- Plotly `figure.show()` (including modules imported after node start); calling
  both `display(fig)` and `fig.show()` emits one figure.

Capture is headless (`MPLBACKEND=Agg`, Plotly JSON renderer) and best effort:
a broken repr or oversized figure is dropped/truncated rather than failing the
node. Remote `stderr`/warnings stay in server logs. Outputs are included in
`NODE_COMPLETED`/`NODE_ERROR` telemetry as `rich_outputs`, cached with the node,
and never added to `EXECUTION_COMPLETED` (the per-node frames already carry
them). Matplotlib, seaborn and Plotly are not backend runtime dependencies;
nodes declare them in PEP 723 headers and the capturer detects loaded modules.

Per-node limits: 10 outputs, 12,000 text characters, 60,000 HTML characters,
a 20×30 table preview, 1 MB per image, 1.5 MB per Plotly figure, and 2.5 MB of
rich output total. A truncation notice is emitted when a limit drops content.
Rich outputs are runtime-only and are not exported in `.cuigraph.json`; terminal
run-history records retain a bounded snapshot for later inspection.

Start the server (CORS pre-configured for `http://localhost:5173`):

```sh
uv run uvicorn app.main:app --reload
```

Production-style single-server run (serves the built frontend from `../frontend/dist`):

```sh
uv run uvicorn app.main:app --port 8000
# PowerShell: $env:COMPLEXUI_SERVE_DIST = "..\frontend\dist"
```

Operator launcher (preferred): `uv run python launch.py [dev|backend|frontend|prod|check]`
or the repo-root `start.cmd`. Console logging honors `COMPLEXUI_LOG_LEVEL`
(default `INFO`), `COMPLEXUI_LOG_FORMAT` (`text` or newline-delimited `json`;
uvicorn access lines remain plain text), `NO_COLOR` / `FORCE_COLOR`, and
`--no-color` / `--log-level` / `--log-format`.

Plain-text logs are also written daily to `backend/workspace/logs/` (14 days
retained), with one file per execution under `backend/workspace/logs/runs/`.
Set `COMPLEXUI_LOG_DIR` to relocate them, or `0` to disable files. The launcher
accepts `--log-dir PATH` and `--no-log-file`.

Run the test suite from this directory:

```sh
uv run ruff check .    # lint (app/templates excluded: they use the node DSL)
uv run pytest
```

The default suite skips nodes whose optional runtimes are absent. To exercise
the full catalog (torch, torchvision, Ultralytics, OpenCV, XGBoost, LightGBM),
run the overlay gate:

```sh
uv run --with torch --with torchvision --with ultralytics \
  --with safetensors --with onnxruntime --with xgboost --with lightgbm pytest
```

Vision workflows use offline random weights/YAML models by default, so their
tests do not download checkpoints.
