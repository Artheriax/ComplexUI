# ComplexUI Backend

FastAPI service and node-code toolchain for ComplexUI.

- `app/schemas.py` — Pydantic v2 models for node schemas, workflow graphs, and execution telemetry events.
- `app/transpiler.py` — node-code DSL transpiler (PEP 723 extraction, typed
  returns, input/widget/output extraction, function wrapping).
- `app/executor.py` — topological DAG executor with a bounded
  content-addressable LRU cache (64 outputs / 32 MiB), trust gating,
  cancellation checks and WebSocket-ready event streaming.
- `app/output_capture.py` — best-effort capture of `print`, `display`, tables,
  images, HTML and Plotly figure side effects for each node.
- `app/env_manager.py` — PEP 723 dependency checks and a cancellable
  `uv pip install` runner (10-minute timeout; failures emit `NODE_ERROR`).
- `app/security.py` — SHA-256 template registry and session approval tracking.
- `app/network_security.py` — loopback-first `Host`/WebSocket `Origin`
  validation and a request-size guard.
- `app/template_catalog.py` — display metadata (title, category, description,
  palette visibility) for the official node catalog. Example dataset loaders
  (iris/wine/breast-cancer/diabetes/digits), the synthetic classification and
  messy generators, and the synthetic tensor dataset are flagged `hidden`:
  they ship for the bundled workflows and older canvases but stay out of the
  node library.
- `app/workflow_catalog.py` — example workflow models, socket/DAG validation
  and `WorkflowGraph` materialisation.
- `app/main.py` — FastAPI server: REST endpoints plus `/ws/execution` streaming.
- `app/templates/` — 60 official ML node templates across 8 categories:
  data loaders (sklearn sets, synthetic/messy generators, CSV/Parquet/Excel/
  NDJSON), cleaning, EDA, preprocessing, imbalanced-data samplers, models
  (scikit-learn + XGBoost/LightGBM), evaluation, and the PyTorch tensor path.
  Optional dependencies (`torch`, `xgboost`, `lightgbm`, `fastexcel`,
  `imbalanced-learn`) are declared in PEP 723 headers and installed on demand.
- `app/workflows/` — 12 example workflow definitions (JSON) served by the API
  and validated at load time.

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
| `/api/health` | GET | Liveness probe (`status`, `version`, template and workflow counts) used by the launcher. |
| `/api/status` | GET | Execution status (`idle`/`running` + `run_id`); clients call it after a WebSocket reconnect to resync. |
| `/api/approve` | POST | Approve custom/modified node code for this session. |
| `/api/execute` | POST | Body `{"graph": ..., "targets": [...] \| null}`. Pre-flights (422 invalid / 403 untrusted / 409 busy), optionally scopes execution to the ancestors of `targets`, then runs the graph; returns `202` with a `run_id`. |
| `/api/cancel` | POST | Ask the running execution to stop between nodes or during a dependency install (`202`; `409` when idle). Completed nodes stay cached. |
| `/ws/execution` | WS | Streams `EXECUTION_STARTED`, per-node `NODE_*` (including `rich_outputs`), and `EXECUTION_COMPLETED`/`EXECUTION_FAILED`/`EXECUTION_CANCELLED` events as JSON. |

Requests enforce node-code, graph-size, and target-count caps; advertised
request bodies larger than 16 MiB are rejected with `413`. The server binds
loopback by default and accepts only loopback `Host` headers and WebSocket
`Origin`s; set `COMPLEXUI_ALLOWED_HOSTS` (comma-separated) for custom hostnames
or use the launcher's `--allow-network` flag, which disables the guard and warns.
The API has no authentication and executes node code, so non-loopback exposure
is only appropriate on trusted networks.

Cancellation takes effect between nodes and while a dependency install runs;
node code that is already executing is not forcibly interrupted.

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
Rich outputs are
runtime-only and are not exported in `.cuigraph.json`.

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

Run the test suite from this directory:

```sh
uv run ruff check .    # lint (app/templates excluded: they use the node DSL)
uv run pytest
```

The default suite skips nodes whose optional runtimes are absent. To exercise
the full catalog (torch, XGBoost, LightGBM), run the overlay gate:

```sh
uv run --with torch --with xgboost --with lightgbm pytest
```
