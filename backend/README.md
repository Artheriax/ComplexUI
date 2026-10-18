# ComplexUI Backend

FastAPI service and node-code toolchain for ComplexUI.

- `app/schemas.py` — Pydantic v2 models for node schemas, workflow graphs, and execution telemetry events.
- `app/transpiler.py` — node-code DSL transpiler (PEP 723 extraction, typed
  returns, input/widget/output extraction, function wrapping).
- `app/executor.py` — topological DAG executor with content-addressable
  caching, trust gating and WebSocket-ready event streaming.
- `app/env_manager.py` — PEP 723 dependency checks and `uv pip install` runner.
- `app/security.py` — SHA-256 template registry and session approval tracking.
- `app/main.py` — FastAPI server: REST endpoints plus `/ws/execution` streaming.
- `app/templates/` — official ML node templates: classical path (Iris load,
  train/test split, KNN classifier, evaluation metrics) and tensor path
  (`tensor_dataset`, `mlp_train`, `evaluate_torch` — PEP 723 `torch>=2.2`,
  installed on demand).

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/parse` | POST | Parse node code into a `NodeSchema` (never executes code). |
| `/api/templates` | GET | Official templates with code, hash and socket schema. |
| `/api/health` | GET | Liveness probe (`status`, `version`, template count) used by the launcher. |
| `/api/approve` | POST | Approve custom/modified node code for this session. |
| `/api/execute` | POST | Body `{"graph": ..., "targets": [...] \| null}`. Pre-flights (422 invalid / 403 untrusted / 409 busy), optionally scopes execution to the ancestors of `targets`, then runs the graph; returns `202` with a `run_id`. |
| `/api/cancel` | POST | Ask the running execution to stop after the current node (`202`; `409` when idle). Completed nodes stay cached. |
| `/ws/execution` | WS | Streams `EXECUTION_STARTED`, per-node `NODE_*`, and `EXECUTION_COMPLETED`/`EXECUTION_FAILED`/`EXECUTION_CANCELLED` events as JSON. |

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
(default `INFO`), `NO_COLOR` / `FORCE_COLOR`, and `--no-color` / `--log-level`.

Run the test suite from this directory:

```sh
uv run pytest
```
