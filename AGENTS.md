# AGENTS.md — ComplexUI Development Rules

ComplexUI is a local-first, code-first node-graph IDE for Machine Learning. Each
node is editable Python parsed into typed sockets/widgets; workflows execute as a
validated DAG in an isolated child process and stream telemetry over WebSocket.

Read `README.md`, `backend/README.md`, and `frontend/README.md` before changing
behavior. Those files own product/API/operator details. This file owns durable
engineering rules; keep it short and do not duplicate feature documentation.

## Architecture map

- Request flow: node code -> `POST /api/parse` -> `NodeSchema`; frontend edits
  graph state; run -> subgraphs flattened client-side -> `POST /api/execute` ->
  `validate_graph` -> disposable worker -> `executor.execute_graph` -> WebSocket
  events -> run history.
- Backend: `schemas.py` is the Pydantic v2 API contract; `transpiler.py` is the
  AST/tokenize node DSL parser/compiler; `executor.py` owns DAG validation,
  topological order, cache, trust checks and node execution; `execution_host.py`
  and `execution_worker.py` own the process boundary and resource guards;
  `main.py` is HTTP/WebSocket orchestration only. Storage is split across
  `workspace.py`, `datasets.py`, `models.py` and `run_history.py`; trust and
  network policy live in `security.py` and `network_security.py`.
- Frontend: `types/graph.ts` mirrors backend contracts; `lib/api.ts` is the REST
  client; `lib/persistence.ts` owns the save/autosave format; `lib/subgraph.ts`
  owns flattening and boundary ports. Zustand stores own mutable state, hooks own
  side effects, components own rendering.
- Runtime data: workspace projects/versions/runs/logs, repo-root `inputs/`,
  `outputs/` and `models/`. All overrides use `COMPLEXUI_*` environment variables.
- Optimization/quantization work: read `TODO.md` §9.2 for the remaining
  checklist, implementation map, source links and validated package versions
  before changing attention, precision, loader or memory code.

## 1. Architecture and modularity

- Preserve dependency direction: `main.py` -> domain modules; never import
  FastAPI, HTTP or WebSocket concerns into `transpiler.py` or `executor.py`.
  The executor must remain usable inside the spawned worker process.
- Keep parsing, validation, execution, persistence and presentation separate.
  Add behavior to the module that already owns that concern instead of creating
  a new layer or wrapper.
- Frontend: pure transforms belong in `lib/`, state transitions in stores, side
  effects in hooks, and JSX in components. Do not add unrelated domains to
  `useGraphStore.ts`; it is already the largest module in the app.
- Avoid speculative abstractions, generic frameworks and indirection. Use the
  simplest design that fits the current requirement and existing patterns.

## 2. Code quality and maintainability

- Python >= 3.12, type hints on public functions, Pydantic v2 at API/storage
  boundaries, dataclasses for internal records, Ruff line length 100.
- TypeScript is strict with `noUnusedLocals`/`noUnusedParameters`. Avoid `any`,
  `as any`, `@ts-ignore` and unchecked casts; model variants as discriminated
  unions and validate unknown input.
- No `TODO`, `FIXME`, bare `pass`, dead code or commented-out implementations in
  core logic. No placeholders in shipped behavior.
- Keep functions and components focused. Reuse existing helpers (`api.ts`,
  `persistence.ts`, `subgraph.ts`, `socketCompatibility.ts`, `WidgetField`) before
  adding new ones. Match surrounding naming, formatting and docstring style.
- Keep diffs scoped. Do not mix unrelated refactors, formatting churn or
  dependency upgrades into a feature/bug fix.

## 3. Security and privacy

- Node code and model files are executable input. Never bypass the template hash
  registry, session approval, or diff-review flow. Keep frontend/backend code
  hashing line-ending normalization identical, and keep approvals scoped to
  `(node_id, code_hash)`; flattened nodes use `approval_id`.
- Keep the default loopback-only `Host`/WebSocket `Origin` protection, explicit
  CORS origins, request-size limits, upload streaming caps and path confinement.
  Do not add wildcard hosts/origins or network exposure without an explicit
  requirement and a documented risk.
- The API has no authentication by design. Never place secrets in the repo,
  logs, fixtures or error payloads. Do not log full node payloads or credentials.
- Resolve user paths only through `DatasetStore`, `OutputStore`, `ModelStore` or
  workspace helpers; reject absolute/parent/symlink escapes. Validate names,
  IDs, sizes and extensions at the boundary.
- Sanitize rich HTML with DOMPurify before rendering. Do not introduce unsafe
  `innerHTML`, remote scripts/CDNs or new `eval`-style execution paths.

## 4. Performance and resource efficiency

- Never block the event loop with CPU-heavy or synchronous I/O work in `async`
  endpoints; use `asyncio.to_thread`/the existing threadpool pattern. Heavy node
  work belongs in the execution worker.
- Preserve one-run-at-a-time coordination, cancellation semantics, wall-clock
  and memory guards, and bounded cache/output/log retention unless the task
  explicitly changes the protocol.
- Keep node cache keys stable and complete: code, widget values, upstream state
  and model fingerprints all participate. Do not make uncacheable outputs
  cacheable without checking determinism.
- Bound every growing collection: caches, WebSocket queues, run records, logs,
  rich outputs, media previews and browser state.
- Frontend: use narrow Zustand selectors, lazy-load heavy Monaco/Plotly paths,
  clean up sockets/timers/pollers, and pause polling when the page is hidden.

## 5. DRY, SOLID and appropriate OOP

- Backend `schemas.py` is the contract source of truth. Update the TypeScript
  mirrors and their tests in the same change as any API/contract change.
- Keep one typed source for socket types and compatibility rules. Do not
  hand-maintain parallel enum/list sets; derive validation sets from existing
  typed constants and test every member.
- Prefer pure functions for graph transforms and validation. Use classes only
  for stateful stores/managers (the existing pattern); do not introduce deep
  inheritance or a god object.
- A rule implemented in both backend and frontend (socket compatibility,
  workflow limits, hashing) must have matching semantics and regression tests.

## 6. API and dependency management

- Backend packages: `uv` only (`uv add`, `uv sync`, `uv run ...`); never raw
  `pip` or manually managed virtualenvs. Frontend packages: `npm`, with
  `package-lock.json` committed.
- Add a dependency only when the requirement cannot be met with the standard
  library or existing packages. Optional ML runtimes belong in PEP 723 headers,
  the backend dev group, or the ML overlay workflow — not the core runtime.
  Keep optional backends isolated (for example Ultralytics/YOLO vs torchvision).
- API payloads and responses are Pydantic models; enforce the existing node,
  edge, code-size, target and preview caps. Use established statuses (`422`
  invalid, `403` untrusted, `409` busy/conflict, `413` too large).
- WebSocket event names and payload shape are a client contract. Extend
  additively; do not remove or rename events/fields without updating the
  frontend reducer and run history.
- Workflow save documents are versioned. Prefer additive optional fields; a
  breaking format change requires a version bump and import compatibility.
- Preserve `COMPLEXUI_*` environment names and document any new setting in the
  relevant README.

## 7. UI/UX consistency and accessibility

- Reuse the existing dark Tailwind styling, `icons.tsx`, `WidgetField`, dialog
  shell and `useDialogA11y`. Do not add a second UI kit, theme system or CSS
  framework.
- Dialogs must use `useDialogA11y`, `role="dialog"` and `aria-modal="true"`;
  preserve focus trapping/restoration and Escape-to-close. Icon-only controls
  need an accessible name; announce async status/errors with the existing live
  regions.
- Keep keyboard behavior consistent: Ctrl/Cmd shortcuts, no canvas shortcuts
  while typing in inputs or while modals are open, and no dead-end interactions.
- Treat the launcher console as a product surface: keep the colorized startup
  diagnostics, single updating progress line and filtered polling access logs.
  Do not add noisy per-request output or a second console logging path.
- Provide loading, empty and error states for every async surface. Do not hide
  failures behind spinners or silent console output. Test UI changes at the
  component level and, for canvas workflows, with Playwright.

## 8. Error handling, logging and validation

- Use explicit domain exceptions (`GraphValidationError`, `DatasetError`,
  `WorkspaceError`, `RunHistoryError`, etc.) and translate them to HTTP status
  codes in `main.py`. Do not raise generic `Exception` for expected failures.
- Broad `except Exception` is only acceptable at best-effort boundaries
  (optional sensors, rich-output capture, logging, cache transfer, run-history
  persistence) and must log a warning/debug with context. Never swallow an
  execution or validation error silently.
- Validate all external input with Pydantic and explicit range/length checks.
  Return actionable messages without leaking tracebacks or filesystem details
  beyond the existing API contract.
- Use module loggers (`logging.getLogger("complexui....")`), not `print`, for
  server behavior. Keep per-run logs and structured fields. Frontend errors go
  through `ApiError` and visible UI state; avoid stray `console.*` in production
  paths.

## 9. Testing, linting, type checking and build verification

Run targeted tests while developing and the full gate before declaring work done:

| Area | Command (working directory) |
| --- | --- |
| Backend lint | `uv run ruff check .` (`backend/`) |
| Backend typecheck | `uv run basedpyright` (`backend/`) |
| Backend tests | `uv run pytest` (`backend/`) |
| Backend coverage | `uv run pytest --cov=app --cov-report=term-missing` (`backend/`) |
| Frontend lint | `npm run lint` (`frontend/`) |
| Frontend typecheck | `npm run typecheck` (`frontend/`) |
| Frontend tests | `npm test` (`frontend/`) |
| Frontend build | `npm run build` (`frontend/`, includes typecheck) |
| Full gate | `start.cmd check` (repo root) |
| Browser E2E | `npx playwright install chromium` once, then `npm run e2e` (`frontend/`) |

- Add a regression test for every bug and a contract test for every API/schema,
  socket-type, persistence or execution-semantics change. Frontend component
  tests are `.test.ts`/`.test.tsx` colocated under `src/`; backend tests live in
  `backend/tests/`.
- Use the `tmp_path`/isolated workspace fixtures; never write into repo
  `inputs/`, `outputs/`, `models/` or `workspace/` during tests. Do not rely on
  network model downloads. Skip optional runtimes explicitly, not silently.
- Run `tests/test_catalog_nodes.py`, `tests/test_workflows.py` and the relevant
  ML node tests when touching templates or bundled workflows. Use the ML
  overlay workflow when a change may affect torch/torchvision/Ultralytics/
  XGBoost/LightGBM.
- Run Playwright when changing canvas interaction, dialogs, persistence or
  run/history UI. Do not claim UI verification from typecheck alone.
- Never weaken/delete a failing test to make the gate pass; fix the behavior or
  add a narrowly justified skip with a reason.

## 10. Documentation and developer experience

- Update the README that owns the changed behavior (`README.md`,
  `backend/README.md` or `frontend/README.md`). Keep API tables, environment
  variables, template/workflow counts and commands accurate; do not restate
  them in `AGENTS.md`.
- Document non-obvious invariants in module docstrings, not inline narration.
  Keep public function docstrings useful and current.
- Do not commit generated artifacts, caches, `node_modules`, `dist`,
  `playwright-report`, workspace data, model binaries, uploads or outputs. Keep
  secrets and machine-specific paths out of the repo.
- Only commit, amend, push or open PRs when explicitly asked. Inspect
  `git status`/`git diff` first and stage only intended files.
- Prefer the launcher (`start.cmd` / `uv run python launch.py`) for a consistent
  developer experience. If a command or environment variable changes, update
  the launcher help and the relevant README together.

## Definition of done

- Behavior is implemented without bypassing security, path, size or retention
  rules.
- Backend Ruff, basedpyright and pytest with coverage pass; frontend lint,
  typecheck, tests and build pass.
- Relevant Playwright/ML-overlay checks pass or are explicitly reported as not
  run.
- Contracts, save format, docs and regression tests are updated together.
- The diff is focused and contains no generated files, secrets or placeholders.
