# ComplexUI Frontend

React 19 + TypeScript (strict) + XYFlow + Monaco + Zustand + TailwindCSS v4.

- `src/types/graph.ts` — TypeScript mirrors of the backend Pydantic contracts.
- `src/lib/api.ts` — REST client (`/api/*` via the Vite dev proxy), including
  export, dataset upload/list/preview/delete (multipart uploads use a dedicated
  FormData request), named-project/version, run-history, and recovery-session
  requests.
- `src/lib/hash.ts` — SHA-256 code hashing with the same line-ending
  normalization as `backend/app/security.py`.
- `src/lib/persistence.ts` — `.cuigraph.json` save format, size/shape
  validation, localStorage workflow/workspace/project-identity storage, and
  file download/upload helpers.
- `src/lib/subgraph.ts` — reusable subgraph boundary derivation, shared
  interface managers, cycle checks, and recursive path-scoped execution
  flattening (`sub:<path>/<id>`).
- `src/lib/monaco.ts` — local Monaco bundling (no CDN), worker configured.
- `src/store/useGraphStore.ts` — canvas state, independent workflow windows,
  nested subgraph navigation and definitions, trust workflow, project
  saving/opening/history, multi-selection arrangement and frame grouping,
  execution telemetry reducer, run-status resync.
- `src/store/useUiStore.ts` — sidebar drawer, console drawer and shortcuts
  dialog visibility.
- `src/store/useDatasetsStore.ts` — workspace dataset list, uploads, deletion,
  loading/error state.
- `src/hooks/useExecutionSocket.ts` — `/ws/execution` streaming with
  exponential-backoff reconnect and `/api/status` resync on every open.
- `src/hooks/useGraphShortcuts.ts` — undo/redo, clipboard, duplicate,
  group/ungroup, select-all, and `Ctrl+S` save shortcuts for the canvas.
- `src/hooks/useViewportShortcuts.ts` — keyboard zoom for the canvas.
- `src/hooks/useWorkflowFileActions.ts` — File-menu save/export/import/clear
  actions.
- `src/hooks/useWorkflowAutosave.ts` — debounced localStorage workspace and
  active-workflow autosave plus `pagehide`/`visibilitychange` server-session
  persistence.
- `src/components/` — `WindowTabs` (workflow windows), `Canvas` (with
  nested-canvas breadcrumbs and double-click node search), `CanvasStatusBar`
  (backend status, node count, run controls), `ComplexNode` (widgets, sockets,
  frames, subgraph instances and shared interface managers), `CanvasContextMenu`
  (pane actions with a File submenu), `CodeModal` (Monaco editor with debounced
  live parsing), `DiffModal` (security review of modified code), `NodePalette`
  (square icon rail plus Nodes / Workflows / Templates / Datasets drawers),
  `NodeLibrary` (searchable templates plus reusable
  subgraphs), `WorkflowLibrary` (collapsible example-workflow categories),
  `ProjectLibrary`, `DatasetLibrary` (upload, search, preview, delete),
  `DatasetPickerDialog`, `DatasetPreviewDialog`, `WidgetField` (typed
  parameters, fixed choices, step buttons and linked state),
  `FileWidgetField` (workspace file picker for `File` widgets),
  `SelectionToolbar` (multi-select alignment, distribution, grid layout,
  frame grouping/ungrouping and reusable subgraph creation),
  `SaveProjectDialog`, `ProjectHistoryDialog`, `ShortcutsDialog`,
  `ExecutionPanel`, `RunHistoryDialog` (retained terminal runs with per-node
  timings and outputs), `RichOutputView` (text, tables, images, sanitized HTML
  and static/interactive Plotly).

## Features

- A compact square-button rail expands **Nodes**, **Workflows**, **Templates**, and **Datasets** drawers on demand, keeping the canvas
  wide. The same rail toggles the Execution (Console) drawer, opens the
  keyboard-shortcut reference and Settings.
- Categorized node library with search and collapsible sections that start
  collapsed (production templates across values, data loaders, cleaning, EDA,
  preprocessing, imbalanced data, models, evaluation and PyTorch). Example
  dataset loaders and synthetic test nodes are hidden from the palette — they
  power the bundled workflows.
- **Workflow windows**: the top bar is a tab strip of open workflows. The `+`
  button creates a blank window; opening a saved workflow or bundled template
  adds a new window, and opening a project already in a tab focuses it. Each
  window keeps its own graph, viewport, undo history, execution state and
  project identity. Open windows are restored from localStorage, and only one
  backend run is allowed at a time (the running tab shows a spinner).
- **Templates drawer**: 12 ready-to-run examples with collapsible categories;
  opening one adds a fully wired graph (nodes, widgets and positions) in a new
  window.
- **Datasets drawer**: upload CSV, TSV, Parquet, Excel or NDJSON files (100 MiB
  each) into `workspace/data/uploads`, search the library, preview up to 20
  rows, and delete files. Loader nodes expose a **File** widget that picks a
  dataset or accepts a workspace-relative path; uploads can also be made from
  the picker. Execution rejects absolute paths and symlink escapes outside the
  workspace.
- **Undo/redo** with a 100-step history (node/edge/widget edits, drags,
  arrangement and code commits), plus a **multi-node clipboard**: select several
  nodes and copy/cut/paste/duplicate them with the edges between them preserved.
- **Selection authoring**: Shift or Ctrl click to multi-select, Shift/Ctrl drag
  for box selection, and `Ctrl+A` to select all. A floating toolbar aligns
  left/center/right and top/middle/bottom, distributes three or more nodes with
  equal gaps, lays selected nodes out in a grid, and clears the selection. Each
  arrangement is one undo step.
- **Canvas navigation**: double-click empty canvas to open the searchable node
  picker, or right-click for the action menu (add node, paste, select all, fit
  view, undo/redo, and a **File** submenu with Save, Export, Import and Clear).
  Double-click a node to edit its code, enter a subgraph or rename a frame.
- **Groups/frames**: group two or more top-level nodes (`Ctrl+G`) into a
  resizable, colored frame. Dragging the frame moves its children together;
  ungroup (`Ctrl+Shift+G`) restores their absolute positions. Frames persist in
  workflows but are excluded from execution and Python/notebook export. Deleting
  a frame keeps its contents, while copy/paste and duplicate preserve the
  complete frame hierarchy.
- **Reusable subgraphs**: select nodes and choose **Create subgraph** to replace
  them with a subgraph instance, automatically deriving input/output ports from
  the boundary edges. Each definition has one shared **Inputs** and **Outputs**
  manager pair; double-click an instance (or use the breadcrumb) to edit the
  nested canvas. Inside a definition, use **Add input** / **Add output** on the
  manager nodes for an ANY-typed port, or drag a wire to/from the dashed +
  handles (or drop it on the manager body) to create a port with the inferred
  type and name. Dragging a + handle onto empty canvas opens the same filtered
  node picker used elsewhere and creates the node already wired. Definitions
  appear in the Nodes library for drag/click reuse; deleting the last instance
  removes the definition too (undo restores both). Execution and
  Python/notebook export recursively flatten instances with
  path-scoped ids (`sub:<path>/<id>`), and approvals remain node-scoped through
  the `approval_id` carried by flattened nodes. Run from **Root**; nested
  canvases disable the run button.
- Drag official templates from the node library, or create **scratch nodes**
  and edit their Python code in place (schemas re-parse while you type).
- **Value nodes**: Int, Float, Bool and String templates expose a single typed
  output. Float/Int also expose an editable `step` parameter and render
  `<`/`>` buttons around `value`; the buttons jump by that step.
  Their outputs can be wired either into typed input sockets or directly into
  compatible widget parameters (for example `test_size`). A linked widget is
  read-only and marked **linked** on the node; the incoming value overrides the
  stored widget value at run time.
- **Fixed choices**: a widget declared as
  `level: Literal["low", "medium", "high"] = "low"` renders as a dropdown.
  Stored defaults, workflow overrides, and values arriving over a wire are
  validated against those choices before the node runs.
- Typed sockets: connections are only allowed between equal types (or `ANY`),
  and each input/widget parameter accepts one edge. Releasing a connection on
  empty canvas opens a searchable node picker filtered to compatible sockets or
  widget parameters and creates the new node already wired (toggleable in
  Settings).
- **Trust workflow**: modified/scratch code is diffed against its template in
  a Monaco diff editor and must be approved before it can run.
- **Export**: the right-click **File** submenu downloads workflow JSON, a
  runnable PEP 723 Python script, or a Jupyter notebook with one cell per node.
  Python/notebook exports are generated by the backend from the same transpiled
  node functions and topological order used for execution.
- **Named projects**: press `Ctrl+S` to name the active window. Later saves are
  silent; name conflicts show an overwrite confirmation. The Workflows drawer lists projects, supports search/delete, and opens a retained version
  snapshot for review. Server-side overwrites keep the 30 newest versions.
- **Persistence and recovery**: all open workflow windows autosave to
  localStorage and are restored on reload; the active workflow is also written
  to the legacy single-workflow slot. Workflows can be exported/imported as
  `.cuigraph.json` files, with imports opening in a new tab. Closing or hiding
  the page also writes a server recovery session; startup prefers the local
  workspace and falls back to that session. Imports enforce the same
  node/edge/code caps as the backend, reject duplicate ids, discard corrupt
  autosaves, validate subgraph references/cycles, and re-parse schemas on load
  so old files always match the current transpiler. Restored widget values are
  dropped when their type no longer matches the parsed widget. Subgraph
  definitions persist in the same `.cuigraph.json` document and implicit
  interface manager nodes are reconstructed from each definition's port
  contract.
- Live execution telemetry: per-node status, cached badges and typed previews
  (DataFrame head, dataset shapes, model params, tensor shape/dtype/sample,
  metrics). After a socket reconnect the client asks `/api/status` and marks a
  run interrupted if the backend is no longer executing it.
- **Run history**: terminal runs (completed, failed or cancelled) persist in
  the workspace and open from the Execution (Console) drawer. Each record keeps per-node
  statuses, timings, cache badges, previews, bounded rich outputs and error
  tracebacks, so results remain inspectable after reloads or later runs.
- **Rich node outputs**: `print()`/stdout, tables, images, sanitized HTML and
  Plotly figures render inline on each node (compact, static plots) and in the
  Execution panel (full output, interactive Plotly). HTML is sanitized with
  DOMPurify; Plotly is code-split and only loaded when a figure appears.
  Outputs are runtime-only, so exports still contain graph structure and code
  only.
- Two template tracks: classical ML (scikit-learn/polars) and a **tensor path**
  (`Tensor` sockets; torch templates auto-install `torch` via PEP 723 on first
  run).
- **Accessibility**: dialogs trap focus, close on Escape and restore focus;
  ARIA roles/live regions announce parsing errors, execution status and backend
  connectivity, and icon-only controls carry accessible labels.

## Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` / `Ctrl+Y` | Redo |
| `Ctrl+C` / `Ctrl+X` / `Ctrl+V` | Copy / cut / paste selected nodes (internal edges included) |
| `Ctrl+D` | Duplicate the selection |
| `Ctrl+A` | Select all nodes |
| `Ctrl+G` / `Ctrl+Shift+G` | Group selected nodes in a frame / ungroup selected frames |
| `Ctrl+S` | Save the project, prompting for a name the first time |
| `Ctrl`+`+` / `Ctrl`+`-` | Zoom in / out |
| `Delete` / `Backspace` | Delete selected nodes and edges |
| `Ctrl`+drag | Box-select nodes (hold `Ctrl` and drag on empty canvas) |
| Double-click empty canvas | Open the node search |
| Right-click empty canvas | Open the canvas/File menu |
| Double-click node | Edit code, enter a subgraph or rename a frame |

Canvas shortcuts are ignored while the code/diff modals are open or focus is
inside a text field; `Ctrl+S` still works in fields so a save is never lost.
History is kept per workflow window for the session and is not written to
`.cuigraph.json` exports. On macOS, `Cmd` works in place of `Ctrl`.

## Development

The dev server proxies `/api` and `/ws` to `http://127.0.0.1:8000`, so start
the backend first:

```sh
# terminal 1 — backend
cd ../backend
uv run uvicorn app.main:app --reload

# terminal 2 — frontend
npm install
npm run dev            # http://localhost:5173
```

## Verification

```sh
npm run lint           # ESLint (Babel parser; TypeScript 7 has no JS compiler API)
npm run typecheck      # tsc --noEmit (strict)
npm test               # vitest
npm run build          # typecheck + production build
```

Browser smoke tests use Playwright and start both servers automatically:

```sh
npx playwright install chromium   # once per machine
npm run e2e                       # headless Chromium
npm run e2e:ui                    # interactive runner
```

The E2E suite covers opening/running bundled workflows in tabs, updating
settings, double-click node search versus the right-click File menu, exporting
Python/notebook artifacts, saving/reopening a named project with a silent
version snapshot, uploading/previewing/deleting a dataset, aligning a
multi-selection with undo, grouping/ungrouping a frame with undo/redo, creating,
reopening and running a reusable nested subgraph, inspecting a completed run in
run history, and approving modified scratch code. CI
(`.github/workflows/ci.yml`) runs backend checks, frontend checks, and
Playwright; the optional ML overlay workflow also exercises torch, xgboost, and
lightgbm.

