# ComplexUI Frontend

React 19 + TypeScript (strict) + XYFlow + Monaco + Zustand + TailwindCSS v4.

- `src/types/graph.ts` — TypeScript mirrors of the backend Pydantic contracts.
- `src/lib/api.ts` — REST client (`/api/*` via the Vite dev proxy), including
  export, dataset upload/list/preview/delete (multipart uploads use a dedicated
  FormData request), repo model-library listing, named-project/version,
  run-history, generated-output raw/open/reveal/delete, and recovery-session
  requests.
- `src/lib/clipboard.ts` — copies generated images to the browser clipboard,
  converting non-PNG outputs through a canvas and reporting unsupported
  browsers clearly.
- `src/lib/outputNames.ts` — Save Media token expansion, filename
  sanitization, batch-index insertion and the "Will save as" preview.
- `src/lib/widgetVisibility.ts` — `ShowIf` condition matching for conditional
  widget rows and handles.
- `src/hooks/useMediaActions.ts` — shared copy/open/reveal actions and status
  state for output files (including external Save Media destinations).
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
- `src/store/useOutputsStore.ts` — generated-output gallery list, refresh,
  deletion and error state.
- `src/lib/modelMetadata.ts` — factual model tag/size/parameter/memory
  formatters shared by the Models Library and MODEL previews.
- `src/store/useModelsStore.ts` — repo model library cache by type, refresh
  coalescing, stale-selection lookup and load state.
- `src/store/useEdgeDragStore.ts` / `useConnectionGuideStore.ts` — transient
  connected-input retarget drag state and compatible-handle highlighting.
- `src/hooks/useModelsAutoRefresh.ts` — visibility-aware polling and the R
  refresh shortcut for open model views.
- `src/hooks/useSystemMetrics.ts` — system-monitor polling with visibility
  pausing, retry scheduling, and abort handling.
- `src/lib/systemMetrics.ts` — byte/percent/temperature formatting for the
  monitor.
- `src/hooks/useHealthInfo.ts` — backend runtime diagnostics
  (accelerator/torch/CUDA/NVIDIA/limits) for Settings.
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
  frames, subgraph instances, shared interface managers and the Save Media
  folder/filename editor), `SaveMediaFields` (native Browse, tokens, preview,
  format/quality controls), `SaveMediaActions` (Save/Preview/Open/Copy/Folder),
  `SwitchFields` (renamed five-input selector), `FolderPathField` (typed path
  plus native folder Browse), `CanvasContextMenu`
  (pane actions with a File submenu), `CodeModal` (Monaco editor with debounced
  live parsing), `DiffModal` (security review of modified code), `NodePalette`
  (square icon rail plus Nodes / Workflows / Templates / Assets drawers),
  `NodeLibrary` (searchable templates plus reusable
  subgraphs), `AssetLibrary` (Files / Outputs / Models tabs), `OutputsLibrary`
  (generated-media gallery with copy-image/open/reveal actions), `ModelsLibrary` (read-only
  repo model browser with search, type filtering, refresh and auto-refresh),
  `ModelSelectorField` / `ModelPickerDialog` (type-scoped local model picker),
  `WorkflowLibrary` (collapsible example-workflow categories),
  `ProjectLibrary`, `DatasetLibrary` (upload, search, preview, delete),
  `OutputsLibrary` (generated-media gallery, preview, copy image, open, show
  folder, delete),
  `DatasetPickerDialog`, `DatasetPreviewDialog`, `WidgetField` (typed
  parameters, fixed choices, conditional visibility, step buttons and linked
  state), `FileWidgetField` (workspace file picker for `File` widgets),
  `SelectionToolbar` (multi-select alignment, distribution, grid layout,
  frame grouping/ungrouping and reusable subgraph creation),
  `SaveProjectDialog`, `ProjectHistoryDialog`, `ShortcutsDialog`,
  `ExecutionPanel`, `RunHistoryDialog` (retained terminal runs with per-node
  timings and outputs), `SystemMonitor` (draggable CPU/GPU/VRAM/RAM and
  GPU-temperature overlay), `PreviewValueView` (dataset, tensor, image, video,
  vision-model, detection, mask and classification previews), and
  `RichOutputView` (text, tables, images, sanitized HTML and
  static/interactive Plotly).

## Features

- A compact square-button rail expands **Nodes**, **Workflows**, **Templates**, and **Assets** drawers on demand, keeping the canvas
  wide. The same rail toggles the Execution (Console) drawer, opens the
  keyboard-shortcut reference and Settings.
- **System monitor**: a background-less, draggable overlay with CPU/GPU
  utilization, VRAM/RAM usage and GPU temperature. Position, visibility and
  refresh interval persist. Settings also exposes read-only accelerator, torch
  build, CUDA/NVIDIA, attention backend, memory profile/allocator, compilation
  mode, FP16-accumulation and execution-limit diagnostics. Run history shows
  peak VRAM for nodes that report CUDA metrics.
- **Execution behavior**: Settings controls whether the Execution panel opens
  automatically on run and whether Canvas Clear asks for confirmation.
- Image-generation and CLIP model cards expose effective attention, precision,
  offload, compilation, pre-quantization and memory-estimate tags from the
  `MODEL` wrapper preview. Quantization tags are factual checkpoint metadata;
  no per-GPU recommendation is shown.
- Categorized node library with search and collapsible sections that start
  collapsed (production templates across values, data loaders, cleaning, EDA,
  preprocessing, imbalanced data, models, evaluation, PyTorch and Computer
  Vision). EDA, evaluation, model and PyTorch nodes render Plotly charts in the
  output pane.
  Example dataset loaders and synthetic test nodes are hidden from the
  palette — they power the bundled workflows.
- **Workflow windows**: the top bar is a tab strip of open workflows. The `+`
  button creates a blank window; opening a saved workflow or bundled template
  adds a new window, and opening a project already in a tab focuses it. Each
  window keeps its own graph, viewport, undo history, execution state and
  project identity. Open windows are restored from localStorage, and only one
  backend run is allowed at a time (the running tab shows a spinner).
- **Templates drawer**: 39 ready-to-run examples with collapsible categories;
  opening one adds a fully wired graph (nodes, widgets and positions) in a new
  window.
- **Assets drawer**: **Files** tab uploads CSV, TSV, Parquet, Excel, NDJSON,
  images or videos into repo-root `inputs/` (100 MiB table cap; 500 MiB media
  cap), with search, inline previews and deletion. Loader nodes expose
  kind-filtered **File** widgets that pick an asset or accept an `inputs/...`
  path; uploads can also be made from the picker. Execution rejects absolute
  paths and symlink escapes outside the workspace, inputs, and outputs roots.
- **Save Media editor**: inline folder field with native OS Browse, filename
  template with `%date%`, `%time%`, `%counter%`, `%model%`, `%width%`,
  `%height%` and `%batch%` tokens, Clear/Reset, a live "Will save as" preview,
  PNG/JPG/WebP buttons plus extra formats, and a settings popover for model
  label/quality/compression/optimization. Metadata-only tokens are disabled
  until metadata support lands. The default folder is `%date%` under
  `outputs/`; absolute folders may be chosen. After a save, Save/Preview/Open/
  Copy/Show-folder actions are available and external files are fetched through
  run-recorded artifact endpoints.
- **Batches and folder loading**: `IMAGE_BATCH` wires into scalar `IMAGE`
  inputs with a batch chip and aggregate preview; each node runs once per
  image and reports batch-aware progress. `Load Image` exposes an
  alpha/luminance mask output alongside the RGB image. `Load Images From Folder` uses the
  `FolderPath` widget with native Browse, and the five-input `Switch` lets users
  rename inputs and pass the selected one through.
- **Conditional sizing and resize fields**: widgets with `ShowIf` conditions
  hide when their controlling mode/value does not match. Resolution and Empty
  Latent Image switch between width/height and megapixel presets (Preview
  through UHD, plus Custom). `Resize Image` and `Resize Latent` show only the
  fields for dimensions, side, megapixels or multiplier mode.
- **Outputs tab**: browse generated images/videos from repo-root `outputs/`,
  including nested date/subfolders, with search, image thumbnails, inline
  media preview, copy-image (clipboard), open, show-folder, deletion, and
  automatic refresh when a run completes. Open/show-folder are disabled when
  the backend accepts non-loopback hosts. External Save Media destinations are
  not listed here; use the Save Media node actions instead.
- **Models tab**: read-only browser for the repo-root `models/<type>` library.
  It shows format, size, modified time, type, folder badge, parameter count,
  pre-quantization format and static resident/offload estimates, with search,
  type filtering, a refresh button, **R** refresh, and automatic polling while
  open. Custom model folders and diffusers-style model directories appear
  automatically. Estimates are heuristics based on checkpoint size and metadata;
  run history remains the measured memory source.
- **Local model selector**: nodes can declare
  `local_model: ModelSelector = ModelSelector(type="yolo")`. The widget opens a
  type-scoped picker, stores a relative `models/<type>/<path>` path, warns when
  a saved file or folder disappears, and can clear back to the predefined
  built-in fallback. `select="file"|"folder"|"any"` filters the picker and is
  enforced by the backend; `required=True` removes the fallback and blocks
  execution until a local selection is made; `include=[...]` searches extra
  type folders (for example a VAE selector that also lists full diffusers
  repos). The picker shows the source model type when multiple types are
  included. Diffusion MODEL/CLIP/VAE previews show architecture, component
  classes, latent space and encoder counts; conditioning previews also show how
  many reference images are attached and whether inpainting state is present.
  Guider previews show guidance mode/value and the CFG window, and sigma previews
  include the architecture. Vision-model previews show a
  `local`/`built-in` badge, path, format and fingerprint in both live output and
  run history.
- **Optional input sockets**: inputs declared as `Optional[IMAGE] = None` (or
  `IMAGE | None = None`) render an `optional` badge and a dashed handle. They
  may stay unconnected; connected values are not expanded as image batches.
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
  stored widget value at run time. When a value node is connected to a
  `Literal` parameter, its value field inherits the same fixed-choice dropdown
  and normalizes to a valid option.
- Widget parameter connection dots are hidden until the parameter row is
  hovered/focused, the parameter is already wired, or an active connection line
  approaches the dot. Input/output socket dots remain visible.
- **Fixed choices**: a widget declared as
  `level: Literal["low", "medium", "high"] = "low"` renders as a dropdown.
  Stored defaults, workflow overrides, and values arriving over a wire are
  validated against those choices before the node runs.
- Typed sockets: connections are only allowed between equal types (or `ANY`),
  and each input/widget parameter accepts one edge. Releasing a connection on
  empty canvas opens a searchable node picker filtered to compatible sockets or
  widget parameters and creates the new node already wired (toggleable in
  Settings).
- Drag a connected input dot to move its wire to another compatible input.
  The source output stays fixed, the preview line uses the normal connection
  style, and the wire body itself is not draggable. Dropping on empty canvas
  deletes the wire and opens node search; choosing a node creates a new target
  and wire, while an invalid drop restores the wire. Outputs can fan out to
  multiple inputs.
- `Shift`-drag an output to move its whole outgoing wire bundle together onto
  another same-typed output. Compatible targets stay colored, ghost lines show
  the affected wires, and an invalid/cancelled drop restores the original
  bundle. Dropping the bundle on empty canvas opens the compatible node search
  and re-sources all wires to the new node.
- While a wire is being dragged, handles that cannot accept it (incompatible
  type, occupied input, wrong direction, or self) are greyed out; only colored
  handles are valid drop targets.
- Runnable nodes and subgraph instances expose a bottom-right resize handle.
  Explicit dimensions persist through autosave/projects, undo, duplicate and
  paste. Parameter controls keep their fixed sizes, while image and video
  previews grow to fill the available node body.
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
- Live execution telemetry: per-node status, progress bars/messages, cached
  badges and typed previews (DataFrame head, dataset shapes, model params,
  tensor shape/dtype/sample, metrics, images, videos, vision-model descriptors,
  detections, masks and classifications). **Stop** is cooperative inside a node
  that checks cancellation, in addition to the between-node check. After a
  socket reconnect the client asks `/api/status` and marks a run interrupted if
  the backend is no longer executing it.
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
- Three template tracks: classical ML (scikit-learn/polars), a **tensor path**
  (`Tensor` sockets; torch templates auto-install `torch` via PEP 723 on first
  run), and a **vision path** (`IMAGE`, `VIDEO`, `MEDIA`, model, detection,
  mask and classification sockets). Torchvision and Ultralytics dependencies
  are installed on demand per node; Ultralytics is AGPL-3.0. Local model
  selectors support `.pt`, `.pth`, `.onnx` and `.safetensors` subject to each
  template's backend; only trusted model files should be added because `.pt`
  loading can execute code.
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
| `Shift`+drag output | Move all wires from that output to another same-typed output |
| Double-click empty canvas | Open the node search |
| Right-click empty canvas | Open the canvas/File menu |
| Double-click node | Edit code, enter a subgraph or rename a frame |

Canvas shortcuts are ignored while any modal dialog is open or focus is inside a
text field outside a dialog; `Ctrl+S` still saves from ordinary inputs so work
is not lost. History is kept per workflow window for the session and is not
written to `.cuigraph.json` exports. On macOS, `Cmd` works in place of `Ctrl`.

## Development

Node.js `^20.19.0` or `>=22.12.0` is required (Vite 8's engine range);
`frontend/.nvmrc` pins 22. From a fresh clone install with `npm ci` so the
committed `package-lock.json` is authoritative. `../start.cmd dev` does this
automatically and sets both ports for you.

The dev server proxies `/api` and `/ws` to `http://127.0.0.1:8000`, so start
the backend first:

```sh
# terminal 1 — backend
cd ../backend
uv run uvicorn app.main:app --reload

# terminal 2 — frontend
npm ci
npm run dev            # http://localhost:5173
```

When the backend runs elsewhere or on another port, set
`COMPLEXUI_BACKEND_URL` (for example `http://127.0.0.1:9000`) instead of editing
`vite.config.ts`; both the HTTP proxy and the WebSocket target follow it. The
launcher sets this variable together with the backend's CORS origins whenever
`--backend-port`/`--frontend-port` override the defaults. In launcher
`frontend`-only mode an existing value is preserved, so it can target a backend
on another machine.

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
Playwright; the optional ML overlay workflow also exercises torch, torchvision,
Ultralytics, xgboost, and lightgbm.

