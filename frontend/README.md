# ComplexUI Frontend

React 19 + TypeScript (strict) + XYFlow + Monaco + Zustand + TailwindCSS v4.

- `src/types/graph.ts` — TypeScript mirrors of the backend Pydantic contracts.
- `src/lib/api.ts` — REST client (`/api/*` via the Vite dev proxy).
- `src/lib/hash.ts` — SHA-256 code hashing with the same line-ending
  normalization as `backend/app/security.py`.
- `src/lib/persistence.ts` — `.cuigraph.json` save format, size/shape
  validation, localStorage storage and file download/upload helpers.
- `src/lib/monaco.ts` — local Monaco bundling (no CDN), worker configured.
- `src/store/useGraphStore.ts` — canvas state, trust workflow, execution
  telemetry reducer, run-status resync.
- `src/hooks/useExecutionSocket.ts` — `/ws/execution` streaming with
  exponential-backoff reconnect and `/api/status` resync on every open.
- `src/hooks/useGraphShortcuts.ts` — undo/redo, clipboard and duplicate
  keyboard shortcuts for the canvas.
- `src/hooks/useViewportShortcuts.ts` — keyboard zoom for the canvas.
- `src/hooks/useWorkflowAutosave.ts` — debounced autosave of the canvas
  (restored on reload).
- `src/components/` — `Canvas`, `ComplexNode` (widgets + sockets), `CodeModal`
  (Monaco editor with debounced live parsing), `DiffModal` (security review of
  modified code), `NodePalette` (Nodes \| Workflows tabs), `NodeLibrary`
  (searchable, categorized templates), `WorkflowLibrary` (one-click example
  graphs), `ExecutionPanel`, `RichOutputView` (text, tables, images,
  sanitized HTML and static/interactive Plotly), `Toolbar`.

## Features

- Categorized node library with search and collapsible sections that start
  collapsed (production templates across data loaders, cleaning, EDA, preprocessing, imbalanced
  data, models, evaluation and PyTorch). Example dataset loaders and synthetic
  test nodes are hidden from the palette — they power the bundled workflows.
- **Workflows tab**: 12 ready-to-run examples grouped by category; loading one
  replaces the canvas with a fully wired graph (nodes, widgets and positions).
- **Undo/redo** with a 100-step history (node/edge/widget edits, drags and code
  commits), plus a **multi-node clipboard**: select several nodes and
  copy/cut/paste/duplicate them with the edges between them preserved.
- Drag official templates from the node library, or create **scratch nodes**
  and edit their Python code in place (schemas re-parse while you type).
- Typed sockets: connections are only allowed between equal types (or `ANY`).
- **Trust workflow**: modified/scratch code is diffed against its template in
  a Monaco diff editor and must be approved before it can run.
- **Persistence**: workflows autosave to localStorage (restored on reload) and
  can be exported/imported as `.cuigraph.json` files. Imports enforce the same
  node/edge/code caps as the backend, reject duplicate ids, discard corrupt
  autosaves, and re-parse schemas on load so old files always match the current
  transpiler. Restored widget values are dropped when their type no longer
  matches the parsed widget.
- Live execution telemetry: per-node status, cached badges and typed previews
  (DataFrame head, dataset shapes, model params, tensor shape/dtype/sample,
  metrics). After a socket reconnect the client asks `/api/status` and marks a
  run interrupted if the backend is no longer executing it.
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
| `Ctrl`+`+` / `Ctrl`+`-` | Zoom in / out |
| `Delete` / `Backspace` | Delete selected nodes and edges |
| `Ctrl`+drag | Box-select nodes (hold `Ctrl` and drag on empty canvas) |

Shortcuts are ignored while the code/diff modals are open or focus is inside a
text field. History lives in memory for the session and is not written to
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
