# ComplexUI Frontend

React 19 + TypeScript (strict) + XYFlow + Monaco + Zustand + TailwindCSS v4.

- `src/types/graph.ts` — TypeScript mirrors of the backend Pydantic contracts.
- `src/lib/api.ts` — REST client (`/api/*` via the Vite dev proxy).
- `src/lib/hash.ts` — SHA-256 code hashing with the same line-ending
  normalization as `backend/app/security.py`.
- `src/lib/persistence.ts` — `.cuigraph.json` save format, validation,
  localStorage storage and file download/upload helpers.
- `src/lib/monaco.ts` — local Monaco bundling (no CDN), worker configured.
- `src/store/useGraphStore.ts` — canvas state, trust workflow, execution
  telemetry reducer.
- `src/hooks/useExecutionSocket.ts` — `/ws/execution` streaming with
  exponential-backoff reconnect.
- `src/hooks/useWorkflowAutosave.ts` — debounced autosave of the canvas
  (restored on reload).
- `src/components/` — `Canvas`, `ComplexNode` (widgets + sockets), `CodeModal`
  (Monaco editor with debounced live parsing), `DiffModal` (security review of
  modified code), `NodePalette`, `ExecutionPanel`, `Toolbar`.

## Features

- Drag official templates from the node library, or create **scratch nodes**
  and edit their Python code in place (schemas re-parse while you type).
- Typed sockets: connections are only allowed between equal types (or `ANY`).
- **Trust workflow**: modified/scratch code is diffed against its template in
  a Monaco diff editor and must be approved before it can run.
- **Persistence**: workflows autosave to localStorage (restored on reload) and
  can be exported/imported as `.cuigraph.json` files. Schemas are re-parsed on
  load so old files always match the current transpiler.
- Live execution telemetry: per-node status, cached badges and typed previews
  (DataFrame head, dataset shapes, model params, tensor shape/dtype/sample,
  metrics).
- Two template tracks: classical ML (scikit-learn/polars) and a **tensor path**
  (`Tensor` sockets; torch templates auto-install `torch` via PEP 723 on first
  run).

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
npm run typecheck      # tsc --noEmit (strict)
npm test               # vitest
npm run build          # typecheck + production build
```
