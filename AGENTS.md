# AGENTS.md — ComplexUI Master Specification & Execution Plan

## 1. System Context & Agent Directives
You are an autonomous principal engineer building **ComplexUI**, a visual, code-first node graph IDE for Machine Learning (Scikit-Learn, Polars, PyTorch). Every node on the canvas contains live, editable Python code that automatically generates UI input/output sockets and parameter widgets via AST parsing.

### Operational Rules for AI Coding Agent:
1. **Incremental Execution:** Implement strictly one milestone at a time. Do not proceed to the next milestone until all verification tests for the current milestone pass.
2. **Tooling & Commands:**
   - Python environment & package management: Always use `uv` (`uv init`, `uv add`, `uv run pytest`). Never use raw `pip` or `venv`.
   - Frontend package management: Use `npm` or `pnpm` with Vite.
3. **Code Standards:**
   - Python >= 3.12. Enforce strict type hints and Pydantic v2 models for all API payloads.
   - Frontend: React 19, TypeScript (strict mode), XYFlow (`@xyflow/react`), `@monaco-editor/react`, TailwindCSS, Zustand.
   - Zero placeholders: Never leave `# TODO` or pass statements in core logic.

---

## 2. Target Directory Structure
```text
complex-ui/
├── AGENTS.md
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI entry + WebSocket manager
│   │   ├── schemas.py           # Pydantic models for Graph, Nodes, Sockets
│   │   ├── transpiler.py        # Regex pre-processor + AST parser + wrapper
│   │   ├── executor.py          # Topological DAG sorter + state cache
│   │   ├── execution_host.py    # Isolated run process + timeout/memory guards
│   │   ├── execution_worker.py  # Child-process execution entry point
│   │   ├── exporter.py          # PEP 723 scripts + Jupyter notebooks
│   │   ├── workspace.py         # Named projects, versions, recovery sessions
│   │   ├── datasets.py          # Workspace uploads, validation, previews
│   │   ├── env_manager.py       # PEP 723 parser + `uv pip install` runner
│   │   ├── security.py          # SHA-256 template hashing & diff verifier
│   │   ├── run_history.py       # Terminal run records + retention
│   │   └── templates/           # Built-in ML node templates (.py files)
│   │       ├── load_iris.py
│   │       ├── train_test_split.py
│   │       ├── knn_train.py
│   │       └── evaluate_metrics.py
│   └── tests/
│       ├── test_transpiler.py
│       ├── test_executor.py
│       ├── test_workspace.py
│       └── test_uploads.py
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    └── src/
        ├── App.tsx
        ├── lib/
│           ├── subgraph.ts      # Nested subgraph flattening + interface helpers
│           ├── nodeSearchPlacement.ts
│           └── persistence.ts
        ├── store/
│           ├── useGraphStore.ts
│           └── useDatasetsStore.ts
        ├── components/
│           ├── Canvas.tsx
│           ├── CanvasContextMenu.tsx  # Pane actions beside the node picker
│           ├── ComplexNode.tsx  # widgets/sockets, frames, subgraph managers/instances
│           ├── CodeModal.tsx    # Monaco Editor pop-up
│           ├── DiffModal.tsx    # Security audit modal for untrusted code
│           ├── ProjectLibrary.tsx
│           ├── DatasetLibrary.tsx
│           ├── DatasetPickerDialog.tsx
│           ├── DatasetPreviewDialog.tsx
│           ├── FileWidgetField.tsx
│           ├── SelectionToolbar.tsx
│           ├── RunHistoryDialog.tsx
│           ├── SaveProjectDialog.tsx
│           └── ProjectHistoryDialog.tsx
        └── types/graph.ts