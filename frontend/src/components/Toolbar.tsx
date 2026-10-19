import { useRef } from "react";

import {
  downloadWorkflow,
  readWorkflowFile,
  serialiseWorkflow,
} from "../lib/persistence";
import { useGraphStore } from "../store/useGraphStore";
import {
  BoltIcon,
  DownloadIcon,
  PlayIcon,
  RedoIcon,
  SpinnerIcon,
  StopIcon,
  TrashIcon,
  UndoIcon,
  UploadIcon,
} from "./icons";

const ghostButton =
  "flex items-center gap-1.5 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600";
const iconButton =
  "flex items-center rounded-md border border-zinc-700 p-1.5 text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600";

export function Toolbar() {
  const wsConnected = useGraphStore((state) => state.wsConnected);
  const status = useGraphStore((state) => state.execution.status);
  const nodeCount = useGraphStore((state) => state.nodes.length);
  const canUndo = useGraphStore((state) => state.past.length > 0);
  const canRedo = useGraphStore((state) => state.future.length > 0);
  const runGraph = useGraphStore((state) => state.runGraph);
  const cancelRun = useGraphStore((state) => state.cancelRun);
  const undo = useGraphStore((state) => state.undo);
  const redo = useGraphStore((state) => state.redo);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const running = status === "running";
  const runDisabled = running || nodeCount === 0;

  const onExport = () => {
    const state = useGraphStore.getState();
    downloadWorkflow(serialiseWorkflow(state.nodes, state.edges));
  };

  const onImportFile = async (file: File) => {
    try {
      const saved = await readWorkflowFile(file);
      const state = useGraphStore.getState();
      if (
        state.nodes.length > 0 &&
        !window.confirm("Replace the current canvas with this workflow?")
      ) {
        return;
      }
      await state.restoreWorkflow(saved);
    } catch (error) {
      useGraphStore.setState((state) => ({
        execution: {
          ...state.execution,
          status: "idle",
          error: {
            message: error instanceof Error ? error.message : String(error),
            errorType: "Import",
            nodeId: null,
            line: null,
            traceback: "",
          },
        },
      }));
    }
  };

  const onClear = () => {
    const state = useGraphStore.getState();
    if (state.nodes.length === 0 && state.edges.length === 0) {
      return;
    }
    if (
      window.confirm(
        "Clear the canvas? Autosave will also be overwritten. You can undo this with Ctrl+Z.",
      )
    ) {
      state.clearCanvas();
    }
  };

  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-zinc-800 bg-zinc-900/70 px-3">
      <div className="flex items-center gap-1.5">
        <BoltIcon className="h-4 w-4 text-cyan-400" />
        <span className="text-sm font-semibold tracking-tight text-zinc-100">ComplexUI</span>
        <span className="hidden text-[11px] text-zinc-500 sm:inline">
          visual ML node graph
        </span>
      </div>

      <div className="flex items-center gap-1.5">
        <button
          type="button"
          className={iconButton}
          title="Undo (Ctrl+Z)"
          aria-label="Undo"
          disabled={running || !canUndo}
          onClick={undo}
        >
          <UndoIcon />
        </button>
        <button
          type="button"
          className={iconButton}
          title="Redo (Ctrl+Shift+Z)"
          aria-label="Redo"
          disabled={running || !canRedo}
          onClick={redo}
        >
          <RedoIcon />
        </button>
        <button
          type="button"
          className={ghostButton}
          title="Export the workflow as a .cuigraph.json file"
          disabled={nodeCount === 0}
          onClick={onExport}
        >
          <DownloadIcon /> Export
        </button>
        <button
          type="button"
          className={ghostButton}
          title="Import a .cuigraph.json workflow"
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadIcon /> Import
        </button>
        <button
          type="button"
          className={ghostButton}
          title="Clear the canvas"
          disabled={nodeCount === 0}
          onClick={onClear}
        >
          <TrashIcon /> Clear
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          aria-label="Import workflow file"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) {
              void onImportFile(file);
            }
          }}
        />
      </div>

      <div className="ml-auto flex items-center gap-3">
        <span
          role="status"
          aria-live="polite"
          className="flex items-center gap-1.5 text-[11px] text-zinc-500"
          title={wsConnected ? "Backend streaming connected" : "Backend streaming offline"}
        >
          <span
            aria-hidden="true"
            className={`h-2 w-2 rounded-full ${wsConnected ? "bg-emerald-400" : "bg-red-500"}`}
          />
          <span className="sr-only">
            {wsConnected ? "Backend connected" : "Backend offline"}
          </span>
          <span aria-hidden="true">backend</span>
        </span>

        <span className="text-[11px] text-zinc-600">
          {nodeCount} node{nodeCount === 1 ? "" : "s"}
        </span>

        <button
          type="button"
          disabled={runDisabled}
          onClick={() => {
            void runGraph();
          }}
          className="flex items-center gap-1.5 rounded-md border border-cyan-700 bg-cyan-950/60 px-3 py-1 text-xs font-medium text-cyan-200 transition-colors hover:bg-cyan-900/60 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-600"
        >
          {running ? (
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <PlayIcon className="h-3.5 w-3.5" />
          )}
          {running ? "Running…" : "Run workflow"}
        </button>

        {running && (
          <button
            type="button"
            title="Stop the workflow (checked between nodes)"
            onClick={() => {
              void cancelRun();
            }}
            className="flex items-center gap-1.5 rounded-md border border-red-800 bg-red-950/50 px-3 py-1 text-xs font-medium text-red-200 transition-colors hover:bg-red-900/50"
          >
            <StopIcon className="h-3.5 w-3.5" />
            Stop
          </button>
        )}
      </div>
    </header>
  );
}
