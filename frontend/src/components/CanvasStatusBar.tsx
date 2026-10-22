import { useGraphStore } from "../store/useGraphStore";
import { useSettingsStore } from "../store/useSettingsStore";
import { useUiStore } from "../store/useUiStore";
import { isRunnableNode } from "../types/graph";
import { EyeIcon, EyeOffIcon, PlayIcon, SpinnerIcon, StopIcon } from "./icons";

const compactButton =
  "flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-900/90 px-2 py-1 text-[10px] text-zinc-300 shadow-md transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600";

export function CanvasStatusBar() {
  const wsConnected = useGraphStore((state) => state.wsConnected);
  const status = useGraphStore((state) => state.execution.status);
  const nodeCount = useGraphStore((state) => state.nodes.length);
  const runnableCount = useGraphStore(
    (state) => state.nodes.filter(isRunnableNode).length,
  );
  const scopeDepth = useGraphStore((state) => state.scopeStack.length);
  const otherRunning = useGraphStore((state) =>
    state.windows.some(
      (window) =>
        window.id !== state.activeWindowId &&
        window.execution.status === "running",
    ),
  );
  const runGraph = useGraphStore((state) => state.runGraph);
  const cancelRun = useGraphStore((state) => state.cancelRun);
  const settings = useSettingsStore((state) => state.settings);
  const updateSetting = useSettingsStore((state) => state.updateSetting);

  const running = status === "running";
  const runDisabled =
    running || otherRunning || scopeDepth > 0 || runnableCount === 0;

  return (
    <div
      data-testid="canvas-status"
      className="absolute right-3 top-3 z-20 flex items-center gap-1.5"
    >
      <span
        role="status"
        aria-live="polite"
        className="flex items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900/90 px-2 py-1 text-[10px] text-zinc-500 shadow-md"
        title={
          wsConnected
            ? "Backend streaming connected"
            : "Backend streaming offline"
        }
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

      <span className="rounded-md border border-zinc-800 bg-zinc-900/90 px-2 py-1 text-[10px] text-zinc-500 shadow-md">
        {nodeCount} node{nodeCount === 1 ? "" : "s"}
      </span>

      <button
        type="button"
        onClick={() => updateSetting("showEdges", !settings.showEdges)}
        aria-pressed={settings.showEdges}
        aria-label={
          settings.showEdges
            ? "Hide connection lines"
            : "Show connection lines"
        }
        title={
          settings.showEdges
            ? "Hide connection lines"
            : "Show connection lines"
        }
        className={compactButton}
      >
        {settings.showEdges ? <EyeIcon /> : <EyeOffIcon />}
        <span>{settings.showEdges ? "Lines" : "Lines hidden"}</span>
      </button>

      {running ? (
        <button
          type="button"
          title="Stop the workflow (checked between nodes)"
          onClick={() => {
            void cancelRun();
          }}
          className="flex items-center gap-1.5 rounded-md border border-red-800 bg-red-950/70 px-2.5 py-1 text-[10px] font-medium text-red-200 shadow-md transition-colors hover:bg-red-900/60"
        >
          <StopIcon className="h-3.5 w-3.5" />
          Stop
        </button>
      ) : (
        <button
          type="button"
          title={
            otherRunning
              ? "Another workflow window is running"
              : scopeDepth > 0
                ? "Exit the subgraph to run the workflow from the root canvas"
                : "Run the workflow"
          }
          disabled={runDisabled}
          onClick={() => {
            if (settings.autoOpenConsole) {
              useUiStore.getState().setConsoleOpen(true);
            }
            void runGraph();
          }}
          className="flex items-center gap-1.5 rounded-md border border-cyan-700 bg-cyan-950/80 px-2.5 py-1 text-[10px] font-medium text-cyan-200 shadow-md transition-colors hover:bg-cyan-900/60 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-600"
        >
          <PlayIcon className="h-3.5 w-3.5" />
          Run workflow
        </button>
      )}

      {running && (
        <span className="flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/90 px-2 py-1 text-[10px] text-cyan-300 shadow-md">
          <SpinnerIcon className="h-3 w-3 animate-spin" />
          Running
        </span>
      )}
    </div>
  );
}
