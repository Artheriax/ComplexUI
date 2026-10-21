import type { MouseEvent } from "react";

import { useGraphStore } from "../store/useGraphStore";
import type { WorkflowWindow } from "../store/useGraphStore";
import { CloseIcon, PlusIcon, SpinnerIcon } from "./icons";

function documentHasContent(
  nodes: unknown[],
  edges: unknown[],
  subgraphs: object,
): boolean {
  return (
    nodes.length > 0 ||
    edges.length > 0 ||
    Object.keys(subgraphs).length > 0
  );
}

export function WindowTabs() {
  const windows = useGraphStore((state) => state.windows);
  const activeWindowId = useGraphStore((state) => state.activeWindowId);
  const activeExecution = useGraphStore((state) => state.execution);
  const activeProjectName = useGraphStore((state) => state.projectName);
  const activeNodes = useGraphStore((state) => state.nodes);
  const activeEdges = useGraphStore((state) => state.edges);
  const activeSubgraphs = useGraphStore((state) => state.subgraphs);
  const activateWorkflowWindow = useGraphStore(
    (state) => state.activateWorkflowWindow,
  );
  const closeWorkflowWindow = useGraphStore(
    (state) => state.closeWorkflowWindow,
  );
  const newWorkflowWindow = useGraphStore((state) => state.newWorkflowWindow);

  const running = (window: WorkflowWindow): boolean =>
    window.id === activeWindowId
      ? activeExecution.status === "running"
      : window.execution.status === "running";

  const requestClose = (event: MouseEvent, window: WorkflowWindow) => {
    event.stopPropagation();
    if (running(window)) {
      return;
    }
    const state = useGraphStore.getState();
    const active = window.id === activeWindowId;
    const nodes = active ? state.nodes : window.nodes;
    const edges = active ? state.edges : window.edges;
    const subgraphs = active ? state.subgraphs : window.subgraphs;
    const projectName = active ? state.projectName : window.projectName;
    const projectSlug = active ? state.projectSlug : window.projectSlug;
    if (
      documentHasContent(nodes, edges, subgraphs) &&
      projectSlug === null &&
      !globalThis.confirm(
        `Close “${projectName ?? window.title}”? Its unsaved local draft will be discarded.`,
      )
    ) {
      return;
    }
    closeWorkflowWindow(window.id);
  };

  return (
    <header
      role="banner"
      className="flex h-10 shrink-0 items-center border-b border-zinc-800 bg-zinc-950/95"
    >
      <div
        role="tablist"
        aria-label="Open workflows"
        className="flex min-w-0 flex-1 items-stretch overflow-x-auto"
      >
        {windows.map((window) => {
          const active = window.id === activeWindowId;
          const title =
            (active ? activeProjectName : window.projectName) ?? window.title;
          const isRunning = running(window);
          const unsaved =
            window.projectSlug === null &&
            documentHasContent(
              active ? activeNodes : window.nodes,
              active ? activeEdges : window.edges,
              active ? activeSubgraphs : window.subgraphs,
            );
          return (
            <div
              key={window.id}
              className={`group flex h-10 min-w-36 max-w-64 shrink-0 items-center border-r border-zinc-800 ${
                active
                  ? "bg-zinc-900 text-zinc-100"
                  : "bg-zinc-950/60 text-zinc-500 hover:bg-zinc-900/60 hover:text-zinc-300"
              }`}
            >
              <button
                type="button"
                role="tab"
                aria-selected={active}
                title={title}
                onClick={() => activateWorkflowWindow(window.id)}
                onAuxClick={(event) => {
                  if (event.button === 1) {
                    requestClose(event, window);
                  }
                }}
                className="flex min-w-0 flex-1 items-center gap-1.5 px-3 py-2 text-left text-[11px] font-medium"
              >
                {isRunning ? (
                  <SpinnerIcon className="h-3 w-3 shrink-0 animate-spin text-cyan-300" />
                ) : (
                  <span
                    aria-hidden="true"
                    className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                      active ? "bg-cyan-400" : "bg-zinc-700"
                    }`}
                  />
                )}
                <span className="truncate">{title}</span>
                {unsaved && (
                  <span
                    aria-label="Unsaved workflow"
                    title="Unsaved workflow"
                    className="h-1 w-1 shrink-0 rounded-full bg-amber-400"
                  />
                )}
              </button>
              <button
                type="button"
                aria-label={`Close ${title}`}
                title={
                  isRunning ? "Stop the run before closing" : `Close ${title}`
                }
                disabled={isRunning}
                onClick={(event) => requestClose(event, window)}
                className="mr-1 rounded p-1 text-zinc-600 opacity-0 transition-opacity hover:bg-zinc-800 hover:text-zinc-200 focus:opacity-100 disabled:cursor-not-allowed disabled:opacity-30 group-hover:opacity-100"
              >
                <CloseIcon className="h-3 w-3" />
              </button>
            </div>
          );
        })}
      </div>
      <button
        type="button"
        aria-label="New workflow window"
        title="New workflow window"
        onClick={newWorkflowWindow}
        className="mx-1 rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-cyan-200"
      >
        <PlusIcon className="h-4 w-4" />
      </button>
    </header>
  );
}
