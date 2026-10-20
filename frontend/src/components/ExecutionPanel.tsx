import { useMemo } from "react";

import { useGraphStore } from "../store/useGraphStore";
import type { ExecutionError } from "../types/graph";
import { CheckIcon, SpinnerIcon, WarningIcon } from "./icons";
import { PreviewValueView } from "./PreviewValueView";
import { RichOutputView } from "./RichOutputView";

function ErrorCard({ error }: { error: ExecutionError }) {
  const nodeLabel = useGraphStore(
    (state) =>
      state.nodes.find((node) => node.id === error.nodeId)?.data.label ?? null,
  );
  return (
    <div
      role="alert"
      className="rounded-lg border border-red-900/70 bg-red-950/40 p-2.5"
    >
      <div className="flex items-center gap-1.5 text-xs font-semibold text-red-300">
        <WarningIcon className="h-3.5 w-3.5" />
        {error.errorType}
        {error.line !== null && <span className="font-normal">line {error.line}</span>}
      </div>
      <p className="mt-1 text-[11px] leading-snug text-red-200">
        {nodeLabel ? `[${nodeLabel}] ` : ""}
        {error.message}
      </p>
      {error.traceback && (
        <details className="mt-1.5">
          <summary className="cursor-pointer text-[10px] text-red-400/80 hover:text-red-300">
            Traceback
          </summary>
          <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-zinc-950/80 p-2 text-[10px] leading-snug text-red-200/90">
            {error.traceback}
          </pre>
        </details>
      )}
    </div>
  );
}

const statusBadge: Record<string, string> = {
  idle: "bg-zinc-800 text-zinc-400",
  running: "bg-cyan-950 text-cyan-300",
  completed: "bg-emerald-950 text-emerald-300",
  failed: "bg-red-950 text-red-300",
  cancelled: "bg-amber-950 text-amber-300",
};

export function ExecutionPanel() {
  const execution = useGraphStore((state) => state.execution);
  const nodes = useGraphStore((state) => state.nodes);
  const nodeLabels = useMemo(
    () => Object.fromEntries(nodes.map((node) => [node.id, node.data.label])),
    [nodes],
  );

  const outputNodeIds = useMemo(() => {
    const ids = new Set([
      ...Object.keys(execution.previews),
      ...Object.keys(execution.richOutputs),
    ]);
    const ordered = execution.order.filter((nodeId) => ids.has(nodeId));
    for (const nodeId of ids) {
      if (!ordered.includes(nodeId)) {
        ordered.push(nodeId);
      }
    }
    return ordered;
  }, [execution.order, execution.previews, execution.richOutputs]);

  return (
    <aside
      aria-label="Execution output"
      className="flex w-80 shrink-0 flex-col border-l border-zinc-800 bg-zinc-900/40"
    >
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          Execution
        </span>
        <span
          role="status"
          aria-live="polite"
          className={`ml-auto rounded px-1.5 py-0.5 text-[10px] font-medium ${statusBadge[execution.status]}`}
        >
          {execution.status}
        </span>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-2.5">
        {execution.runId && (
          <div className="text-[10px] text-zinc-600">
            run <span className="font-mono">{execution.runId.slice(0, 12)}</span>
            {execution.order.length > 0 && (
              <span className="ml-2">order: {execution.order.join(" → ")}</span>
            )}
          </div>
        )}

        {execution.status === "running" && (
          <div className="flex items-center gap-2 text-xs text-cyan-300">
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Executing workflow…
          </div>
        )}

        {execution.error && <ErrorCard error={execution.error} />}

        {execution.status === "cancelled" && (
          <p className="text-[11px] leading-relaxed text-amber-300/90">
            Execution stopped between nodes. Completed outputs stay cached and
            are reused on the next run.
          </p>
        )}

        {outputNodeIds.length === 0 && execution.status !== "running" && (
          <p className="text-[11px] leading-relaxed text-zinc-600">
            Run the workflow to inspect dataset shapes, model parameters,
            console output, tables and plots here.
          </p>
        )}

        {outputNodeIds.map((nodeId) => {
          const outputs = execution.previews[nodeId] ?? {};
          const richOutputs = execution.richOutputs[nodeId] ?? [];
          const errored = execution.nodeStates[nodeId]?.status === "error";
          return (
            <div
              key={nodeId}
              className="rounded-lg border border-zinc-800 bg-zinc-900/70 p-2.5"
            >
              <div className="mb-1.5 flex items-center gap-1.5">
                {errored ? (
                  <WarningIcon className="h-3 w-3 text-red-400" />
                ) : (
                  <CheckIcon className="h-3 w-3 text-emerald-400" />
                )}
                <span className="text-xs font-semibold text-zinc-200">
                  {nodeLabels[nodeId] ?? nodeId}
                </span>
                {execution.cached[nodeId] && (
                  <span className="rounded bg-zinc-800 px-1 py-px text-[9px] text-zinc-400">
                    cached
                  </span>
                )}
              </div>
              {Object.keys(outputs).length > 0 && (
                <div className="space-y-2">
                  {Object.entries(outputs).map(([outputName, value]) => (
                    <div key={outputName}>
                      <div className="mb-0.5 font-mono text-[10px] text-cyan-300/80">
                        {outputName}
                      </div>
                      <PreviewValueView value={value} />
                    </div>
                  ))}
                </div>
              )}
              {richOutputs.length > 0 && (
                <div
                  className={`space-y-2 ${
                    Object.keys(outputs).length > 0
                      ? "mt-2 border-t border-zinc-800 pt-2"
                      : ""
                  }`}
                >
                  {richOutputs.map((output, index) => (
                    <div key={index}>
                      <div className="mb-0.5 font-mono text-[9px] uppercase text-zinc-500">
                        {output.source}
                      </div>
                      <RichOutputView output={output} interactive />
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
