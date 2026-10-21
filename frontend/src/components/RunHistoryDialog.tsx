import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { fetchRun, fetchRuns } from "../lib/api";
import type {
  RunDetail,
  RunErrorRecord,
  RunNodeRecord,
  RunSummary,
} from "../lib/api";
import { PreviewValueView } from "./PreviewValueView";
import { RichOutputView } from "./RichOutputView";
import {
  CheckIcon,
  CloseIcon,
  HistoryIcon,
  SpinnerIcon,
  WarningIcon,
} from "./icons";

function formatDate(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatDuration(durationMs: number | null): string {
  if (durationMs === null) {
    return "not run";
  }
  if (durationMs < 1_000) {
    return `${Math.round(durationMs)} ms`;
  }
  return `${(durationMs / 1_000).toFixed(2)} s`;
}

const runStatusClass: Record<string, string> = {
  completed: "bg-emerald-950 text-emerald-300",
  failed: "bg-red-950 text-red-300",
  cancelled: "bg-amber-950 text-amber-300",
};

const nodeStatusClass: Record<string, string> = {
  completed: "text-emerald-400",
  error: "text-red-400",
  pending: "text-zinc-600",
};

function RunErrorCard({
  error,
  label,
}: {
  error: RunErrorRecord;
  label: string | null;
}) {
  return (
    <div className="rounded-lg border border-red-900/70 bg-red-950/40 p-2.5">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-red-300">
        <WarningIcon className="h-3.5 w-3.5" />
        {error.error_type}
        {error.line !== null && (
          <span className="font-normal">line {error.line}</span>
        )}
      </div>
      <p className="mt-1 text-[11px] leading-snug text-red-200">
        {label ? `[${label}] ` : ""}
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

function RunNodeCard({ node }: { node: RunNodeRecord }) {
  const hasOutputs =
    Object.keys(node.previews).length > 0 || node.rich_outputs.length > 0;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-2.5">
      <div className="flex items-center gap-2">
        {node.status === "completed" ? (
          <CheckIcon
            className={`h-3.5 w-3.5 ${nodeStatusClass[node.status]}`}
          />
        ) : node.status === "error" ? (
          <WarningIcon
            className={`h-3.5 w-3.5 ${nodeStatusClass[node.status]}`}
          />
        ) : (
          <span
            aria-hidden="true"
            className={`h-2 w-2 rounded-full bg-zinc-700 ${nodeStatusClass[node.status]}`}
          />
        )}
        <span className="truncate text-xs font-semibold text-zinc-200">
          {node.label}
        </span>
        <span className="ml-auto shrink-0 text-[10px] text-zinc-500">
          {formatDuration(node.duration_ms)}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[9px] text-zinc-500">
        <span className={nodeStatusClass[node.status]}>{node.status}</span>
        {node.cached && (
          <span className="rounded bg-zinc-800 px-1 py-px text-zinc-400">
            cached
          </span>
        )}
      </div>
      {node.error && (
        <div className="mt-2">
          <RunErrorCard error={node.error} label={null} />
        </div>
      )}
      {hasOutputs && (
        <div className="mt-2 space-y-2 border-t border-zinc-800 pt-2">
          {Object.entries(node.previews).map(([outputName, value]) => (
            <div key={outputName}>
              <div className="mb-0.5 font-mono text-[10px] text-cyan-300/80">
                {outputName}
              </div>
              <PreviewValueView value={value} compact />
            </div>
          ))}
          {node.rich_outputs.map((output, index) => (
            <div key={index}>
              <div className="mb-0.5 font-mono text-[9px] uppercase text-zinc-500">
                {output.source}
              </div>
              <RichOutputView output={output} compact />
            </div>
          ))}
          {node.rich_outputs_truncated && (
            <p className="text-[10px] italic text-amber-400/80">
              Rich outputs omitted to keep this run record within its storage
              limit.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function RunDetailPane({
  detail,
  loading,
  error,
}: {
  detail: RunDetail | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 p-2 text-xs text-zinc-500">
        <SpinnerIcon className="h-3.5 w-3.5 animate-spin" />
        Loading run…
      </div>
    );
  }
  if (error) {
    return (
      <p
        role="alert"
        className="rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300"
      >
        {error}
      </p>
    );
  }
  if (!detail) {
    return (
      <p className="p-2 text-[11px] text-zinc-500">
        Select a run to inspect its node statuses, timings and outputs.
      </p>
    );
  }
  const errorLabel = detail.error?.node_id
    ? (detail.node_labels[detail.error.node_id] ?? detail.error.node_id)
    : null;

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${runStatusClass[detail.status]}`}
          >
            {detail.status}
          </span>
          <span className="font-mono text-[10px] text-zinc-500">
            {detail.run_id}
          </span>
        </div>
        <p className="mt-2 text-[11px] text-zinc-300">
          {formatDate(detail.started_at)} · {formatDuration(detail.duration_ms)}
        </p>
        <p className="mt-0.5 text-[10px] text-zinc-500">
          {detail.completed_count}/{detail.node_count} nodes completed
          {detail.cached_count > 0
            ? ` · ${detail.cached_count} cached`
            : ""}
          {detail.targets ? ` · scoped to ${detail.targets.length} target(s)` : ""}
        </p>
        {detail.order.length > 0 && (
          <p className="mt-1 break-words text-[10px] text-zinc-600">
            order: {detail.order.join(" → ")}
          </p>
        )}
      </div>

      {detail.error && (
        <RunErrorCard error={detail.error} label={errorLabel} />
      )}

      <div className="space-y-2">
        {detail.nodes.map((node) => (
          <RunNodeCard key={node.node_id} node={node} />
        ))}
      </div>
    </div>
  );
}

export function RunHistoryDialog({ onClose }: { onClose: () => void }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const detailRequest = useRef(0);
  const dialogRef = useDialogA11y<HTMLDivElement>(onClose);

  const loadDetail = useCallback(async (runId: string) => {
    const request = ++detailRequest.current;
    setLoadingDetail(true);
    setDetailError(null);
    try {
      const loaded = await fetchRun(runId);
      if (request === detailRequest.current) {
        setDetail(loaded);
      }
    } catch (cause) {
      if (request === detailRequest.current) {
        setDetailError(cause instanceof Error ? cause.message : String(cause));
      }
    } finally {
      if (request === detailRequest.current) {
        setLoadingDetail(false);
      }
    }
  }, []);

  const loadRuns = useCallback(
    async (preserveSelection: string | null) => {
      try {
        const loaded = await fetchRuns();
        setRuns(loaded);
        setListError(null);
        const nextId =
          preserveSelection &&
          loaded.some((run) => run.run_id === preserveSelection)
            ? preserveSelection
            : (loaded[0]?.run_id ?? null);
        setSelectedId(nextId);
        if (nextId) {
          await loadDetail(nextId);
        } else {
          setDetail(null);
          setDetailError(null);
        }
      } catch (cause) {
        setListError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setLoadingRuns(false);
      }
    },
    [loadDetail],
  );

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const loaded = await fetchRuns();
        if (!active) {
          return;
        }
        setRuns(loaded);
        setListError(null);
        const nextId = loaded[0]?.run_id ?? null;
        setSelectedId(nextId);
        if (nextId) {
          await loadDetail(nextId);
        } else {
          setDetail(null);
          setDetailError(null);
        }
      } catch (cause) {
        if (active) {
          setListError(cause instanceof Error ? cause.message : String(cause));
        }
      } finally {
        if (active) {
          setLoadingRuns(false);
        }
      }
    })();
    return () => {
      active = false;
      detailRequest.current += 1;
    };
  }, [loadDetail]);

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="run-history-title"
        aria-describedby="run-history-description"
        tabIndex={-1}
        className="flex max-h-[82vh] w-full max-w-4xl flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-start justify-between gap-4 border-b border-zinc-800 p-4">
          <div>
            <h2
              id="run-history-title"
              className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100"
            >
              <HistoryIcon className="h-4 w-4 text-cyan-400" />
              Run history
            </h2>
            <p
              id="run-history-description"
              className="mt-1 text-[11px] text-zinc-500"
            >
              Terminal runs are retained in the workspace so you can inspect
              statuses, timings, outputs and failures after execution.
            </p>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              disabled={loadingRuns}
              onClick={() => {
                setLoadingRuns(true);
                void loadRuns(selectedId);
              }}
              className="rounded-md border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-600"
            >
              {loadingRuns ? "Refreshing…" : "Refresh"}
            </button>
            <button
              type="button"
              title="Close run history"
              aria-label="Close run history"
              onClick={onClose}
              className="rounded-md border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800"
            >
              <CloseIcon />
            </button>
          </div>
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[260px_1fr]">
          <div className="min-h-0 overflow-y-auto border-b border-zinc-800 p-2 md:border-b-0 md:border-r">
            {listError && (
              <p
                role="alert"
                className="mb-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300"
              >
                {listError}
              </p>
            )}
            {loadingRuns && runs.length === 0 && (
              <div className="flex items-center gap-2 p-2 text-xs text-zinc-500">
                <SpinnerIcon className="h-3.5 w-3.5 animate-spin" />
                Loading runs…
              </div>
            )}
            {!loadingRuns && !listError && runs.length === 0 && (
              <div className="p-2">
                <p className="text-[11px] text-zinc-500">
                  No runs recorded yet. Run the workflow and its terminal result
                  will appear here.
                </p>
              </div>
            )}
            <div className="space-y-1.5">
              {runs.map((run) => (
                <button
                  key={run.run_id}
                  type="button"
                  onClick={() => {
                    setSelectedId(run.run_id);
                    setDetail(null);
                    void loadDetail(run.run_id);
                  }}
                  className={`block w-full rounded-lg border p-2 text-left transition-colors ${
                    run.run_id === selectedId
                      ? "border-cyan-700 bg-cyan-950/30"
                      : "border-zinc-800 bg-zinc-950/40 hover:border-zinc-700"
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`rounded px-1 py-px text-[9px] font-medium ${runStatusClass[run.status]}`}
                    >
                      {run.status}
                    </span>
                    <span className="ml-auto text-[9px] text-zinc-600">
                      {formatDuration(run.duration_ms)}
                    </span>
                  </div>
                  <p className="mt-1 truncate text-[11px] text-zinc-300">
                    {formatDate(run.started_at)}
                  </p>
                  <p className="mt-0.5 text-[9px] text-zinc-500">
                    {run.completed_count}/{run.node_count} nodes
                    {run.cached_count > 0 ? ` · ${run.cached_count} cached` : ""}
                  </p>
                  {run.error_type && (
                    <p className="mt-0.5 truncate text-[9px] text-red-400">
                      {run.error_type}
                    </p>
                  )}
                </button>
              ))}
            </div>
          </div>
          <div className="min-h-0 overflow-y-auto p-3">
            <RunDetailPane
              detail={detail}
              loading={loadingDetail}
              error={detailError}
            />
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
