import { useMemo } from "react";

import { useGraphStore } from "../store/useGraphStore";
import type {
  DataFramePreview,
  DatasetPreview,
  ExecutionError,
  ModelPreview,
  PreviewValue,
  TensorPreview,
} from "../types/graph";
import { CheckIcon, SpinnerIcon, WarningIcon } from "./icons";

function isDataFramePreview(value: PreviewValue): value is DataFramePreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "rows" in value &&
    "cols" in value &&
    "head" in value
  );
}

function isDatasetPreview(value: PreviewValue): value is DatasetPreview {
  return typeof value === "object" && value !== null && "X_shape" in value;
}

function isTensorPreview(value: PreviewValue): value is TensorPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "shape" in value &&
    "dtype" in value &&
    "sample" in value
  );
}

function isModelPreview(value: PreviewValue): value is ModelPreview {
  return typeof value === "object" && value !== null && "class" in value;
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function DataFrameView({ preview }: { preview: DataFramePreview }) {
  const columns = preview.head[0] ? Object.keys(preview.head[0]) : [];
  return (
    <div>
      <div className="mb-1 text-[10px] text-zinc-500">
        {preview.rows} rows x {preview.cols} cols — first {preview.head.length}
      </div>
      <div className="overflow-x-auto rounded border border-zinc-800">
        <table className="w-full border-collapse text-[10px]">
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="border-b border-zinc-800 bg-zinc-900 px-1.5 py-0.5 text-left font-medium text-zinc-400"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.head.map((row, index) => (
              <tr key={index} className="odd:bg-zinc-900/40">
                {columns.map((column) => (
                  <td key={column} className="px-1.5 py-0.5 text-zinc-300">
                    {String(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PreviewView({ value }: { value: PreviewValue }) {
  if (isDataFramePreview(value)) {
    return <DataFrameView preview={value} />;
  }
  if (isDatasetPreview(value)) {
    return (
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-zinc-300">
        <span>X {JSON.stringify(value.X_shape)}</span>
        <span>y {JSON.stringify(value.y_shape)}</span>
        <span className="text-zinc-500">{value.dtype}</span>
      </div>
    );
  }
  if (isTensorPreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="flex flex-wrap gap-x-3">
          <span>shape {JSON.stringify(value.shape)}</span>
          <span className="text-zinc-500">{value.dtype}</span>
          <span className="text-zinc-500">{value.device}</span>
        </div>
        {value.sample.length > 0 && (
          <div className="mt-0.5 font-mono text-zinc-400">
            [{value.sample.map((item) => item.toFixed(4)).join(", ")}
            {value.sample.length >= 5 ? ", …" : ""}]
          </div>
        )}
      </div>
    );
  }
  if (isModelPreview(value)) {
    const params =
      value.params !== undefined
        ? Object.entries(value.params)
            .slice(0, 6)
            .map(([key, paramValue]) => `${key}=${JSON.stringify(paramValue)}`)
            .join(", ")
        : value.parameters !== undefined
          ? `${value.parameters} parameters · ${value.layers} tensors`
          : "";
    return (
      <div className="text-[10px] text-zinc-300">
        <span className="text-amber-300">{value.class}</span>
        {params && <div className="mt-0.5 text-zinc-500">{params}</div>}
      </div>
    );
  }
  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
        {entries.map(([key, entryValue]) => (
          <div key={key} className="col-span-2 flex justify-between gap-2">
            <dt className="text-zinc-500">{key}</dt>
            <dd className="text-zinc-200">
              {typeof entryValue === "number"
                ? formatNumber(entryValue)
                : typeof entryValue === "string"
                  ? entryValue
                  : JSON.stringify(entryValue)}
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  return <pre className="whitespace-pre-wrap text-[10px] text-zinc-300">{String(value)}</pre>;
}

function ErrorCard({ error }: { error: ExecutionError }) {
  const nodeLabel = useGraphStore(
    (state) =>
      state.nodes.find((node) => node.id === error.nodeId)?.data.label ?? null,
  );
  return (
    <div className="rounded-lg border border-red-900/70 bg-red-950/40 p-2.5">
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

  const previewEntries = Object.entries(execution.previews);

  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-zinc-800 bg-zinc-900/40">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          Execution
        </span>
        <span
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

        {previewEntries.length === 0 && execution.status !== "running" && (
          <p className="text-[11px] leading-relaxed text-zinc-600">
            Run the workflow to inspect dataset shapes, model parameters and
            evaluation metrics here.
          </p>
        )}

        {previewEntries.map(([nodeId, outputs]) => (
          <div key={nodeId} className="rounded-lg border border-zinc-800 bg-zinc-900/70 p-2.5">
            <div className="mb-1.5 flex items-center gap-1.5">
              <CheckIcon className="h-3 w-3 text-emerald-400" />
              <span className="text-xs font-semibold text-zinc-200">
                {nodeLabels[nodeId] ?? nodeId}
              </span>
              {execution.cached[nodeId] && (
                <span className="rounded bg-zinc-800 px-1 py-px text-[9px] text-zinc-400">cached</span>
              )}
            </div>
            <div className="space-y-2">
              {Object.entries(outputs).map(([outputName, value]) => (
                <div key={outputName}>
                  <div className="mb-0.5 font-mono text-[10px] text-cyan-300/80">{outputName}</div>
                  <PreviewView value={value} />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
