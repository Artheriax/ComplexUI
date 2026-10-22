import { useEffect, useMemo, useState } from "react";

import { outputRawUrl } from "../lib/api";
import type { OutputSummary } from "../lib/api";
import {
  formatDatasetDate,
  formatDatasetSize,
} from "../lib/datasets";
import { useGraphStore } from "../store/useGraphStore";
import { useOutputsStore } from "../store/useOutputsStore";
import { DatasetPreviewDialog } from "./DatasetPreviewDialog";
import {
  EyeIcon,
  ResetIcon,
  SpinnerIcon,
  TrashIcon,
  WarningIcon,
} from "./icons";

function OutputCard({
  output,
  onPreview,
}: {
  output: OutputSummary;
  onPreview: (output: OutputSummary) => void;
}) {
  const removeOutput = useOutputsStore((state) => state.removeOutput);

  const remove = () => {
    if (!window.confirm(`Delete “${output.name}” from the output library?`)) {
      return;
    }
    void removeOutput(output.id);
  };

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900">
      <button
        type="button"
        onClick={() => onPreview(output)}
        className="block h-24 w-full bg-zinc-950/70 text-left"
        title={`Preview ${output.name}`}
      >
        {output.kind === "image" ? (
          <img
            src={outputRawUrl(output.path)}
            alt={output.name}
            loading="lazy"
            className="h-24 w-full object-cover"
          />
        ) : (
          <span className="flex h-24 w-full items-center justify-center text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
            Video · {output.format}
          </span>
        )}
      </button>
      <div className="p-2">
        <p className="truncate text-xs font-semibold text-zinc-100">
          {output.name}
        </p>
        <p className="mt-0.5 truncate text-[10px] text-zinc-500" title={output.path}>
          {output.path}
        </p>
        <p className="mt-1 text-[10px] text-zinc-600">
          {output.kind} · {formatDatasetSize(output.size_bytes)} ·{" "}
          {formatDatasetDate(output.modified_at)}
        </p>
        <div className="mt-2 flex items-center justify-end gap-1">
          <button
            type="button"
            title={`Preview ${output.name}`}
            aria-label={`Preview ${output.name}`}
            onClick={() => onPreview(output)}
            className="flex items-center gap-1 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            <EyeIcon className="h-3 w-3" /> Preview
          </button>
          <button
            type="button"
            title={`Delete ${output.name}`}
            aria-label={`Delete ${output.name}`}
            onClick={remove}
            className="rounded border border-zinc-700 p-1 text-zinc-500 hover:border-red-800 hover:bg-red-950/40 hover:text-red-300"
          >
            <TrashIcon className="h-3 w-3" />
          </button>
        </div>
      </div>
    </div>
  );
}

export function OutputsLibrary() {
  const outputs = useOutputsStore((state) => state.outputs);
  const outputsLoaded = useOutputsStore((state) => state.outputsLoaded);
  const outputsLoading = useOutputsStore((state) => state.outputsLoading);
  const outputsError = useOutputsStore((state) => state.outputsError);
  const loadOutputs = useOutputsStore((state) => state.loadOutputs);
  const executionStatus = useGraphStore((state) => state.execution.status);
  const [query, setQuery] = useState("");
  const [previewOutput, setPreviewOutput] = useState<OutputSummary | null>(null);

  useEffect(() => {
    void loadOutputs(true);
  }, [loadOutputs]);

  useEffect(() => {
    if (executionStatus === "completed") {
      void loadOutputs(true);
    }
  }, [executionStatus, loadOutputs]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return outputs;
    }
    return outputs.filter((output) =>
      `${output.name} ${output.path} ${output.format} ${output.kind}`
        .toLowerCase()
        .includes(needle),
    );
  }, [outputs, query]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-1.5 border-b border-zinc-800 p-2.5">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search outputs…"
          aria-label="Search outputs"
          className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-cyan-600"
        />
        <button
          type="button"
          title="Refresh outputs"
          aria-label="Refresh outputs"
          onClick={() => void loadOutputs(true)}
          disabled={outputsLoading}
          className="rounded border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800 disabled:cursor-wait"
        >
          <ResetIcon
            className={`h-3.5 w-3.5 ${outputsLoading ? "animate-spin" : ""}`}
          />
        </button>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-2.5">
        {outputsLoading && !outputsLoaded && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Loading
            outputs…
          </div>
        )}
        {outputsError && (
          <div className="flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
            <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{outputsError}</span>
          </div>
        )}
        {outputsLoaded && !outputsError && outputs.length === 0 && (
          <p className="px-1 py-2 text-[11px] leading-relaxed text-zinc-600">
            No generated outputs yet. Save an image or video from a run and it
            will appear here.
          </p>
        )}
        {outputsLoaded && outputs.length > 0 && visible.length === 0 && (
          <p className="px-1 py-2 text-[11px] text-zinc-600">
            No outputs match “{query.trim()}”.
          </p>
        )}
        {visible.map((output) => (
          <OutputCard
            key={output.id}
            output={output}
            onPreview={setPreviewOutput}
          />
        ))}
      </div>
      <div className="border-t border-zinc-800 p-2.5 text-[10px] leading-relaxed text-zinc-600">
        Generated media lives in <span className="font-mono">outputs/</span>.
        Refreshes automatically when a run completes.
      </div>
      {previewOutput && (
        <DatasetPreviewDialog
          dataset={{
            id: previewOutput.id,
            name: previewOutput.name,
            path: previewOutput.path,
            format: previewOutput.format,
            kind: previewOutput.kind,
            size_bytes: previewOutput.size_bytes,
            modified_at: previewOutput.modified_at,
          }}
          rawUrl={outputRawUrl(previewOutput.path)}
          onClose={() => setPreviewOutput(null)}
        />
      )}
    </div>
  );
}
