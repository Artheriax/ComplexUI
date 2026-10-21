import { useEffect, useMemo, useState } from "react";

import type { DatasetSummary } from "../lib/api";
import {
  ASSET_ACCEPT,
  formatDatasetDate,
  formatDatasetSize,
} from "../lib/datasets";
import { useDatasetsStore } from "../store/useDatasetsStore";
import { DatasetPreviewDialog } from "./DatasetPreviewDialog";
import {
  EyeIcon,
  SpinnerIcon,
  TrashIcon,
  UploadIcon,
  WarningIcon,
} from "./icons";

function DatasetCard({
  dataset,
  onPreview,
}: {
  dataset: DatasetSummary;
  onPreview: (dataset: DatasetSummary) => void;
}) {
  const removeDataset = useDatasetsStore((state) => state.removeDataset);

  const remove = () => {
    if (!window.confirm(`Delete “${dataset.name}” from the workspace?`)) {
      return;
    }
    void removeDataset(dataset.id);
  };

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-2.5">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-xs font-semibold text-zinc-100">
            {dataset.name}
          </p>
          <p
            className="mt-0.5 truncate text-[10px] text-zinc-500"
            title={dataset.path}
          >
            {dataset.path}
          </p>
        </div>
        <span className="shrink-0 rounded bg-zinc-800/80 px-1 py-px text-[9px] text-zinc-500">
          {dataset.kind === "table" ? dataset.format : dataset.kind}
        </span>
      </div>
      <p className="mt-1 text-[10px] text-zinc-600">
        {formatDatasetSize(dataset.size_bytes)} ·{" "}
        {formatDatasetDate(dataset.modified_at)}
      </p>
      <div className="mt-2 flex items-center justify-end gap-1">
        <button
          type="button"
          title={`Preview ${dataset.name}`}
          aria-label={`Preview ${dataset.name}`}
          onClick={() => onPreview(dataset)}
          className="flex items-center gap-1 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <EyeIcon className="h-3 w-3" /> Preview
        </button>
        <button
          type="button"
          title={`Delete ${dataset.name}`}
          aria-label={`Delete ${dataset.name}`}
          onClick={remove}
          className="rounded border border-zinc-700 p-1 text-zinc-500 hover:border-red-800 hover:bg-red-950/40 hover:text-red-300"
        >
          <TrashIcon className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}

export function DatasetLibrary() {
  const datasets = useDatasetsStore((state) => state.datasets);
  const datasetsLoaded = useDatasetsStore((state) => state.datasetsLoaded);
  const datasetsLoading = useDatasetsStore((state) => state.datasetsLoading);
  const datasetsError = useDatasetsStore((state) => state.datasetsError);
  const uploading = useDatasetsStore((state) => state.uploading);
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets);
  const uploadFiles = useDatasetsStore((state) => state.uploadFiles);
  const [query, setQuery] = useState("");
  const [previewDataset, setPreviewDataset] =
    useState<DatasetSummary | null>(null);

  useEffect(() => {
    void loadDatasets();
  }, [loadDatasets]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return datasets;
    }
    return datasets.filter((dataset) =>
      `${dataset.name} ${dataset.path} ${dataset.format} ${dataset.kind}`
        .toLowerCase()
        .includes(needle),
    );
  }, [datasets, query]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="space-y-1.5 border-b border-zinc-800 p-2.5">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search assets…"
          className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-cyan-600"
        />
        <label
          className={`flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-md border px-2 py-1 text-[11px] ${
            uploading
              ? "cursor-wait border-zinc-800 text-zinc-600"
              : "border-cyan-900/70 bg-cyan-950/30 text-cyan-200 hover:bg-cyan-950/50"
          }`}
        >
          {uploading ? (
            <SpinnerIcon className="h-3 w-3 animate-spin" />
          ) : (
            <UploadIcon className="h-3 w-3" />
          )}
          {uploading ? "Uploading…" : "Upload assets"}
          <input
            type="file"
            multiple
            accept={ASSET_ACCEPT}
            aria-label="Upload asset files"
            disabled={uploading}
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              event.target.value = "";
              void uploadFiles(files);
            }}
            className="sr-only"
          />
        </label>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-2.5">
        {datasetsLoading && !datasetsLoaded && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Loading
            datasets…
          </div>
        )}
        {datasetsError && (
          <div className="flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
            <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{datasetsError}</span>
          </div>
        )}
        {datasetsLoaded && !datasetsError && datasets.length === 0 && (
          <p className="px-1 py-2 text-[11px] leading-relaxed text-zinc-600">
            No assets yet. Upload tables, images, or videos, then choose one
            from a file widget.
          </p>
        )}
        {datasetsLoaded && datasets.length > 0 && visible.length === 0 && (
          <p className="px-1 py-2 text-[11px] text-zinc-600">
            No assets match “{query.trim()}”.
          </p>
        )}
        {visible.map((dataset) => (
          <DatasetCard
            key={dataset.id}
            dataset={dataset}
            onPreview={setPreviewDataset}
          />
        ))}
      </div>
      <div className="border-t border-zinc-800 p-2.5 text-[10px] leading-relaxed text-zinc-600">
        Files live in <span className="font-mono">workspace/data/uploads</span>.
        Tables are capped at 100 MiB; images and videos at 500 MiB.
      </div>
      {previewDataset && (
        <DatasetPreviewDialog
          dataset={previewDataset}
          onClose={() => setPreviewDataset(null)}
        />
      )}
    </div>
  );
}
