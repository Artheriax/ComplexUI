import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { datasetRawUrl, fetchDatasetPreview } from "../lib/api";
import type { DatasetPreview, DatasetSummary } from "../lib/api";
import { formatDatasetSize } from "../lib/datasets";
import { errorMessage } from "../lib/errors";
import { CloseIcon, SpinnerIcon, WarningIcon } from "./icons";

interface DatasetPreviewDialogProps {
  dataset: DatasetSummary;
  rawUrl?: string;
  onClose: () => void;
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export function DatasetPreviewDialog({
  dataset,
  rawUrl,
  onClose,
}: DatasetPreviewDialogProps) {
  const dialogRef = useDialogA11y<HTMLDivElement>(onClose);
  const isTable = dataset.kind === "table";
  const mediaUrl = rawUrl ?? datasetRawUrl(dataset.id);
  const [loadedPreview, setLoadedPreview] = useState<DatasetPreview | null>(
    null,
  );
  const [loadError, setLoadError] = useState<{
    datasetId: string;
    message: string;
  } | null>(null);
  const preview =
    loadedPreview?.dataset.id === dataset.id ? loadedPreview : null;
  const error =
    loadError?.datasetId === dataset.id ? loadError.message : null;
  const loading = isTable && preview === null && error === null;

  useEffect(() => {
    if (!isTable) {
      return;
    }
    let active = true;
    void (async () => {
      try {
        const loaded = await fetchDatasetPreview(dataset.id);
        if (active) {
          setLoadedPreview(loaded);
          setLoadError((current) =>
            current?.datasetId === dataset.id ? null : current,
          );
        }
      } catch (cause) {
        if (active) {
          setLoadError({
            datasetId: dataset.id,
            message: errorMessage(cause),
          });
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [dataset.id, isTable]);

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-preview-title"
        tabIndex={-1}
        className="flex max-h-[82vh] w-full max-w-5xl flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-start justify-between gap-4 border-b border-zinc-800 p-4">
          <div className="min-w-0">
            <h2
              id="dataset-preview-title"
              className="truncate text-sm font-semibold text-zinc-100"
            >
              {isTable ? "Dataset" : "Asset"} preview — {dataset.name}
            </h2>
            <p className="mt-1 text-[11px] text-zinc-500">
              {dataset.kind === "table" ? dataset.format : dataset.kind} ·{" "}
              {formatDatasetSize(dataset.size_bytes)} · {dataset.path}
            </p>
          </div>
          <button
            type="button"
            title="Close dataset preview"
            aria-label="Close dataset preview"
            onClick={onClose}
            className="rounded-md border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-3">
          {loading && (
            <div className="flex items-center gap-2 p-2 text-xs text-zinc-500">
              <SpinnerIcon className="h-3.5 w-3.5 animate-spin" />
              Loading preview…
            </div>
          )}
          {error && (
            <div className="flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
              <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
          {!isTable && !error && (
            <div className="flex min-h-64 items-center justify-center rounded-lg bg-zinc-950/60 p-3">
              {dataset.kind === "image" ? (
                <img
                  src={mediaUrl}
                  alt={dataset.name}
                  className="max-h-[62vh] max-w-full rounded object-contain"
                  onError={() =>
                    setLoadError({
                      datasetId: dataset.id,
                      message: "Unable to load this image.",
                    })
                  }
                />
              ) : (
                <video
                  src={mediaUrl}
                  controls
                  preload="metadata"
                  className="max-h-[62vh] max-w-full rounded"
                  onError={() =>
                    setLoadError({
                      datasetId: dataset.id,
                      message: "Unable to load this video.",
                    })
                  }
                >
                  Your browser does not support video playback.
                </video>
              )}
            </div>
          )}
          {preview && !error && (
            <>
              {preview.columns.length === 0 ? (
                <p className="p-2 text-[11px] text-zinc-500">
                  This dataset has no columns to preview.
                </p>
              ) : (
                <table className="w-full border-collapse text-left text-[11px]">
                  <thead className="sticky top-0 bg-zinc-900">
                    <tr>
                      {preview.columns.map((column) => (
                        <th
                          key={column}
                          scope="col"
                          className="whitespace-nowrap border-b border-zinc-700 px-2 py-1.5 font-semibold text-zinc-300"
                        >
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, rowIndex) => (
                      <tr
                        key={`row-${rowIndex}`}
                        className="odd:bg-zinc-950/40"
                      >
                        {row.map((cell, cellIndex) => (
                          <td
                            key={`cell-${rowIndex}-${cellIndex}`}
                            className="max-w-64 truncate border-b border-zinc-800/70 px-2 py-1 font-mono text-zinc-400"
                            title={formatCell(cell)}
                          >
                            {formatCell(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {preview.rows.length === 0 && preview.columns.length > 0 && (
                <p className="p-2 text-[11px] text-zinc-500">
                  The dataset has columns but no data rows.
                </p>
              )}
            </>
          )}
        </div>

        <div className="border-t border-zinc-800 px-4 py-2 text-[10px] text-zinc-600">
          {isTable
            ? preview
              ? `Showing up to ${preview.row_limit} rows${
                  preview.truncated ? " (more rows available)" : ""
                }.`
              : "Previewing the first 20 rows."
            : `${dataset.kind === "image" ? "Image" : "Video"} served inline ${
                rawUrl ? "from the output library" : "from the workspace"
              }.`}
        </div>
      </div>
    </div>,
    document.body,
  );
}
