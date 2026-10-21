import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import type { AssetKind } from "../lib/api";
import { assetAccept, formatDatasetSize } from "../lib/datasets";
import { useDatasetsStore } from "../store/useDatasetsStore";
import { CloseIcon, SpinnerIcon, UploadIcon, WarningIcon } from "./icons";

interface DatasetPickerDialogProps {
  currentPath: string;
  onSelect: (path: string) => void;
  onClose: () => void;
  assetKind?: AssetKind;
}

const KIND_LABELS: Record<AssetKind, string> = {
  table: "table",
  image: "image",
  video: "video",
};

export function DatasetPickerDialog({
  currentPath,
  onSelect,
  onClose,
  assetKind = "table",
}: DatasetPickerDialogProps) {
  const datasets = useDatasetsStore((state) => state.datasets);
  const datasetsLoaded = useDatasetsStore((state) => state.datasetsLoaded);
  const datasetsLoading = useDatasetsStore((state) => state.datasetsLoading);
  const datasetsError = useDatasetsStore((state) => state.datasetsError);
  const uploading = useDatasetsStore((state) => state.uploading);
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets);
  const uploadFiles = useDatasetsStore((state) => state.uploadFiles);
  const [path, setPath] = useState(currentPath);
  const dialogRef = useDialogA11y<HTMLDivElement>(onClose);
  const visible = datasets.filter((dataset) => dataset.kind === assetKind);

  useEffect(() => {
    void loadDatasets();
  }, [loadDatasets]);

  const submit = () => {
    onSelect(path.trim());
    onClose();
  };

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-picker-title"
        tabIndex={-1}
        className="flex max-h-[76vh] w-full max-w-2xl flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-start justify-between gap-4 border-b border-zinc-800 p-4">
          <div>
            <h2
              id="dataset-picker-title"
              className="text-sm font-semibold text-zinc-100"
            >
              Choose {KIND_LABELS[assetKind]} asset
            </h2>
            <p className="mt-1 text-[11px] text-zinc-500">
              Pick a workspace upload or enter a workspace-relative path.
            </p>
          </div>
          <button
            type="button"
            title="Close dataset picker"
            aria-label="Close dataset picker"
            onClick={onClose}
            className="rounded-md border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          <label className="mb-2 flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/50 px-2 py-1.5">
            <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
              Path
            </span>
            <input
              type="text"
              aria-label="File path"
              value={path}
              placeholder={
                assetKind === "image"
                  ? "data/uploads/photo.png"
                  : assetKind === "video"
                    ? "data/uploads/clip.mp4"
                    : "data/uploads/data.csv"
              }
              onChange={(event) => setPath(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  submit();
                }
              }}
              className="min-w-0 flex-1 bg-transparent font-mono text-xs text-zinc-200 outline-none placeholder:text-zinc-700"
            />
            <button
              type="button"
              onClick={submit}
              className="shrink-0 rounded border border-cyan-800 bg-cyan-950/40 px-2 py-0.5 text-[10px] text-cyan-200 hover:bg-cyan-900/50"
            >
              Use path
            </button>
          </label>

          <label
            className={`mb-2 flex cursor-pointer items-center justify-center gap-1.5 rounded-lg border border-dashed px-2 py-2 text-[11px] ${
              uploading
                ? "cursor-wait border-zinc-800 text-zinc-600"
                : "border-cyan-900/70 text-cyan-200 hover:bg-cyan-950/20"
            }`}
          >
            {uploading ? (
              <SpinnerIcon className="h-3 w-3 animate-spin" />
            ) : (
              <UploadIcon className="h-3 w-3" />
            )}
            {uploading ? "Uploading…" : `Upload ${KIND_LABELS[assetKind]}`}
            <input
              type="file"
              multiple
              accept={assetAccept(assetKind)}
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

          {datasetsError && (
            <div className="mb-2 flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
              <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{datasetsError}</span>
            </div>
          )}
          {datasetsLoading && !datasetsLoaded && (
            <div className="flex items-center gap-2 p-2 text-xs text-zinc-500">
              <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Loading
              datasets…
            </div>
          )}
          {!datasetsLoading && visible.length === 0 && (
            <p className="p-2 text-[11px] text-zinc-600">
              No uploaded {KIND_LABELS[assetKind]} assets. Upload a file above or
              enter a path.
            </p>
          )}
          <div className="space-y-1.5">
            {visible.map((dataset) => (
              <button
                key={dataset.id}
                type="button"
                onClick={() => {
                  onSelect(dataset.path);
                  onClose();
                }}
                className="flex w-full items-center justify-between gap-3 rounded-lg border border-zinc-800 bg-zinc-950/40 px-2.5 py-2 text-left hover:border-cyan-800 hover:bg-zinc-800/60"
              >
                <span className="min-w-0">
                  <span className="block truncate text-xs font-medium text-zinc-200">
                    {dataset.name}
                  </span>
                  <span className="mt-0.5 block truncate text-[10px] text-zinc-600">
                    {dataset.path}
                  </span>
                </span>
                <span className="shrink-0 text-[10px] text-zinc-500">
                  {dataset.format} · {formatDatasetSize(dataset.size_bytes)}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
