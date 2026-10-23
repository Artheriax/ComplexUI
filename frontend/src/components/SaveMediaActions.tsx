import { useState } from "react";

import { savedMediaRawUrl } from "../lib/api";
import type { DatasetSummary } from "../lib/api";
import { useMediaActions } from "../hooks/useMediaActions";
import { useGraphStore } from "../store/useGraphStore";
import type { NodeExecutionState, PreviewValue } from "../types/graph";
import { DatasetPreviewDialog } from "./DatasetPreviewDialog";
import {
  CopyIcon,
  EyeIcon,
  FolderIcon,
  NewWindowIcon,
  SaveIcon,
} from "./icons";

interface SaveMediaActionsProps {
  nodeId: string;
  values: Record<string, unknown>;
  execution?: NodeExecutionState;
  running: boolean;
  isTrusted: boolean;
  hasMediaInput: boolean;
}

function savedKind(
  preview: PreviewValue | undefined,
  path: string | null,
): "image" | "video" | null {
  if (typeof preview === "object" && preview !== null && "kind" in preview) {
    if (preview.kind === "video") {
      return "video";
    }
    if (preview.kind === "image" || preview.kind === "image_batch") {
      return "image";
    }
  }
  if (path !== null) {
    return /\.(mp4|webm|mov|avi|mkv|m4v)$/i.test(path) ? "video" : "image";
  }
  return null;
}

function fileLabel(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? "saved output";
}

function fileExtension(path: string): string {
  const match = /\.([A-Za-z0-9]+)$/.exec(path);
  return match ? match[1].toUpperCase() : "MEDIA";
}

export function SaveMediaActions({
  nodeId,
  values,
  execution,
  running,
  isTrusted,
  hasMediaInput,
}: SaveMediaActionsProps) {
  const runGraph = useGraphStore((state) => state.runGraph);
  const updateWidgetValue = useGraphStore(
    (state) => state.updateWidgetValue,
  );
  const [previewOpen, setPreviewOpen] = useState(false);

  const rawSavedPath = execution?.outputs?.saved_path;
  const savedPath =
    typeof rawSavedPath === "string" && rawSavedPath.trim()
      ? rawSavedPath
      : null;
  const savedPreview = execution?.outputs?.saved;
  const kind = savedKind(savedPreview, savedPath);
  const actions = useMediaActions(savedPath, kind);
  const filename = typeof values.filename === "string" ? values.filename : "";
  const filenameValid =
    filename.trim().length > 0 && /%counter%/i.test(filename);
  const disabledReason = running
    ? "The workflow is already running."
    : !isTrusted
      ? "Review and approve this node's code first."
      : !hasMediaInput
        ? "Connect a media input before saving."
        : !filenameValid
          ? "Filename template must include %counter%."
          : null;

  const save = () => {
    updateWidgetValue(
      nodeId,
      "save_nonce",
      Number(values.save_nonce ?? 0) + 1,
    );
    void runGraph([nodeId]);
  };

  const label = savedPath ? fileLabel(savedPath) : "saved output";
  const previewDataset: DatasetSummary | null = savedPath
    ? {
        id: label,
        name: label,
        path: savedPath,
        format: fileExtension(savedPath),
        kind: kind === "video" ? "video" : "image",
        size_bytes: 0,
        modified_at: new Date().toISOString(),
      }
    : null;

  return (
    <div className="shrink-0 border-t border-zinc-800 px-2 py-1.5">
      <div className="flex flex-wrap items-center gap-1">
        <button
          type="button"
          className="nodrag flex items-center gap-1 rounded border border-cyan-800 bg-cyan-950/50 px-1.5 py-0.5 text-[10px] text-cyan-100 hover:bg-cyan-900/60 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-600"
          disabled={disabledReason !== null}
          title={disabledReason ?? "Run this node and save its media"}
          onClick={save}
        >
          <SaveIcon className="h-3 w-3" /> Save
        </button>
        <button
          type="button"
          className="nodrag flex items-center gap-1 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-600"
          disabled={savedPath === null}
          title={savedPath ? "Preview saved media" : "Save the node first"}
          onClick={() => setPreviewOpen(true)}
        >
          <EyeIcon className="h-3 w-3" /> Preview
        </button>
        {kind === "image" && (
          <button
            type="button"
            className="nodrag rounded border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:text-zinc-700"
            aria-label={`Copy ${label} to clipboard`}
            title="Copy image"
            disabled={savedPath === null || actions.busy}
            onClick={actions.copy}
          >
            <CopyIcon className="h-3 w-3" />
          </button>
        )}
        <button
          type="button"
          className="nodrag rounded border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:text-zinc-700"
          aria-label={`Open ${label}`}
          title="Open in default app"
          disabled={savedPath === null || actions.busy}
          onClick={actions.open}
        >
          <NewWindowIcon className="h-3 w-3" />
        </button>
        <button
          type="button"
          className="nodrag rounded border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:text-zinc-700"
          aria-label={`Show ${label} in folder`}
          title="Show in folder"
          disabled={savedPath === null || actions.busy}
          onClick={actions.reveal}
        >
          <FolderIcon className="h-3 w-3" />
        </button>
      </div>
      {savedPath !== null && (
        <p
          className="mt-1 truncate font-mono text-[9px] text-zinc-600"
          title={savedPath}
        >
          {savedPath}
        </p>
      )}
      {actions.state.kind === "working" && (
        <p role="status" className="mt-0.5 text-[9px] text-zinc-500">
          Working…
        </p>
      )}
      {actions.state.kind === "success" && (
        <p role="status" className="mt-0.5 text-[9px] text-emerald-400">
          {actions.state.message}
        </p>
      )}
      {actions.state.kind === "error" && (
        <p role="alert" className="mt-0.5 text-[9px] text-red-400">
          {actions.state.message}
        </p>
      )}
      {previewOpen && previewDataset !== null && (
        <DatasetPreviewDialog
          dataset={previewDataset}
          rawUrl={savedMediaRawUrl(savedPath ?? "")}
          onClose={() => setPreviewOpen(false)}
        />
      )}
    </div>
  );
}
