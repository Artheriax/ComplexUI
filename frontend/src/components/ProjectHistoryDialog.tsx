import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { fetchProjectVersions } from "../lib/api";
import type { ProjectSummary, ProjectVersionInfo } from "../lib/api";
import { useDialogA11y } from "../hooks/useDialogA11y";
import { useGraphStore } from "../store/useGraphStore";
import { CloseIcon, SpinnerIcon } from "./icons";

interface ProjectHistoryDialogProps {
  project: ProjectSummary;
  onClose: () => void;
}

function formatDate(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatSize(bytes: number): string {
  return bytes < 1_024 ? `${bytes} B` : `${(bytes / 1_024).toFixed(1)} KB`;
}

export function ProjectHistoryDialog({
  project,
  onClose,
}: ProjectHistoryDialogProps) {
  const restoreProjectVersion = useGraphStore(
    (state) => state.restoreProjectVersion,
  );
  const [versions, setVersions] = useState<ProjectVersionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState<string | null>(null);
  const dialogRef = useDialogA11y<HTMLDivElement>(onClose);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const loaded = await fetchProjectVersions(project.slug);
        if (active) {
          setVersions(loaded);
          setError(null);
        }
      } catch (cause) {
        if (active) {
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [project.slug]);

  const restore = async (version: string) => {
    setRestoring(version);
    const ok = await restoreProjectVersion(project.slug, version);
    if (ok) {
      onClose();
    } else {
      setRestoring(null);
      setError("Could not restore this version.");
    }
  };

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="project-history-title"
        aria-describedby="project-history-description"
        tabIndex={-1}
        className="flex max-h-[70vh] w-full max-w-lg flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-start justify-between gap-4 border-b border-zinc-800 p-4">
          <div>
            <h2
              id="project-history-title"
              className="text-sm font-semibold text-zinc-100"
            >
              Version history — {project.name}
            </h2>
            <p
              id="project-history-description"
              className="mt-1 text-[11px] text-zinc-500"
            >
              Snapshots are created when a project is overwritten. Opening one
              replaces the canvas; press Ctrl+S to make it current.
            </p>
          </div>
          <button
            type="button"
            title="Close version history"
            aria-label="Close version history"
            onClick={onClose}
            className="rounded-md border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          {loading && (
            <div className="flex items-center gap-2 p-2 text-xs text-zinc-500">
              <SpinnerIcon className="h-3.5 w-3.5 animate-spin" />
              Loading versions…
            </div>
          )}
          {error && (
            <p className="rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
              {error}
            </p>
          )}
          {!loading && !error && versions.length === 0 && (
            <p className="p-2 text-[11px] text-zinc-500">
              No snapshots yet. Saving over the project will retain its previous
              contents here.
            </p>
          )}
          <div className="space-y-2">
            {versions.map((version) => (
              <div
                key={version.version}
                className="flex items-center justify-between gap-3 rounded-lg border border-zinc-800 bg-zinc-950/50 p-2.5"
              >
                <div className="min-w-0">
                  <p className="text-xs font-medium text-zinc-200">
                    {formatDate(version.saved_at)}
                  </p>
                  <p className="mt-0.5 text-[10px] text-zinc-500">
                    {version.node_count} nodes · {version.edge_count} edges ·{" "}
                    {formatSize(version.size_bytes)}
                  </p>
                </div>
                <button
                  type="button"
                  disabled={restoring !== null}
                  onClick={() => {
                    void restore(version.version);
                  }}
                  className="shrink-0 rounded-md border border-cyan-800 bg-cyan-950/40 px-2.5 py-1 text-[11px] text-cyan-200 hover:bg-cyan-900/50 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600"
                >
                  {restoring === version.version ? "Opening…" : "Open"}
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
