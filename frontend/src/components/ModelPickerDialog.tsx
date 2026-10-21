import { useMemo, useState } from "react";
import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { useModelsAutoRefresh } from "../hooks/useModelsAutoRefresh";
import type { ModelSummary } from "../lib/api";
import {
  formatDatasetDate,
  formatDatasetSize,
} from "../lib/datasets";
import { useModelsStore } from "../store/useModelsStore";
import {
  CheckIcon,
  CloseIcon,
  FolderIcon,
  ResetIcon,
  SpinnerIcon,
  WarningIcon,
} from "./icons";

const EMPTY_MODELS: ModelSummary[] = [];

interface ModelPickerDialogProps {
  modelType: string;
  currentPath: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}

export function ModelPickerDialog({
  modelType,
  currentPath,
  onSelect,
  onClose,
}: ModelPickerDialogProps) {
  const models = useModelsStore((state) =>
    modelType ? (state.modelsByType[modelType] ?? EMPTY_MODELS) : EMPTY_MODELS,
  );
  const loaded = useModelsStore((state) =>
    modelType ? Boolean(state.loadedTypes[modelType]) : false,
  );
  const loading = useModelsStore((state) => state.loading);
  const refreshing = useModelsStore((state) => state.refreshing);
  const error = useModelsStore((state) => state.error);
  const refresh = useModelsStore((state) => state.refresh);
  const dialogRef = useDialogA11y<HTMLDivElement>(onClose);
  const [query, setQuery] = useState("");
  useModelsAutoRefresh(modelType, true);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return models;
    }
    return models.filter((model) =>
      `${model.name} ${model.path} ${model.format}`
        .toLowerCase()
        .includes(needle),
    );
  }, [models, query]);
  const selectedMissing =
    loaded &&
    currentPath !== "" &&
    !models.some((model) => model.path === currentPath);

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-picker-title"
        tabIndex={-1}
        className="flex max-h-[80vh] w-full max-w-2xl flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-start justify-between gap-4 border-b border-zinc-800 p-4">
          <div className="min-w-0">
            <h2
              id="model-picker-title"
              className="text-sm font-semibold text-zinc-100"
            >
              Choose local model
            </h2>
            <p className="mt-1 flex items-center gap-1 truncate font-mono text-[11px] text-zinc-500">
              <FolderIcon className="h-3 w-3 shrink-0 text-cyan-400" />
              models/{modelType}
            </p>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              title="Refresh models (R)"
              aria-label="Refresh models"
              onClick={() => void refresh(modelType)}
              disabled={refreshing}
              className="rounded-md border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800 disabled:cursor-wait"
            >
              <ResetIcon
                className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`}
              />
            </button>
            <button
              type="button"
              title="Close model picker"
              aria-label="Close model picker"
              onClick={onClose}
              className="rounded-md border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800"
            >
              <CloseIcon />
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          <div className="mb-2 flex items-start gap-2 rounded border border-amber-900/50 bg-amber-950/25 p-2 text-[10px] leading-relaxed text-amber-200/80">
            <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>
              Model files can execute code when loaded. Only use trusted files;
              prefer SafeTensors or ONNX. YOLO <code>.pt</code> loading is not a
              sandbox.
            </span>
          </div>

          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search models…"
            aria-label="Search models"
            className="mb-2 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-cyan-600"
          />

          {error && (
            <div className="mb-2 flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
              <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
          {selectedMissing && (
            <div className="mb-2 flex items-start gap-2 rounded border border-amber-900/60 bg-amber-950/30 p-2 text-[11px] text-amber-200">
              <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                The selected model is no longer present. Choose another local
                model or clear it to use the built-in fallback.
              </span>
            </div>
          )}

          <button
            type="button"
            onClick={() => {
              onSelect("");
              onClose();
            }}
            className={`mb-2 flex w-full items-center gap-2 rounded-lg border px-2.5 py-2 text-left ${
              currentPath === ""
                ? "border-cyan-700 bg-cyan-950/30"
                : "border-zinc-800 bg-zinc-950/40 hover:border-cyan-800 hover:bg-zinc-800/60"
            }`}
          >
            <span className="flex h-6 w-6 items-center justify-center rounded bg-zinc-800 text-[10px] text-zinc-400">
              B
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-xs font-medium text-zinc-200">
                Built-in fallback
              </span>
              <span className="block text-[10px] text-zinc-500">
                Use the predefined architecture and weights widgets
              </span>
            </span>
            {currentPath === "" && <CheckIcon className="h-3.5 w-3.5 text-cyan-300" />}
          </button>

          {!loaded && (loading || refreshing) && (
            <div className="flex items-center gap-2 p-2 text-xs text-zinc-500">
              <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Loading
              models…
            </div>
          )}
          {loaded && visible.length === 0 && (
            <p className="p-2 text-[11px] leading-relaxed text-zinc-600">
              No models in <span className="font-mono">models/{modelType}</span>.
              Add a supported file and press <kbd>R</kbd> to refresh.
            </p>
          )}
          <div className="space-y-1.5">
            {visible.map((model) => (
              <button
                key={model.path}
                type="button"
                onClick={() => {
                  onSelect(model.path);
                  onClose();
                }}
                className={`flex w-full items-center justify-between gap-3 rounded-lg border px-2.5 py-2 text-left ${
                  currentPath === model.path
                    ? "border-cyan-700 bg-cyan-950/30"
                    : "border-zinc-800 bg-zinc-950/40 hover:border-cyan-800 hover:bg-zinc-800/60"
                }`}
              >
                <span className="min-w-0">
                  <span className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
                    <span className="truncate">{model.name}</span>
                    {model.executable && (
                      <span
                        title="PyTorch pickle file: may execute code when loaded"
                        className="shrink-0 rounded bg-amber-950/60 px-1 py-px text-[8px] uppercase text-amber-300"
                      >
                        pickle
                      </span>
                    )}
                    {currentPath === model.path && (
                      <CheckIcon className="h-3 w-3 shrink-0 text-cyan-300" />
                    )}
                  </span>
                  <span className="mt-0.5 block truncate text-[10px] text-zinc-600">
                    {model.path}
                  </span>
                </span>
                <span className="shrink-0 text-right text-[10px] text-zinc-500">
                  <span className="block">{model.format}</span>
                  <span className="block">
                    {formatDatasetSize(model.size_bytes)}
                  </span>
                  <span className="block text-zinc-700">
                    {formatDatasetDate(model.modified_at)}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center justify-between border-t border-zinc-800 px-3 py-2 text-[10px] text-zinc-600">
          <span>Auto-refreshes while open · Press R to refresh</span>
          <span className="font-mono">models/{modelType}</span>
        </div>
      </div>
    </div>,
    document.body,
  );
}
