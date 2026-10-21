import { useMemo, useState } from "react";

import { useModelsAutoRefresh } from "../hooks/useModelsAutoRefresh";
import {
  formatDatasetDate,
  formatDatasetSize,
} from "../lib/datasets";
import { useModelsStore } from "../store/useModelsStore";
import {
  ResetIcon,
  SpinnerIcon,
  WarningIcon,
} from "./icons";

export function ModelsLibrary() {
  const types = useModelsStore((state) => state.types);
  const modelsByType = useModelsStore((state) => state.modelsByType);
  const loading = useModelsStore((state) => state.loading);
  const refreshing = useModelsStore((state) => state.refreshing);
  const error = useModelsStore((state) => state.error);
  const lastUpdated = useModelsStore((state) => state.lastUpdated);
  const refresh = useModelsStore((state) => state.refresh);
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  useModelsAutoRefresh(null, true);

  const models = useMemo(
    () =>
      Object.values(modelsByType)
        .flat()
        .sort(
          (left, right) =>
            left.model_type.localeCompare(right.model_type) ||
            left.name.localeCompare(right.name),
        ),
    [modelsByType],
  );
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return models.filter((model) => {
      if (typeFilter !== "all" && model.model_type !== typeFilter) {
        return false;
      }
      return (
        !needle ||
        `${model.name} ${model.path} ${model.format} ${model.model_type}`
          .toLowerCase()
          .includes(needle)
      );
    });
  }, [models, query, typeFilter]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="space-y-1.5 border-b border-zinc-800 p-2.5">
        <div className="flex items-center gap-1.5">
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search models…"
            aria-label="Search models"
            className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-cyan-600"
          />
          <button
            type="button"
            title="Refresh models (R)"
            aria-label="Refresh models"
            onClick={() => void refresh(null)}
            disabled={refreshing}
            className="rounded border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800 disabled:cursor-wait"
          >
            <ResetIcon
              className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`}
            />
          </button>
        </div>
        <select
          value={typeFilter}
          onChange={(event) => setTypeFilter(event.target.value)}
          aria-label="Filter models by type"
          className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-cyan-600"
        >
          <option value="all">All model types</option>
          {types.map((type) => (
            <option key={type.name} value={type.name}>
              {type.name} ({type.count})
            </option>
          ))}
        </select>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-2.5">
        <div className="flex items-start gap-2 rounded border border-amber-900/50 bg-amber-950/25 p-2 text-[10px] leading-relaxed text-amber-200/80">
          <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            Only add trusted models. Loading <code>.pt</code> files can execute
            code; prefer SafeTensors or ONNX.
          </span>
        </div>
        {error && (
          <div className="flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
            <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Refreshing
            models…
          </div>
        )}
        {!loading && !error && visible.length === 0 && (
          <p className="px-1 py-2 text-[11px] leading-relaxed text-zinc-600">
            No models match. Add files under{" "}
            <span className="font-mono">
              models/{typeFilter === "all" ? "<type>" : typeFilter}
            </span>{" "}
            and press <kbd>R</kbd> to refresh.
          </p>
        )}
        {visible.map((model) => (
          <div
            key={model.path}
            className="rounded-lg border border-zinc-800 bg-zinc-900 p-2.5"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="truncate text-xs font-semibold text-zinc-100">
                  {model.name}
                </p>
                <p
                  className="mt-0.5 truncate font-mono text-[10px] text-zinc-500"
                  title={model.path}
                >
                  {model.path}
                </p>
              </div>
              <div className="flex shrink-0 flex-col items-end gap-0.5">
                <span className="rounded bg-zinc-800/80 px-1 py-px text-[9px] text-zinc-400">
                  {model.model_type}
                </span>
                {model.executable && (
                  <span
                    title="PyTorch pickle file: may execute code when loaded"
                    className="rounded bg-amber-950/60 px-1 py-px text-[8px] uppercase text-amber-300"
                  >
                    pickle
                  </span>
                )}
              </div>
            </div>
            <p className="mt-1 text-[10px] text-zinc-600">
              {model.format} · {formatDatasetSize(model.size_bytes)} ·{" "}
              {formatDatasetDate(model.modified_at)}
            </p>
            <p
              className="mt-0.5 truncate font-mono text-[9px] text-zinc-700"
              title={model.fingerprint}
            >
              fingerprint {model.fingerprint}
            </p>
          </div>
        ))}
      </div>
      <div className="border-t border-zinc-800 p-2.5 text-[10px] leading-relaxed text-zinc-600">
        Read-only view of the repo <span className="font-mono">models/</span>{" "}
        folder. Auto-refreshes every 2.5 s · press <kbd>R</kbd> to reload.
        {lastUpdated !== null && (
          <span className="mt-0.5 block text-zinc-700">
            Updated {new Date(lastUpdated).toLocaleTimeString()}
          </span>
        )}
      </div>
    </div>
  );
}
