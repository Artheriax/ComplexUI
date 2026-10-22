import { useEffect, useState } from "react";

import type { ModelSummary } from "../lib/api";
import { datasetFileName } from "../lib/datasets";
import { useModelsStore } from "../store/useModelsStore";
import type { WidgetParam } from "../types/graph";
import { FolderIcon, WarningIcon } from "./icons";
import { ModelPickerDialog } from "./ModelPickerDialog";

const EMPTY_MODELS: ModelSummary[] = [];

interface ModelSelectorFieldProps {
  widget: WidgetParam;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean;
}

export function ModelSelectorField({
  widget,
  value,
  onChange,
  disabled = false,
}: ModelSelectorFieldProps) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const modelType = widget.model_type ?? "";
  const path = value === undefined || value === null ? "" : String(value);
  const models = useModelsStore((state) =>
    modelType ? (state.modelsByType[modelType] ?? EMPTY_MODELS) : EMPTY_MODELS,
  );
  const loaded = useModelsStore((state) =>
    modelType ? Boolean(state.loadedTypes[modelType]) : false,
  );
  const ensureType = useModelsStore((state) => state.ensureType);
  const missing =
    loaded && path !== "" && !models.some((model) => model.path === path);
  const label = path ? datasetFileName(path) : "Built-in fallback";

  useEffect(() => {
    if (modelType) {
      void ensureType(modelType);
    }
  }, [ensureType, modelType]);

  return (
    <div className="flex items-center justify-between gap-2">
      <span
        className="min-w-0 truncate text-xs text-zinc-400"
        title={`${widget.name}: model_type=${modelType || "unset"}`}
      >
        {widget.name}
        {disabled && (
          <span className="ml-1 text-[10px] text-cyan-400">linked</span>
        )}
      </span>
      <div className="flex min-w-0 max-w-full items-center gap-1">
        {missing && (
          <span
            title={`Missing local model: ${path}`}
            className="text-amber-400"
          >
            <WarningIcon className="h-3 w-3" />
          </span>
        )}
        <button
          type="button"
          title={path || `Use built-in fallback (models/${modelType})`}
          aria-label={`Choose model for ${widget.name}`}
          aria-invalid={missing || undefined}
          disabled={disabled}
          onClick={() => setPickerOpen(true)}
          className={`nodrag flex w-fit min-w-0 max-w-full items-center gap-1 rounded border bg-zinc-950 px-1.5 py-0.5 text-left text-xs outline-none disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500 ${
            missing
              ? "border-amber-700 text-amber-200"
              : "border-zinc-700 text-zinc-200 hover:border-cyan-600 focus:border-cyan-500"
          }`}
        >
          <FolderIcon className="h-3 w-3 shrink-0 text-cyan-400" />
          <span className={`truncate ${path ? "" : "text-zinc-500"}`}>
            {label}
          </span>
        </button>
      </div>
      {pickerOpen && (
        <ModelPickerDialog
          modelType={modelType}
          currentPath={path}
          onSelect={onChange}
          onClose={() => setPickerOpen(false)}
        />
      )}
    </div>
  );
}
