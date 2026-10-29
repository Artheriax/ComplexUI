import { useEffect, useMemo, useState } from "react";

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
  const select = widget.model_select ?? "file";
  const required = widget.model_required === true;
  const path = value === undefined || value === null ? "" : String(value);
  const typesKey = [modelType, ...(widget.model_include ?? [])]
    .filter(Boolean)
    .join("|");
  const modelTypes = useMemo(
    () => (typesKey ? typesKey.split("|") : []),
    [typesKey],
  );
  const modelsByType = useModelsStore((state) => state.modelsByType);
  const loadedTypes = useModelsStore((state) => state.loadedTypes);
  const ensureType = useModelsStore((state) => state.ensureType);
  const models = useMemo(
    () =>
      modelTypes.flatMap(
        (currentType) => modelsByType[currentType] ?? EMPTY_MODELS,
      ),
    [modelTypes, modelsByType],
  );
  const loaded =
    modelTypes.length > 0 &&
    modelTypes.every((currentType) => Boolean(loadedTypes[currentType]));
  const eligible = models.filter((model) =>
    select === "any" ? true : model.is_directory === (select === "folder"),
  );
  const missing =
    loaded &&
    ((path !== "" && !eligible.some((model) => model.path === path)) ||
      (required && path === ""));
  const label = path
    ? datasetFileName(path)
    : required
      ? "Select model…"
      : "Built-in fallback";

  useEffect(() => {
    for (const currentType of modelTypes) {
      void ensureType(currentType);
    }
  }, [ensureType, modelTypes]);

  return (
    <div className="flex items-center justify-between gap-2">
      <span
        className="min-w-0 truncate text-xs text-zinc-400"
        title={`${widget.name}: model_type=${modelTypes.join(", ") || "unset"}, select=${select}${
          required ? ", required" : ""
        }`}
      >
        {widget.name}
        {disabled && (
          <span className="ml-1 text-[10px] text-cyan-400">linked</span>
        )}
      </span>
      <div className="flex min-w-0 max-w-full items-center gap-1">
        {missing && (
          <span
            title={
              required && path === ""
                ? "A local model is required"
                : `Missing or incompatible local model: ${path}`
            }
            className="text-amber-400"
          >
            <WarningIcon className="h-3 w-3" />
          </span>
        )}
        <button
          type="button"
          title={
            path ||
            (required
              ? `Select a required local ${select === "folder" ? "folder" : "model"}`
              : `Use built-in fallback (${modelTypes.map((type) => `models/${type}`).join(", ")})`)
          }
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
          modelTypes={modelTypes}
          currentPath={path}
          select={select}
          required={required}
          onSelect={onChange}
          onClose={() => setPickerOpen(false)}
        />
      )}
    </div>
  );
}
