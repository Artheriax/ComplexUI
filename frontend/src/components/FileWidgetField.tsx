import { useState } from "react";

import { datasetFileName } from "../lib/datasets";
import type { WidgetParam } from "../types/graph";
import { DatasetPickerDialog } from "./DatasetPickerDialog";
import { FileIcon } from "./icons";

interface FileWidgetFieldProps {
  widget: WidgetParam;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean;
}

export function FileWidgetField({
  widget,
  value,
  onChange,
  disabled = false,
}: FileWidgetFieldProps) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const path = value === undefined || value === null ? "" : String(value);
  const label = path ? datasetFileName(path) : "Choose file";

  return (
    <div className="flex items-center justify-between gap-2">
      <span
        className="truncate text-xs text-zinc-400"
        title={`${widget.name}: ${widget.data_type}`}
      >
        {widget.name}
        {disabled && (
          <span className="ml-1 text-[10px] text-cyan-400">linked</span>
        )}
      </span>
      <button
        type="button"
        title={path || "Choose a workspace file"}
        aria-label={`Choose file for ${widget.name}`}
        disabled={disabled}
        onClick={() => setPickerOpen(true)}
        className="nodrag flex w-28 items-center gap-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-left text-xs text-zinc-200 outline-none hover:border-cyan-600 focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500"
      >
        <FileIcon className="h-3 w-3 shrink-0 text-cyan-400" />
        <span className={`truncate ${path ? "" : "text-zinc-600"}`}>
          {label}
        </span>
      </button>
      {pickerOpen && (
        <DatasetPickerDialog
          currentPath={path}
          onClose={() => setPickerOpen(false)}
          onSelect={onChange}
        />
      )}
    </div>
  );
}
