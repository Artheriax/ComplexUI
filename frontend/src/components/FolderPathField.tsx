import { useState } from "react";

import { pickFolder } from "../lib/api";
import { errorMessage } from "../lib/errors";
import type { WidgetParam } from "../types/graph";
import { FolderIcon, SpinnerIcon, WarningIcon } from "./icons";

interface FolderPathFieldProps {
  widget: WidgetParam;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean;
}

export function FolderPathField({
  widget,
  value,
  onChange,
  disabled = false,
}: FolderPathFieldProps) {
  const [browsing, setBrowsing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const path = value === undefined || value === null ? "" : String(value);

  const browse = async () => {
    setBrowsing(true);
    setError(null);
    try {
      const selected = await pickFolder(path || undefined);
      if (selected !== null) {
        onChange(selected);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBrowsing(false);
    }
  };

  return (
    <div className="flex min-w-0 flex-col gap-1">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <span
          className="truncate text-xs text-zinc-400"
          title={`${widget.name}: ${widget.data_type}`}
        >
          {widget.name}
          {disabled && (
            <span className="ml-1 text-[10px] text-cyan-400">linked</span>
          )}
        </span>
        <div className="flex min-w-0 flex-1 items-center justify-end gap-1">
          <input
            type="text"
            aria-label={`Choose folder for ${widget.name}`}
            className="nodrag min-w-0 flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono text-[10px] text-zinc-100 outline-none placeholder:text-zinc-700 focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500"
            value={path}
            disabled={disabled || browsing}
            placeholder="Choose a folder"
            onChange={(event) => onChange(event.target.value)}
          />
          <button
            type="button"
            aria-label={`Browse for ${widget.name}`}
            title="Browse for a folder"
            disabled={disabled || browsing}
            onClick={() => void browse()}
            className="nodrag flex shrink-0 items-center gap-1 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-600"
          >
            {browsing ? (
              <SpinnerIcon className="h-3 w-3 animate-spin" />
            ) : (
              <FolderIcon className="h-3 w-3" />
            )}
            Browse
          </button>
        </div>
      </div>
      {browsing && (
        <p role="status" className="text-[9px] text-cyan-400">
          Waiting for the folder dialog…
        </p>
      )}
      {error !== null && (
        <p role="alert" className="flex items-start gap-1 text-[9px] text-red-400">
          <WarningIcon className="mt-px h-3 w-3 shrink-0" />
          <span>{error}</span>
        </p>
      )}
    </div>
  );
}
