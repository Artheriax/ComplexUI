import type { WidgetParam } from "../types/graph";

interface WidgetFieldProps {
  widget: WidgetParam;
  value: unknown;
  onChange: (value: unknown) => void;
}

const inputClass =
  "nodrag w-28 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-right text-xs text-zinc-100 outline-none focus:border-cyan-500";

export function WidgetField({ widget, value, onChange }: WidgetFieldProps) {
  const kind = widget.data_type.toLowerCase();

  const control = (() => {
    if (kind.includes("bool")) {
      return (
        <input
          type="checkbox"
          className="nodrag h-3.5 w-3.5 accent-cyan-500"
          checked={Boolean(value)}
          onChange={(event) => onChange(event.target.checked)}
        />
      );
    }
    if (kind.includes("int")) {
      return (
        <input
          type="number"
          step={1}
          className={inputClass}
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => {
            const parsed = Number.parseInt(event.target.value, 10);
            onChange(Number.isNaN(parsed) ? 0 : parsed);
          }}
        />
      );
    }
    if (kind.includes("float")) {
      return (
        <input
          type="number"
          step="any"
          className={inputClass}
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => {
            const parsed = Number.parseFloat(event.target.value);
            onChange(Number.isNaN(parsed) ? 0 : parsed);
          }}
        />
      );
    }
    if (kind === "str" || kind === "string") {
      return (
        <input
          type="text"
          className={`${inputClass} text-left`}
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value)}
        />
      );
    }
    return (
      <input
        type="text"
        className={`${inputClass} text-left`}
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  })();

  return (
    <label className="flex items-center justify-between gap-2">
      <span className="truncate text-xs text-zinc-400" title={`${widget.name}: ${widget.data_type}`}>
        {widget.name}
      </span>
      {control}
    </label>
  );
}
