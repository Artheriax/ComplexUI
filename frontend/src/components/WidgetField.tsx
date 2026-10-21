import type { WidgetChoice, WidgetParam } from "../types/graph";
import { FileWidgetField } from "./FileWidgetField";

interface WidgetFieldProps {
  widget: WidgetParam;
  value: unknown;
  onChange: (value: unknown) => void;
  step?: number;
  connected?: boolean;
}

const inputClass =
  "nodrag w-28 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-right text-xs text-zinc-100 outline-none focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500";
const stepperInputClass =
  "nodrag w-16 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-right text-xs text-zinc-100 outline-none focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500";
const stepButtonClass =
  "nodrag flex h-5 w-5 items-center justify-center rounded border border-zinc-700 text-[11px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-700";
const selectClass =
  "nodrag w-28 rounded border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-xs text-zinc-100 outline-none focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500";

function choiceLabel(option: WidgetChoice): string {
  return typeof option === "boolean" ? String(option).toLowerCase() : String(option);
}

function steppedValue(
  value: unknown,
  step: number,
  direction: -1 | 1,
  integer: boolean,
): number {
  const current =
    typeof value === "number" && Number.isFinite(value) ? value : 0;
  const next = current + step * direction;
  return integer
    ? Math.round(next)
    : Number(next.toPrecision(12));
}

export function WidgetField({
  widget,
  value,
  onChange,
  step,
  connected = false,
}: WidgetFieldProps) {
  const kind = widget.data_type.toLowerCase();
  const options = widget.options ?? [];
  const numericStep =
    step !== undefined && Number.isFinite(step) && step > 0
      ? step
      : undefined;

  if (kind === "file") {
    return (
      <FileWidgetField
        widget={widget}
        value={value}
        onChange={onChange}
        disabled={connected}
      />
    );
  }

  const control = (() => {
    if (options.length > 0) {
      const selectedIndex = options.findIndex(
        (option) => option === value,
      );
      return (
        <select
          className={selectClass}
          value={selectedIndex >= 0 ? String(selectedIndex) : ""}
          disabled={connected}
          onChange={(event) => {
            const index = Number.parseInt(event.target.value, 10);
            if (
              Number.isInteger(index) &&
              index >= 0 &&
              index < options.length
            ) {
              onChange(options[index]);
            }
          }}
        >
          {selectedIndex < 0 && <option value="">Select...</option>}
          {options.map((option, index) => (
            <option key={`${String(option)}-${index}`} value={index}>
              {choiceLabel(option)}
            </option>
          ))}
        </select>
      );
    }
    if (kind.includes("bool")) {
      return (
        <input
          type="checkbox"
          className="nodrag h-3.5 w-3.5 accent-cyan-500 disabled:cursor-not-allowed"
          checked={Boolean(value)}
          disabled={connected}
          onChange={(event) => onChange(event.target.checked)}
        />
      );
    }
    if (kind.includes("int")) {
      const jump = Math.max(1, Math.round(numericStep ?? 1));
      if (numericStep !== undefined) {
        return (
          <div className="nodrag flex items-center gap-1">
            <button
              type="button"
              className={stepButtonClass}
              disabled={connected}
              aria-label={`Decrease ${widget.name} by ${jump}`}
              onClick={() =>
                onChange(steppedValue(value, jump, -1, true))
              }
            >
              &lt;
            </button>
            <input
              type="number"
              step={jump}
              className={stepperInputClass}
              disabled={connected}
              value={
                value === undefined || value === null ? "" : String(value)
              }
              onChange={(event) => {
                const parsed = Number.parseInt(event.target.value, 10);
                onChange(Number.isNaN(parsed) ? 0 : parsed);
              }}
            />
            <button
              type="button"
              className={stepButtonClass}
              disabled={connected}
              aria-label={`Increase ${widget.name} by ${jump}`}
              onClick={() => onChange(steppedValue(value, jump, 1, true))}
            >
              &gt;
            </button>
          </div>
        );
      }
      return (
        <input
          type="number"
          step={1}
          className={inputClass}
          disabled={connected}
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => {
            const parsed = Number.parseInt(event.target.value, 10);
            onChange(Number.isNaN(parsed) ? 0 : parsed);
          }}
        />
      );
    }
    if (kind.includes("float")) {
      if (numericStep !== undefined) {
        return (
          <div className="nodrag flex items-center gap-1">
            <button
              type="button"
              className={stepButtonClass}
              disabled={connected}
              aria-label={`Decrease ${widget.name} by ${numericStep}`}
              onClick={() =>
                onChange(steppedValue(value, numericStep, -1, false))
              }
            >
              &lt;
            </button>
            <input
              type="number"
              step={numericStep}
              className={stepperInputClass}
              disabled={connected}
              value={
                value === undefined || value === null ? "" : String(value)
              }
              onChange={(event) => {
                const parsed = Number.parseFloat(event.target.value);
                onChange(Number.isNaN(parsed) ? 0 : parsed);
              }}
            />
            <button
              type="button"
              className={stepButtonClass}
              disabled={connected}
              aria-label={`Increase ${widget.name} by ${numericStep}`}
              onClick={() =>
                onChange(steppedValue(value, numericStep, 1, false))
              }
            >
              &gt;
            </button>
          </div>
        );
      }
      return (
        <input
          type="number"
          step="any"
          className={inputClass}
          disabled={connected}
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
          disabled={connected}
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value)}
        />
      );
    }
    return (
      <input
        type="text"
        className={`${inputClass} text-left`}
        disabled={connected}
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  })();

  return (
    <label className="flex items-center justify-between gap-2">
      <span
        className="truncate text-xs text-zinc-400"
        title={`${widget.name}: ${widget.data_type}`}
      >
        {widget.name}
        {connected && (
          <span className="ml-1 text-[10px] text-cyan-400">linked</span>
        )}
      </span>
      {control}
    </label>
  );
}
