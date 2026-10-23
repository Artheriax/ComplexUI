import type { ReactNode } from "react";

import type { WidgetParam } from "../types/graph";

const INPUT_NAMES = ["input_1", "input_2", "input_3", "input_4", "input_5"];

interface SwitchFieldsProps {
  widgets: WidgetParam[];
  values: Record<string, unknown>;
  connected: Set<string>;
  onChange: (name: string, value: unknown) => void;
  renderHandle: (
    name: string,
    widget: WidgetParam,
    connected: boolean,
  ) => ReactNode;
}

function labelList(value: unknown): string[] {
  const labels = String(value ?? "")
    .split("|")
    .map((label) => label.trim());
  while (labels.length < INPUT_NAMES.length) {
    labels.push(`Input ${labels.length + 1}`);
  }
  return labels.slice(0, INPUT_NAMES.length);
}

export function SwitchFields({
  widgets,
  values,
  connected,
  onChange,
  renderHandle,
}: SwitchFieldsProps) {
  const byName = new Map(widgets.map((widget) => [widget.name, widget]));
  const inputWidgets = INPUT_NAMES.map((name) => byName.get(name));
  const choicesWidget = byName.get("choices");
  const selectedWidget = byName.get("selected");
  if (
    choicesWidget === undefined ||
    selectedWidget === undefined ||
    inputWidgets.some((widget) => widget === undefined)
  ) {
    return null;
  }

  const labels = labelList(values.choices);
  const rawSelected = Number(values.selected);
  const selected = Number.isInteger(rawSelected)
    ? Math.min(INPUT_NAMES.length - 1, Math.max(0, rawSelected))
    : 0;
  const labelsLocked = connected.has("choices");
  const selectedLocked = connected.has("selected");
  const selectedConnected = connected.has(INPUT_NAMES[selected]);

  const updateLabel = (index: number, label: string) => {
    const next = [...labels];
    next[index] = label.replace(/\|/g, "/");
    onChange("choices", next.join("|"));
  };

  return (
    <div className="space-y-1 px-2 py-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-zinc-400">
          Selected
          {selectedLocked && (
            <span className="ml-1 text-[10px] text-cyan-400">linked</span>
          )}
        </span>
        <select
          aria-label="Switch selection"
          className="nodrag min-w-0 flex-1 rounded border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-[10px] text-zinc-200 outline-none focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500"
          value={String(selected)}
          disabled={selectedLocked}
          onChange={(event) =>
            onChange("selected", Number.parseInt(event.target.value, 10))
          }
        >
          {labels.map((label, index) => (
            <option key={`${index}-${label}`} value={index}>
              {label || `Input ${index + 1}`}
            </option>
          ))}
        </select>
      </div>

      {INPUT_NAMES.map((name, index) => {
        const widget = inputWidgets[index];
        if (widget === undefined) {
          return null;
        }
        const isConnected = connected.has(name);
        return (
          <div key={name} className="cu-widget-row relative pl-3">
            {renderHandle(name, widget, isConnected)}
            <div
              className={`flex min-w-0 items-center gap-1 rounded border px-1 py-0.5 ${
                selected === index
                  ? "border-cyan-900/70 bg-cyan-950/30"
                  : "border-transparent"
              }`}
            >
              <span
                className={`h-2 w-2 shrink-0 rounded-full ${
                  isConnected ? "bg-emerald-400" : "bg-zinc-700"
                }`}
                aria-hidden="true"
              />
              <input
                type="text"
                aria-label={`Input ${index + 1} name`}
                className="nodrag min-w-0 flex-1 rounded border border-zinc-800 bg-zinc-950 px-1 py-0.5 text-[10px] text-zinc-200 outline-none placeholder:text-zinc-700 focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-900 disabled:bg-zinc-900 disabled:text-zinc-500"
                value={labels[index]}
                disabled={labelsLocked}
                placeholder={`Input ${index + 1}`}
                onChange={(event) => updateLabel(index, event.target.value)}
              />
              <span className="shrink-0 text-[9px] text-zinc-600">
                {isConnected ? "linked" : "unused"}
              </span>
            </div>
          </div>
        );
      })}

      <p className="text-[9px] leading-relaxed text-zinc-600">
        Only the selected input is passed through. Unused inputs are ignored.
      </p>
      {!selectedConnected && (
        <p role="status" className="text-[9px] text-amber-400">
          The selected input is not connected.
        </p>
      )}
    </div>
  );
}
