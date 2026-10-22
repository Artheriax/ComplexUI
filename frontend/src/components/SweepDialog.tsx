import { useMemo, useState } from "react";
import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { useGraphStore } from "../store/useGraphStore";
import type { WidgetParam } from "../types/graph";
import { BoltIcon, CheckIcon } from "./icons";

interface SweepDialogProps {
  nodeId: string;
  onClose: () => void;
}

function isSweepable(widget: WidgetParam): boolean {
  const kind = widget.data_type.toLowerCase();
  return (
    (widget.options?.length ?? 0) === 0 &&
    (kind === "int" || kind === "float" || kind === "str" || kind === "string")
  );
}

function valuesForSeed(start: number, count: number, step: number): number[] {
  return Array.from({ length: count }, (_, index) => start + index * step);
}

export function SweepDialog({ nodeId, onClose }: SweepDialogProps) {
  const node = useGraphStore((state) =>
    state.nodes.find((candidate) => candidate.id === nodeId),
  );
  const runSweep = useGraphStore((state) => state.runSweep);
  const widgets = useMemo(
    () => (node?.data.schema.widgets ?? []).filter(isSweepable),
    [node?.data.schema.widgets],
  );
  const [widgetName, setWidgetName] = useState(() => {
    const preferred =
      widgets.find((widget) => widget.name === "seed") ??
      widgets.find((widget) => widget.name === "text") ??
      widgets[0];
    return preferred?.name ?? "";
  });
  const [start, setStart] = useState(0);
  const [count, setCount] = useState(4);
  const [step, setStep] = useState(1);
  const [promptLines, setPromptLines] = useState("");
  const [suffixOutputs, setSuffixOutputs] = useState(true);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);
  const dialogRef = useDialogA11y<HTMLFormElement>(onClose);

  if (!node) {
    return null;
  }
  if (widgets.length === 0) {
    return null;
  }

  const widget = widgets.find((candidate) => candidate.name === widgetName);
  const kind = widget?.data_type.toLowerCase() ?? "int";
  const isText = kind === "str" || kind === "string";
  const promptValues = promptLines
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const valueCount = isText ? promptValues.length : count;
  const canRun =
    widget !== undefined &&
    valueCount > 0 &&
    valueCount <= 32 &&
    !busy;

  const submit = async () => {
    if (!canRun || !widget) {
      return;
    }
    setBusy(true);
    setError(null);
    setFinished(false);
    setProgress(0);
    try {
      const values = isText
        ? promptValues
        : valuesForSeed(start, count, step);
      const result = await runSweep(
        {
          nodeId,
          widget: widget.name,
          values,
          suffixOutputs,
        },
        (completed, total) => {
          setProgress(total > 0 ? completed / total : 0);
        },
      );
      if (result.status === "completed") {
        setFinished(true);
        setProgress(1);
      } else {
        setError("Sweep stopped before every run completed.");
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  return createPortal(
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/70 p-6">
      <form
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="sweep-title"
        aria-describedby="sweep-description"
        tabIndex={-1}
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
        className="w-full max-w-md rounded-xl border border-zinc-700 bg-zinc-900 p-4 shadow-2xl outline-none"
      >
        <h2 id="sweep-title" className="text-sm font-semibold text-zinc-100">
          Sweep {node.data.label}
        </h2>
        <p
          id="sweep-description"
          className="mt-1 text-[11px] leading-snug text-zinc-500"
        >
          Run this workflow once per value. Outputs from Save Media nodes get a
          unique sweep suffix when enabled.
        </p>

        <label className="mt-4 block">
          <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            Parameter
          </span>
          <select
            value={widgetName}
            disabled={busy}
            onChange={(event) => {
              setWidgetName(event.target.value);
              setFinished(false);
              setError(null);
            }}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600"
          >
            {widgets.map((candidate) => (
              <option key={candidate.name} value={candidate.name}>
                {candidate.name} ({candidate.data_type})
              </option>
            ))}
          </select>
        </label>

        {isText ? (
          <label className="mt-3 block">
            <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
              Prompts, one per line ({promptValues.length})
            </span>
            <textarea
              value={promptLines}
              rows={5}
              disabled={busy}
              onChange={(event) => {
                setPromptLines(event.target.value);
                setFinished(false);
                setError(null);
              }}
              placeholder="A red fox in snow&#10;A blue fox in snow"
              className="w-full resize-none rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600 disabled:opacity-60"
            />
          </label>
        ) : (
          <div className="mt-3 grid grid-cols-3 gap-2">
            <label className="block">
              <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                Start
              </span>
              <input
                type="number"
                value={start}
                disabled={busy}
                onChange={(event) => {
                  setStart(Number.parseInt(event.target.value, 10) || 0);
                  setFinished(false);
                }}
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600 disabled:opacity-60"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                Count
              </span>
              <input
                type="number"
                min={1}
                max={32}
                value={count}
                disabled={busy}
                onChange={(event) => {
                  setCount(Math.min(32, Math.max(1, Number.parseInt(event.target.value, 10) || 1)));
                  setFinished(false);
                }}
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600 disabled:opacity-60"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                Step
              </span>
              <input
                type="number"
                value={step}
                disabled={busy}
                onChange={(event) => {
                  setStep(Number.parseInt(event.target.value, 10) || 1);
                  setFinished(false);
                }}
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600 disabled:opacity-60"
              />
            </label>
          </div>
        )}

        <label className="mt-3 flex items-start gap-2 text-[11px] text-zinc-400">
          <input
            type="checkbox"
            checked={suffixOutputs}
            disabled={busy}
            onChange={(event) => setSuffixOutputs(event.target.checked)}
            className="mt-0.5 accent-cyan-500"
          />
          <span>
            Append <code className="text-zinc-300">_sweep_&lt;index&gt;_&lt;value&gt;</code>{" "}
            to Save Media paths.
          </span>
        </label>

        {busy && (
          <div className="mt-3">
            <div className="flex justify-between text-[10px] text-zinc-500">
              <span>Sweeping…</span>
              <span>
                {Math.round(progress * 100)}% ({valueCount} values)
              </span>
            </div>
            <div className="mt-1 h-1 overflow-hidden rounded bg-zinc-800">
              <div
                className="h-full bg-cyan-500 transition-all"
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            </div>
          </div>
        )}
        {finished && (
          <p className="mt-2 flex items-center gap-1.5 text-[11px] text-emerald-300">
            <CheckIcon className="h-3 w-3" /> Sweep completed.
          </p>
        )}
        {error && (
          <p role="alert" className="mt-2 text-[11px] text-red-300">
            {error}
          </p>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-600"
          >
            Close
          </button>
          <button
            type="submit"
            disabled={!canRun}
            className="flex items-center gap-1.5 rounded-md border border-cyan-700 bg-cyan-950/50 px-2.5 py-1 text-xs text-cyan-200 hover:bg-cyan-900/50 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600"
          >
            <BoltIcon className="h-3 w-3" />
            {busy ? "Sweeping…" : "Run sweep"}
          </button>
        </div>
      </form>
    </div>,
    document.body,
  );
}
