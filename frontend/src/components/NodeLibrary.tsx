import { useMemo, useState } from "react";
import type { DragEvent } from "react";

import { groupTemplates } from "../lib/templateSearch";
import { useGraphStore } from "../store/useGraphStore";
import { useSettingsStore } from "../store/useSettingsStore";
import { SOCKET_COLORS } from "../types/graph";
import type { TemplateInfo } from "../types/graph";
import { DRAG_MIME } from "./Canvas";
import {
  ChevronDownIcon,
  PlusIcon,
  SettingsIcon,
  SpinnerIcon,
  WarningIcon,
} from "./icons";

function TemplateCard({ template }: { template: TemplateInfo }) {
  const addNodeFromTemplate = useGraphStore((state) => state.addNodeFromTemplate);

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData(DRAG_MIME, template.name);
    event.dataTransfer.effectAllowed = "move";
  };

  const onClick = () => {
    const nodeCount = useGraphStore.getState().nodes.length;
    addNodeFromTemplate(template, {
      x: 80 + (nodeCount % 5) * 70,
      y: 80 + (nodeCount % 5) * 60,
    });
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      onClick={onClick}
      className="cursor-grab rounded-lg border border-zinc-800 bg-zinc-900 p-2.5 transition-colors hover:border-cyan-700 hover:bg-zinc-800/70 active:cursor-grabbing"
    >
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-semibold text-zinc-100">{template.title}</span>
      </div>
      <p className="mt-0.5 text-[11px] leading-snug text-zinc-500">
        {template.description}
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1">
        {template.node_schema.inputs.map((socket) => (
          <span
            key={`in-${socket.name}`}
            className="rounded px-1 py-px text-[9px]"
            style={{ backgroundColor: `${SOCKET_COLORS[socket.data_type]}22`, color: SOCKET_COLORS[socket.data_type] }}
          >
            {socket.name} in
          </span>
        ))}
        {template.node_schema.outputs.map((socket) => (
          <span
            key={`out-${socket.name}`}
            className="rounded px-1 py-px text-[9px]"
            style={{ backgroundColor: `${SOCKET_COLORS[socket.data_type]}22`, color: SOCKET_COLORS[socket.data_type] }}
          >
            {socket.name} out
          </span>
        ))}
      </div>
    </div>
  );
}

export function NodeLibrary() {
  const templates = useGraphStore((state) => state.templates);
  const templatesLoaded = useGraphStore((state) => state.templatesLoaded);
  const templatesError = useGraphStore((state) => state.templatesError);
  const openSettings = useSettingsStore((state) => state.openSettings);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const groups = useMemo(() => groupTemplates(templates, query), [templates, query]);
  const searching = query.trim().length > 0;

  const toggleCategory = (category: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-zinc-800 p-2.5">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search templates…"
          className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-cyan-600"
        />
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-2.5">
        <button
          type="button"
          onClick={() => {
            const count = useGraphStore.getState().nodes.length;
            useGraphStore.getState().addScratchNode({
              x: 80 + (count % 5) * 70,
              y: 80 + (count % 5) * 60,
            });
          }}
          className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-zinc-600 px-2.5 py-2 text-xs text-zinc-300 transition-colors hover:border-cyan-600 hover:bg-zinc-800/60 hover:text-cyan-200"
        >
          <PlusIcon /> New scratch node
        </button>
        {!templatesLoaded && !templatesError && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Loading templates…
          </div>
        )}
        {templatesError && (
          <div className="flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
            <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{templatesError}</span>
          </div>
        )}
        {searching && groups.length === 0 && templatesLoaded && (
          <p className="px-1 py-2 text-[11px] text-zinc-600">
            No templates match “{query.trim()}”.
          </p>
        )}
        {groups.map(({ category, items }) => {
          const isExpanded = searching || expanded.has(category);
          return (
            <section key={category}>
              <button
                type="button"
                onClick={() => toggleCategory(category)}
                className="mb-1 flex w-full items-center justify-between rounded px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500 transition-colors hover:bg-zinc-800/50 hover:text-zinc-300"
                title={isExpanded ? `Collapse ${category}` : `Expand ${category}`}
              >
                <span>{category}</span>
                <span className="flex items-center gap-1">
                  <span className="rounded bg-zinc-800/80 px-1 py-px text-[9px] font-medium text-zinc-500">
                    {items.length}
                  </span>
                  <ChevronDownIcon
                    className={`h-3 w-3 transition-transform ${isExpanded ? "" : "-rotate-90"}`}
                  />
                </span>
              </button>
              {isExpanded && (
                <div className="space-y-2">
                  {items.map((template) => (
                    <TemplateCard key={template.name} template={template} />
                  ))}
                </div>
              )}
            </section>
          );
        })}
      </div>
      <div className="border-t border-zinc-800 p-2.5">
        <button
          type="button"
          onClick={openSettings}
          className="flex w-full items-center gap-2 rounded-lg border border-zinc-700 px-2.5 py-2 text-xs text-zinc-300 transition-colors hover:border-cyan-700 hover:bg-zinc-800/60 hover:text-cyan-200"
        >
          <SettingsIcon className="h-3.5 w-3.5" />
          <span>Settings</span>
          <span className="ml-auto text-[10px] text-zinc-600">
            Canvas &amp; lines
          </span>
        </button>
      </div>
    </div>
  );
}
