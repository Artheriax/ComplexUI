import { useMemo, useState } from "react";

import { useGraphStore } from "../store/useGraphStore";
import type { WorkflowTemplateInfo } from "../types/graph";
import { CategorySection } from "./CategorySection";
import { SpinnerIcon, WarningIcon } from "./icons";

const CATEGORY_ORDER = [
  "Getting Started",
  "Classical ML",
  "Data Quality & EDA",
  "PyTorch",
  "Computer Vision",
];

interface WorkflowGroup {
  category: string;
  items: WorkflowTemplateInfo[];
}

function WorkflowCard({ workflow }: { workflow: WorkflowTemplateInfo }) {
  const openWorkflowTemplateInWindow = useGraphStore(
    (state) => state.openWorkflowTemplateInWindow,
  );

  return (
    <button
      type="button"
      onClick={() => {
        openWorkflowTemplateInWindow(workflow);
      }}
      className="w-full rounded-lg border border-zinc-800 bg-zinc-900 p-2.5 text-left transition-colors hover:border-cyan-700 hover:bg-zinc-800/70"
    >
      <div className="flex items-center justify-between gap-1">
        <span className="text-xs font-semibold text-zinc-100">
          {workflow.title}
        </span>
        <span className="shrink-0 rounded bg-zinc-800/80 px-1 py-px text-[9px] text-zinc-500">
          {workflow.nodes.length} nodes
        </span>
      </div>
      <p className="mt-0.5 text-[11px] leading-snug text-zinc-500">
        {workflow.description}
      </p>
    </button>
  );
}

function groupWorkflows(
  workflows: WorkflowTemplateInfo[],
  query: string,
): WorkflowGroup[] {
  const needle = query.trim().toLowerCase();
  const visible = needle
    ? workflows.filter((workflow) =>
        `${workflow.title} ${workflow.name} ${workflow.description} ${workflow.category}`
          .toLowerCase()
          .includes(needle),
      )
    : workflows;
  const buckets = new Map<string, WorkflowTemplateInfo[]>();
  for (const workflow of visible) {
    const items = buckets.get(workflow.category);
    if (items) {
      items.push(workflow);
    } else {
      buckets.set(workflow.category, [workflow]);
    }
  }
  const known = CATEGORY_ORDER.filter((category) => buckets.has(category));
  const unknown = [...buckets.keys()]
    .filter((category) => !CATEGORY_ORDER.includes(category))
    .sort();
  return [...known, ...unknown].map((category) => ({
    category,
    items: buckets.get(category) ?? [],
  }));
}

export function WorkflowLibrary() {
  const workflows = useGraphStore((state) => state.workflows);
  const workflowsLoaded = useGraphStore((state) => state.workflowsLoaded);
  const workflowsError = useGraphStore((state) => state.workflowsError);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const groups = useMemo(
    () => groupWorkflows(workflows, query),
    [workflows, query],
  );
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
        {!workflowsLoaded && !workflowsError && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Loading
            workflows…
          </div>
        )}
        {workflowsError && (
          <div className="flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
            <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{workflowsError}</span>
          </div>
        )}
        {searching && groups.length === 0 && workflowsLoaded && (
          <p className="px-1 py-2 text-[11px] text-zinc-600">
            No workflows match “{query.trim()}”.
          </p>
        )}
        {groups.map(({ category, items }) => {
          const isExpanded = searching || expanded.has(category);
          return (
            <CategorySection
              key={category}
              category={category}
              count={items.length}
              expanded={isExpanded}
              onToggle={() => toggleCategory(category)}
            >
              <div className="space-y-2">
                {items.map((workflow) => (
                  <WorkflowCard key={workflow.name} workflow={workflow} />
                ))}
              </div>
            </CategorySection>
          );
        })}
      </div>
      <div className="border-t border-zinc-800 p-2.5 text-[10px] leading-relaxed text-zinc-600">
        Opening a template adds it as a new workflow window.
      </div>
    </div>
  );
}
