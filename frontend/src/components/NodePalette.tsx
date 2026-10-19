import { useState } from "react";

import { NodeLibrary } from "./NodeLibrary";
import { WorkflowLibrary } from "./WorkflowLibrary";

type PaletteTab = "nodes" | "workflows";

const TABS: { value: PaletteTab; label: string }[] = [
  { value: "nodes", label: "Nodes" },
  { value: "workflows", label: "Workflows" },
];

export function NodePalette() {
  const [tab, setTab] = useState<PaletteTab>("nodes");

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/40">
      <div className="flex border-b border-zinc-800" role="tablist">
        {TABS.map(({ value, label }) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={tab === value}
            onClick={() => setTab(value)}
            className={`flex-1 px-3 py-2 text-[11px] font-semibold uppercase tracking-wider transition-colors ${
              tab === value
                ? "border-b-2 border-cyan-500 text-cyan-300"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "nodes" ? <NodeLibrary /> : <WorkflowLibrary />}
    </aside>
  );
}
