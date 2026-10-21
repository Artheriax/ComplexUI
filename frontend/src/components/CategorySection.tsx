import type { ReactNode } from "react";

import { ChevronDownIcon } from "./icons";

export function CategorySection({
  category,
  count,
  expanded,
  onToggle,
  children,
}: {
  category: string;
  count: number;
  expanded: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <section>
      <button
        type="button"
        aria-expanded={expanded}
        onClick={onToggle}
        className="mb-1 flex w-full items-center justify-between rounded px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500 transition-colors hover:bg-zinc-800/50 hover:text-zinc-300"
        title={expanded ? `Collapse ${category}` : `Expand ${category}`}
      >
        <span>{category}</span>
        <span className="flex items-center gap-1">
          <span className="rounded bg-zinc-800/80 px-1 py-px text-[9px] font-medium text-zinc-500">
            {count}
          </span>
          <ChevronDownIcon
            className={`h-3 w-3 transition-transform ${expanded ? "" : "-rotate-90"}`}
          />
        </span>
      </button>
      {expanded && children}
    </section>
  );
}
