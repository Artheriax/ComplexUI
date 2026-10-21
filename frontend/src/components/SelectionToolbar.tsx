import { useMemo } from "react";

import { useGraphStore } from "../store/useGraphStore";
import type { ArrangeMode } from "../store/useGraphStore";
import { isFrameNode, isRunnableNode } from "../types/graph";
import {
  ArrangeIcon,
  CloseIcon,
  GroupIcon,
  SubgraphIcon,
  UngroupIcon,
} from "./icons";
import type { ArrangeIconKind } from "./icons";

const ALIGN_ACTIONS: {
  mode: ArrangeMode;
  kind: ArrangeIconKind;
  label: string;
}[] = [
  { mode: "align-left", kind: "align-left", label: "Align left" },
  { mode: "align-center-x", kind: "align-center-x", label: "Align horizontal centers" },
  { mode: "align-right", kind: "align-right", label: "Align right" },
  { mode: "align-top", kind: "align-top", label: "Align top" },
  { mode: "align-center-y", kind: "align-center-y", label: "Align vertical centers" },
  { mode: "align-bottom", kind: "align-bottom", label: "Align bottom" },
];

const buttonClass =
  "flex h-6 w-6 items-center justify-center rounded text-zinc-400 transition-colors hover:bg-zinc-700 hover:text-zinc-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 disabled:cursor-not-allowed disabled:text-zinc-700 disabled:hover:bg-transparent";

export function SelectionToolbar() {
  const nodes = useGraphStore((state) => state.nodes);
  const arrangeSelection = useGraphStore((state) => state.arrangeSelection);
  const groupSelection = useGraphStore((state) => state.groupSelection);
  const ungroupSelection = useGraphStore((state) => state.ungroupSelection);
  const createSubgraphFromSelection = useGraphStore(
    (state) => state.createSubgraphFromSelection,
  );
  const clearSelection = useGraphStore((state) => state.clearSelection);
  const selected = useMemo(
    () => nodes.filter((node) => node.selected),
    [nodes],
  );
  const selectedCount = selected.length;
  const canUngroup = selected.some(isFrameNode);
  const canArrange =
    selectedCount >= 2 &&
    selected.every((node) => !isFrameNode(node) && !node.parentId);
  const canGroup =
    selectedCount >= 2 &&
    selected.every((node) => isRunnableNode(node) && !node.parentId);
  const canSubgraph = canGroup;
  const canDistribute = canArrange && selectedCount >= 3;

  if (selectedCount < 2 && !canUngroup) {
    return null;
  }

  return (
    <div
      role="toolbar"
      aria-label="Selection tools"
      className="absolute left-1/2 top-3 z-20 flex -translate-x-1/2 items-center gap-1 rounded-lg border border-zinc-700 bg-zinc-900/95 px-2 py-1 shadow-xl backdrop-blur"
    >
      <span className="mr-1 whitespace-nowrap text-[10px] font-medium text-zinc-400">
        {selectedCount} selected
      </span>
      <div
        role="group"
        aria-label="Align selection"
        className="flex items-center gap-0.5 border-l border-zinc-700 pl-1"
      >
        {ALIGN_ACTIONS.map(({ mode, kind, label }) => (
          <button
            key={mode}
            type="button"
            title={label}
            aria-label={label}
            disabled={!canArrange}
            onClick={() => arrangeSelection(mode)}
            className={buttonClass}
          >
            <ArrangeIcon kind={kind} />
          </button>
        ))}
      </div>
      <div
        role="group"
        aria-label="Distribute selection"
        className="flex items-center gap-0.5 border-l border-zinc-700 pl-1"
      >
        <button
          type="button"
          title="Distribute horizontally"
          aria-label="Distribute horizontally"
          disabled={!canDistribute}
          onClick={() => arrangeSelection("distribute-x")}
          className={buttonClass}
        >
          <ArrangeIcon kind="distribute-x" />
        </button>
        <button
          type="button"
          title="Distribute vertically"
          aria-label="Distribute vertically"
          disabled={!canDistribute}
          onClick={() => arrangeSelection("distribute-y")}
          className={buttonClass}
        >
          <ArrangeIcon kind="distribute-y" />
        </button>
      </div>
      <div className="flex items-center gap-0.5 border-l border-zinc-700 pl-1">
        <button
          type="button"
          title="Arrange in a grid"
          aria-label="Arrange in a grid"
          disabled={!canArrange}
          onClick={() => arrangeSelection("grid")}
          className={buttonClass}
        >
          <ArrangeIcon kind="grid" />
        </button>
        <button
          type="button"
          title="Group selection in a frame (Ctrl+G)"
          aria-label="Group selection"
          disabled={!canGroup}
          onClick={() => {
            groupSelection();
          }}
          className={buttonClass}
        >
          <GroupIcon className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          title="Create a reusable subgraph from the selection"
          aria-label="Create subgraph"
          disabled={!canSubgraph}
          onClick={() => {
            createSubgraphFromSelection();
          }}
          className={buttonClass}
        >
          <SubgraphIcon className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          title="Ungroup selected frame (Ctrl+Shift+G)"
          aria-label="Ungroup selection"
          disabled={!canUngroup}
          onClick={() => {
            ungroupSelection();
          }}
          className={buttonClass}
        >
          <UngroupIcon className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          title="Clear selection"
          aria-label="Clear selection"
          onClick={clearSelection}
          className={buttonClass}
        >
          <CloseIcon className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}
