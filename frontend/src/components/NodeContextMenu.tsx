import type { XYPosition } from "@xyflow/react";
import type { ReactNode } from "react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useGraphStore } from "../store/useGraphStore";
import type { NodeColor } from "../types/graph";
import { isFrameNode, NODE_COLOR_ORDER, NODE_COLORS } from "../types/graph";
import {
  CopyIcon,
  CutIcon,
  DuplicateIcon,
  EnterIcon,
  PasteIcon,
  PencilIcon,
  PlayIcon,
  ResetIcon,
  TrashIcon,
  UngroupIcon,
} from "./icons";

interface NodeContextMenuProps {
  nodeId: string;
  anchor: { x: number; y: number };
  flowPosition: XYPosition;
  onClose: () => void;
  onRename: (nodeId: string) => void;
}

const MENU_WIDTH = 224;

function MenuItem({
  icon,
  label,
  shortcut,
  disabled = false,
  danger = false,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  shortcut?: string;
  disabled?: boolean;
  danger?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={onClick}
      className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-[11px] transition-colors disabled:cursor-not-allowed disabled:text-zinc-600 ${
        danger
          ? "text-red-300 hover:bg-red-950/50 disabled:hover:bg-transparent"
          : "text-zinc-300 hover:bg-zinc-800 disabled:hover:bg-transparent"
      }`}
    >
      <span className="flex h-3.5 w-3.5 shrink-0 items-center justify-center">
        {icon}
      </span>
      <span>{label}</span>
      {shortcut && (
        <span className="ml-auto text-[9px] uppercase tracking-wide text-zinc-600">
          {shortcut}
        </span>
      )}
    </button>
  );
}

export function NodeContextMenu({
  nodeId,
  anchor,
  flowPosition,
  onClose,
  onRename,
}: NodeContextMenuProps) {
  const nodes = useGraphStore((state) => state.nodes);
  const templates = useGraphStore((state) => state.templates);
  const hasClipboard = useGraphStore((state) => state.clipboard !== null);
  const running = useGraphStore(
    (state) => state.execution.status === "running",
  );
  const copySelection = useGraphStore((state) => state.copySelection);
  const cutSelection = useGraphStore((state) => state.cutSelection);
  const pasteClipboard = useGraphStore((state) => state.pasteClipboard);
  const duplicateSelection = useGraphStore(
    (state) => state.duplicateSelection,
  );
  const deleteSelection = useGraphStore((state) => state.deleteSelection);
  const openEditor = useGraphStore((state) => state.openEditor);
  const runGraph = useGraphStore((state) => state.runGraph);
  const setNodeColor = useGraphStore((state) => state.setNodeColor);
  const resetNode = useGraphStore((state) => state.resetNode);
  const ungroupSelection = useGraphStore((state) => state.ungroupSelection);
  const enterSubgraph = useGraphStore((state) => state.enterSubgraph);
  const menuRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState(anchor);

  const node = nodes.find((candidate) => candidate.id === nodeId);
  const selectedNodes = nodes.filter((candidate) => candidate.selected);
  const targetIds =
    node?.selected && selectedNodes.length > 0
      ? selectedNodes.map((candidate) => candidate.id)
      : [nodeId];
  const frame = node !== undefined && isFrameNode(node);
  const subgraph = node !== undefined && node.data.isSubgraph === true;
  const resetEnabled =
    node !== undefined &&
    !frame &&
    !subgraph &&
    (node.data.templateName === null ||
      templates.some((template) => template.name === node.data.templateName));

  useEffect(() => {
    const handlePointerDown = (event: PointerEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (
        event.key !== "ArrowDown" &&
        event.key !== "ArrowUp" &&
        event.key !== "Home" &&
        event.key !== "End"
      ) {
        return;
      }
      const menuItems = Array.from(
        menuRef.current?.querySelectorAll<HTMLButtonElement>(
          "button:not([disabled])",
        ) ?? [],
      );
      if (menuItems.length === 0) {
        return;
      }
      const currentIndex = menuItems.indexOf(
        document.activeElement as HTMLButtonElement,
      );
      let nextIndex = 0;
      if (event.key === "ArrowDown") {
        nextIndex =
          currentIndex < 0
            ? 0
            : (currentIndex + 1) % menuItems.length;
      } else if (event.key === "ArrowUp") {
        nextIndex =
          currentIndex < 0
            ? menuItems.length - 1
            : (currentIndex - 1 + menuItems.length) % menuItems.length;
      } else if (event.key === "End") {
        nextIndex = menuItems.length - 1;
      }
      event.preventDefault();
      menuItems[nextIndex]?.focus();
    };
    const handleWheel = () => onClose();
    const handleBlur = () => onClose();
    document.addEventListener("pointerdown", handlePointerDown, true);
    document.addEventListener("keydown", handleKeyDown, true);
    window.addEventListener("wheel", handleWheel, { passive: true });
    window.addEventListener("blur", handleBlur);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown, true);
      document.removeEventListener("keydown", handleKeyDown, true);
      window.removeEventListener("wheel", handleWheel);
      window.removeEventListener("blur", handleBlur);
    };
  }, [onClose]);

  useLayoutEffect(() => {
    const rect = menuRef.current?.getBoundingClientRect();
    setPosition({
      x: Math.max(
        8,
        Math.min(anchor.x, window.innerWidth - (rect?.width ?? MENU_WIDTH) - 8),
      ),
      y: Math.max(
        8,
        Math.min(anchor.y, window.innerHeight - (rect?.height ?? 400) - 8),
      ),
    });
    menuRef.current
      ?.querySelector<HTMLButtonElement>("button:not([disabled])")
      ?.focus();
  }, [anchor.x, anchor.y]);

  if (!node || node.data.subgraphRole) {
    return null;
  }

  const colorTargetIds = targetIds;
  const currentColor = node.data.color;
  const selectedCount = targetIds.length;

  return createPortal(
    <div
      ref={menuRef}
      role="menu"
      aria-label={`Node actions for ${node.data.label}`}
      className="fixed z-[60] max-h-[calc(100vh-16px)] w-56 overflow-y-auto rounded-lg border border-zinc-700 bg-zinc-900 p-1.5 shadow-2xl"
      style={{ left: position.x, top: position.y }}
      onContextMenu={(event) => event.preventDefault()}
    >
      <div className="px-2 pb-1.5 pt-1">
        <div className="truncate text-xs font-semibold text-zinc-100">
          {node.data.label}
        </div>
        <div className="mt-0.5 text-[9px] uppercase tracking-wide text-zinc-600">
          {selectedCount > 1
            ? `${selectedCount} nodes selected`
            : frame
              ? "Frame"
              : subgraph
                ? "Subgraph instance"
                : (node.data.templateName ?? "Scratch node")}
        </div>
      </div>

      <div
        role="group"
        aria-label="Node color"
        className="mb-1 flex items-center gap-1 border-y border-zinc-800 px-2 py-2"
      >
        {NODE_COLOR_ORDER.map((color) => (
          <ColorSwatch
            key={color}
            color={color}
            selected={currentColor === color}
            onClick={() => setNodeColor(colorTargetIds, color)}
          />
        ))}
      </div>

      <MenuItem
        icon={<PencilIcon className="h-3.5 w-3.5" />}
        label={subgraph ? "Rename instance…" : "Rename…"}
        onClick={() => {
          onClose();
          onRename(nodeId);
        }}
      />
      {subgraph && (
        <MenuItem
          icon={<EnterIcon className="h-3.5 w-3.5" />}
          label="Open subgraph"
          onClick={() => {
            enterSubgraph(nodeId);
            onClose();
          }}
        />
      )}
      <MenuItem
        icon={<CopyIcon className="h-3.5 w-3.5" />}
        label="Copy"
        shortcut="Ctrl C"
        disabled={selectedCount === 0}
        onClick={() => {
          copySelection();
          onClose();
        }}
      />
      <MenuItem
        icon={<CutIcon className="h-3.5 w-3.5" />}
        label="Cut"
        shortcut="Ctrl X"
        disabled={selectedCount === 0}
        onClick={() => {
          cutSelection();
          onClose();
        }}
      />
      <MenuItem
        icon={<PasteIcon className="h-3.5 w-3.5" />}
        label="Paste here"
        shortcut="Ctrl V"
        disabled={!hasClipboard}
        onClick={() => {
          void pasteClipboard(flowPosition);
          onClose();
        }}
      />
      <MenuItem
        icon={<DuplicateIcon className="h-3.5 w-3.5" />}
        label="Duplicate"
        shortcut="Ctrl D"
        disabled={selectedCount === 0}
        onClick={() => {
          void duplicateSelection();
          onClose();
        }}
      />
      {frame && (
        <MenuItem
          icon={<UngroupIcon className="h-3.5 w-3.5" />}
          label="Ungroup frame"
          onClick={() => {
            ungroupSelection();
            onClose();
          }}
        />
      )}

      <div className="my-1 border-t border-zinc-800" />

      {!frame && !subgraph && (
        <MenuItem
          icon={<PencilIcon className="h-3.5 w-3.5" />}
          label="Edit code"
          onClick={() => {
            openEditor(nodeId);
            onClose();
          }}
        />
      )}
      {!frame && !subgraph && (
        <MenuItem
          icon={<PlayIcon className="h-3.5 w-3.5" />}
          label="Run to this node"
          disabled={running}
          onClick={() => {
            void runGraph([nodeId]);
            onClose();
          }}
        />
      )}
      {!frame && !subgraph && (
        <MenuItem
          icon={<ResetIcon className="h-3.5 w-3.5" />}
          label="Reset node"
          disabled={!resetEnabled}
          onClick={() => {
            resetNode(nodeId);
            onClose();
          }}
        />
      )}
      <MenuItem
        icon={<TrashIcon className="h-3.5 w-3.5" />}
        label={
          frame
            ? "Delete frame"
            : subgraph
              ? "Delete subgraph instance"
              : "Delete"
        }
        shortcut="Del"
        danger
        disabled={selectedCount === 0}
        onClick={() => {
          deleteSelection();
          onClose();
        }}
      />
    </div>,
    document.body,
  );
}

function ColorSwatch({
  color,
  selected,
  onClick,
}: {
  color: NodeColor;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={`Set color ${NODE_COLORS[color].label}`}
      aria-pressed={selected}
      title={NODE_COLORS[color].label}
      onClick={onClick}
      className={`h-4 w-4 rounded-full border transition-transform hover:scale-110 ${
        selected
          ? "border-white ring-2 ring-cyan-500/60"
          : "border-zinc-950"
      }`}
      style={{ backgroundColor: NODE_COLORS[color].accent }}
    />
  );
}
