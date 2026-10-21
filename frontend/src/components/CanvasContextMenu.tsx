import type { XYPosition } from "@xyflow/react";
import type { ReactNode } from "react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useWorkflowFileActions } from "../hooks/useWorkflowFileActions";
import { useGraphStore } from "../store/useGraphStore";
import {
  CheckIcon,
  ChevronDownIcon,
  DownloadIcon,
  FileIcon,
  FocusIcon,
  PasteIcon,
  PlusIcon,
  RedoIcon,
  SaveIcon,
  TrashIcon,
  UndoIcon,
  UploadIcon,
} from "./icons";

const MENU_WIDTH = 216;
const SUBMENU_WIDTH = 224;
const MENU_GAP = 2;
const VIEWPORT_MARGIN = 8;

function MenuItem({
  icon,
  label,
  shortcut,
  disabled = false,
  danger = false,
  expanded,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  shortcut?: string;
  disabled?: boolean;
  danger?: boolean;
  expanded?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      aria-haspopup={expanded === undefined ? undefined : "menu"}
      aria-expanded={expanded}
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
      {expanded !== undefined && (
        <ChevronDownIcon
          className={`ml-1 h-3 w-3 -rotate-90 ${disabled ? "opacity-40" : ""}`}
        />
      )}
    </button>
  );
}

export function CanvasContextMenu({
  anchor,
  flowPosition,
  onClose,
  onDismissAll,
  onFocusSearch,
  onFitView,
  onImport,
}: {
  anchor: { x: number; y: number };
  flowPosition: XYPosition;
  onClose: () => void;
  onDismissAll: () => void;
  onFocusSearch: () => void;
  onFitView: () => void;
  onImport: () => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const fileItemRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState(anchor);
  const [fileOpen, setFileOpen] = useState(false);
  const [submenuTop, setSubmenuTop] = useState(0);
  const hasClipboard = useGraphStore((state) => state.clipboard !== null);
  const hasContent = useGraphStore(
    (state) =>
      state.nodes.length > 0 ||
      state.edges.length > 0 ||
      Object.keys(state.subgraphs).length > 0,
  );
  const canUndo = useGraphStore((state) => state.past.length > 0);
  const canRedo = useGraphStore((state) => state.future.length > 0);
  const running = useGraphStore(
    (state) => state.execution.status === "running",
  );
  const saveBusy = useGraphStore((state) => state.saveBusy);
  const pasteClipboard = useGraphStore((state) => state.pasteClipboard);
  const selectAllNodes = useGraphStore((state) => state.selectAllNodes);
  const undo = useGraphStore((state) => state.undo);
  const redo = useGraphStore((state) => state.redo);
  const {
    exporting,
    save,
    exportJson,
    exportArtifact,
    clearCanvas,
  } = useWorkflowFileActions();

  const submenuOnLeft =
    position.x + MENU_WIDTH + SUBMENU_WIDTH + VIEWPORT_MARGIN >
    window.innerWidth;

  useEffect(() => {
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Element) {
        if (target.closest("[data-node-search-popup]")) {
          return;
        }
        if (rootRef.current?.contains(target)) {
          return;
        }
      }
      onDismissAll();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        if (fileOpen) {
          setFileOpen(false);
        } else {
          onDismissAll();
        }
      }
    };
    document.addEventListener("pointerdown", handlePointerDown, true);
    document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown, true);
      document.removeEventListener("keydown", handleKeyDown, true);
    };
  }, [fileOpen, onDismissAll]);

  useLayoutEffect(() => {
    const rect = rootRef.current?.getBoundingClientRect();
    setPosition({
      x: Math.max(
        VIEWPORT_MARGIN,
        Math.min(
          anchor.x,
          window.innerWidth - (rect?.width ?? MENU_WIDTH) - VIEWPORT_MARGIN,
        ),
      ),
      y: Math.max(
        VIEWPORT_MARGIN,
        Math.min(
          anchor.y,
          window.innerHeight - (rect?.height ?? 360) - VIEWPORT_MARGIN,
        ),
      ),
    });
  }, [anchor.x, anchor.y]);

  const openFileMenu = () => {
    const root = rootRef.current;
    const item = fileItemRef.current;
    if (root && item) {
      const rootRect = root.getBoundingClientRect();
      const itemRect = item.getBoundingClientRect();
      setSubmenuTop(
        Math.max(
          VIEWPORT_MARGIN - position.y,
          Math.min(
            itemRect.top - rootRect.top - 4,
            window.innerHeight - position.y - 250,
          ),
        ),
      );
    }
    setFileOpen(true);
  };

  const closeAfter = (action: () => void) => {
    action();
    onDismissAll();
  };

  return createPortal(
    <div
      ref={rootRef}
      role="menu"
      aria-label="Canvas actions"
      data-canvas-context-menu="true"
      className="fixed z-[60] rounded-lg border border-zinc-700 bg-zinc-900 p-1.5 shadow-2xl"
      style={{ left: position.x, top: position.y, width: MENU_WIDTH }}
      onContextMenu={(event) => event.preventDefault()}
    >
      <div className="max-h-[calc(100vh-16px)] overflow-y-auto">
        <div ref={fileItemRef}>
          <MenuItem
            icon={<FileIcon className="h-3.5 w-3.5" />}
            label="File"
            expanded={fileOpen}
            onClick={() => {
              if (fileOpen) {
                setFileOpen(false);
              } else {
                openFileMenu();
              }
            }}
          />
        </div>

        <div className="my-1 border-t border-zinc-800" />

        <MenuItem
          icon={<PlusIcon className="h-3.5 w-3.5" />}
          label="Add node…"
          onClick={() => {
            onFocusSearch();
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
            onDismissAll();
          }}
        />
        <MenuItem
          icon={<CheckIcon className="h-3.5 w-3.5" />}
          label="Select all"
          shortcut="Ctrl A"
          disabled={!hasContent}
          onClick={() => {
            selectAllNodes();
            onDismissAll();
          }}
        />
        <MenuItem
          icon={<FocusIcon className="h-3.5 w-3.5" />}
          label="Fit view"
          onClick={() => {
            onFitView();
            onDismissAll();
          }}
        />

        <div className="my-1 border-t border-zinc-800" />

        <MenuItem
          icon={<UndoIcon className="h-3.5 w-3.5" />}
          label="Undo"
          shortcut="Ctrl Z"
          disabled={!canUndo || running}
          onClick={() => {
            undo();
            onDismissAll();
          }}
        />
        <MenuItem
          icon={<RedoIcon className="h-3.5 w-3.5" />}
          label="Redo"
          shortcut="Ctrl Y"
          disabled={!canRedo || running}
          onClick={() => {
            redo();
            onDismissAll();
          }}
        />
      </div>

      {fileOpen && (
        <div
          role="menu"
          aria-label="File actions"
          data-canvas-context-menu="true"
          className="absolute z-[61] rounded-lg border border-zinc-700 bg-zinc-900 p-1.5 shadow-2xl"
          style={{
            width: SUBMENU_WIDTH,
            top: submenuTop,
            ...(submenuOnLeft
              ? { right: MENU_WIDTH - MENU_GAP }
              : { left: MENU_WIDTH - MENU_GAP }),
          }}
        >
          <MenuItem
            icon={<SaveIcon className="h-3.5 w-3.5" />}
            label={saveBusy ? "Saving…" : "Save project"}
            shortcut="Ctrl S"
            disabled={!hasContent || saveBusy}
            onClick={() => closeAfter(save)}
          />
          <div className="my-1 border-t border-zinc-800" />
          <MenuItem
            icon={<DownloadIcon className="h-3.5 w-3.5" />}
            label="Export workflow JSON"
            disabled={!hasContent || exporting}
            onClick={() => closeAfter(exportJson)}
          />
          <MenuItem
            icon={<DownloadIcon className="h-3.5 w-3.5" />}
            label="Export Python script"
            disabled={!hasContent || exporting}
            onClick={() => {
              void exportArtifact("python");
              onDismissAll();
            }}
          />
          <MenuItem
            icon={<DownloadIcon className="h-3.5 w-3.5" />}
            label="Export Jupyter notebook"
            disabled={!hasContent || exporting}
            onClick={() => {
              void exportArtifact("notebook");
              onDismissAll();
            }}
          />
          <div className="my-1 border-t border-zinc-800" />
          <MenuItem
            icon={<UploadIcon className="h-3.5 w-3.5" />}
            label="Import workflow…"
            onClick={() => {
              onImport();
              onDismissAll();
            }}
          />
          <MenuItem
            icon={<TrashIcon className="h-3.5 w-3.5" />}
            label="Clear canvas"
            danger
            disabled={!hasContent || running}
            onClick={() => closeAfter(clearCanvas)}
          />
        </div>
      )}
    </div>,
    document.body,
  );
}

