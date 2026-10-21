import type { XYPosition } from "@xyflow/react";
import type {
  KeyboardEvent as ReactKeyboardEvent,
  RefObject,
} from "react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { nodeSearchPlacement } from "../lib/nodeSearchPlacement";
import { matchTemplateConnection } from "../lib/socketCompatibility";
import type { TemplateConnectionMatch } from "../lib/socketCompatibility";
import { searchTemplates } from "../lib/templateSearch";
import { useGraphStore } from "../store/useGraphStore";
import type {
  PendingConnection,
  SocketType,
  TemplateInfo,
} from "../types/graph";
import { SOCKET_COLORS, SOCKET_LABELS } from "../types/graph";
import { CloseIcon, PlusIcon, SearchIcon } from "./icons";

interface NodeSearchPopupProps {
  anchor: { x: number; y: number };
  flowPosition: XYPosition;
  connection: PendingConnection | null;
  onClose: () => void;
  inputRef?: RefObject<HTMLInputElement | null>;
}



interface SearchItem {
  key: string;
  template: TemplateInfo | null;
  title: string;
  description: string;
  category: string;
  match: TemplateConnectionMatch | null;
  compatible: boolean;
}

function SocketTag({
  name,
  type,
  direction,
}: {
  name: string;
  type: SocketType;
  direction: "input" | "output";
}) {
  return (
    <span
      className="rounded px-1 py-px text-[9px]"
      style={{
        backgroundColor: `${SOCKET_COLORS[type]}22`,
        color: SOCKET_COLORS[type],
      }}
    >
      {name} {direction}
    </span>
  );
}

function connectionLabel(
  item: SearchItem,
  connection: PendingConnection,
): string {
  if (!item.match) {
    return connection.bundleEdgeIds?.length
      ? "No compatible output"
      : "No connection will be created";
  }
  if (connection.bundleEdgeIds?.length) {
    return `Re-source to ${item.match.socketName} · ${SOCKET_LABELS[item.match.socketType]}`;
  }
  return connection.handleType === "source"
    ? `Connect to ${item.match.socketName} · ${SOCKET_LABELS[item.match.socketType]}`
    : `Connect from ${item.match.socketName} · ${SOCKET_LABELS[item.match.socketType]}`;
}

export function NodeSearchPopup({
  anchor,
  flowPosition,
  connection,
  onClose,
  inputRef: externalInputRef,
}: NodeSearchPopupProps) {
  const templates = useGraphStore((state) => state.templates);
  const templatesLoaded = useGraphStore((state) => state.templatesLoaded);
  const templatesError = useGraphStore((state) => state.templatesError);
  const addNodeFromTemplate = useGraphStore(
    (state) => state.addNodeFromTemplate,
  );
  const addConnectedNodeFromTemplate = useGraphStore(
    (state) => state.addConnectedNodeFromTemplate,
  );
  const addScratchNode = useGraphStore((state) => state.addScratchNode);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const popupRef = useRef<HTMLDivElement>(null);
  const localInputRef = useRef<HTMLInputElement>(null);
  const inputRef = externalInputRef ?? localInputRef;
  const listId = useId();

  const templateItems = useMemo<SearchItem[]>(() => {
    return searchTemplates(templates, query).map((template) => {
      const match = connection
        ? connection.bundleEdgeIds?.length
          ? matchTemplateConnection(template, "target", connection.dataType)
          : matchTemplateConnection(
              template,
              connection.handleType,
              connection.dataType,
            )
        : null;
      return {
        key: template.name,
        template,
        title: template.title,
        description: template.description,
        category: template.category,
        match,
        compatible: connection === null || match !== null,
      };
    });
  }, [connection, query, templates]);

  const compatibleExists = templateItems.some((item) => item.compatible);
  const visibleTemplates =
    connection && !showAll
      ? templateItems.filter((item) => item.compatible)
      : templateItems;
  const needle = query.trim().toLowerCase();
  const scratchMatches =
    connection === null &&
    (needle.length === 0 ||
      "new scratch node custom python".includes(needle));
  const items = useMemo<SearchItem[]>(() => {
    const scratch: SearchItem[] = scratchMatches
      ? [
          {
            key: "__scratch__",
            template: null,
            title: "New scratch node",
            description: "Start with an empty Python node.",
            category: "Custom",
            match: null,
            compatible: true,
          },
        ]
      : [];
    return [...scratch, ...visibleTemplates];
  }, [scratchMatches, visibleTemplates]);

  const selectedIndex =
    items.length === 0 ? 0 : Math.min(activeIndex, items.length - 1);

  useEffect(() => {
    inputRef.current?.focus();
  }, [inputRef]);

  useEffect(() => {
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      const inCompanion =
        target instanceof Element &&
        target.closest("[data-canvas-context-menu]") !== null;
      if (
        !inCompanion &&
        popupRef.current &&
        !popupRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    document.addEventListener("pointerdown", handlePointerDown, true);
    document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown, true);
      document.removeEventListener("keydown", handleKeyDown, true);
    };
  }, [onClose]);

  useEffect(() => {
    const active = document.getElementById(
      `${listId}-option-${selectedIndex}`,
    );
    active?.scrollIntoView?.({ block: "nearest" });
  }, [listId, selectedIndex]);

  const choose = (item: SearchItem) => {
    if (!item.template) {
      addScratchNode(flowPosition);
      onClose();
      return;
    }
    if (connection && item.match) {
      const created = addConnectedNodeFromTemplate(
        item.template,
        flowPosition,
        connection,
      );
      if (!created) {
        return;
      }
      onClose();
      return;
    }
    addNodeFromTemplate(item.template, flowPosition);
    onClose();
  };

  const handleInputKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex(
        items.length === 0 ? 0 : Math.min(items.length - 1, selectedIndex + 1),
      );
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex(Math.max(0, selectedIndex - 1));
      return;
    }
    if (event.key === "Enter") {
      const item = items[selectedIndex];
      if (item) {
        event.preventDefault();
        choose(item);
      }
    }
  };

  const { left, top } = nodeSearchPlacement(anchor);

  return createPortal(
    <div
      ref={popupRef}
      data-node-search-popup="true"
      className="fixed z-[60] w-80 overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-2xl"
      style={{ left, top }}
      onContextMenu={(event) => event.preventDefault()}
    >
      <div className="flex items-center gap-2 border-b border-zinc-800 px-2.5 py-2">
        <SearchIcon className="h-3.5 w-3.5 text-cyan-400" />
        <span className="text-xs font-semibold text-zinc-200">
          {connection ? "Connect a node" : "Add a node"}
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close node search"
          title="Close node search"
          className="ml-auto rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <CloseIcon className="h-3 w-3" />
        </button>
      </div>

      <div className="border-b border-zinc-800 p-2">
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-label="Search nodes"
          aria-autocomplete="list"
          aria-expanded="true"
          aria-controls={listId}
          aria-activedescendant={
            items[selectedIndex]
              ? `${listId}-option-${selectedIndex}`
              : undefined
          }
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setActiveIndex(0);
          }}
          onKeyDown={handleInputKeyDown}
          placeholder="Search nodes…"
          className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-cyan-600"
        />
      </div>

      <div
        id={listId}
        role="listbox"
        aria-label="Matching nodes"
        className="max-h-72 overflow-y-auto p-1.5"
      >
        {items.map((item, index) => {
          const active = index === selectedIndex;
          return (
            <button
              key={item.key}
              id={`${listId}-option-${index}`}
              type="button"
              role="option"
              aria-selected={active}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={() => choose(item)}
              className={`mb-1 flex w-full items-start gap-2 rounded-md border px-2 py-1.5 text-left transition-colors ${
                active
                  ? "border-cyan-700 bg-cyan-950/40"
                  : "border-transparent hover:border-zinc-700 hover:bg-zinc-800/60"
              }`}
            >
              {item.template ? (
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded bg-zinc-800 text-[9px] font-semibold uppercase text-zinc-400">
                  {item.template.title.slice(0, 2)}
                </span>
              ) : (
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded bg-zinc-800 text-cyan-300">
                  <PlusIcon className="h-3 w-3" />
                </span>
              )}
              <span className="min-w-0 flex-1">
                <span className="flex items-baseline gap-1.5">
                  <span className="truncate text-xs font-medium text-zinc-100">
                    {item.title}
                  </span>
                  <span className="ml-auto shrink-0 text-[9px] uppercase tracking-wide text-zinc-600">
                    {item.category}
                  </span>
                </span>
                <span className="mt-0.5 block truncate text-[10px] text-zinc-500">
                  {item.description}
                </span>
                {item.template && (
                  <span className="mt-1 flex flex-wrap items-center gap-1">
                    {item.template.node_schema.inputs.slice(0, 2).map((socket) => (
                      <SocketTag
                        key={`in-${socket.name}`}
                        name={socket.name}
                        type={socket.data_type}
                        direction="input"
                      />
                    ))}
                    {item.template.node_schema.outputs
                      .slice(0, 2)
                      .map((socket) => (
                        <SocketTag
                          key={`out-${socket.name}`}
                          name={socket.name}
                          type={socket.data_type}
                          direction="output"
                        />
                      ))}
                  </span>
                )}
                {connection && (
                  <span
                    className={`mt-1 block text-[10px] ${
                      item.match ? "text-cyan-300/80" : "text-amber-300/80"
                    }`}
                  >
                    {connectionLabel(item, connection)}
                  </span>
                )}
              </span>
            </button>
          );
        })}

        {items.length === 0 && (
          <div className="px-2 py-5 text-center text-[11px] text-zinc-500">
            {!templatesLoaded && !templatesError
              ? "Loading templates…"
              : connection && !showAll && !compatibleExists
                ? "No compatible nodes found."
                : query.trim()
                  ? `No nodes match “${query.trim()}”.`
                  : "No nodes available."}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between gap-2 border-t border-zinc-800 px-2.5 py-1.5">
        <span className="text-[10px] text-zinc-600">
          {items.length} result{items.length === 1 ? "" : "s"}
        </span>
        {connection && (
          <button
            type="button"
            onClick={() => {
              setShowAll((value) => !value);
              setActiveIndex(0);
            }}
            className="rounded px-1.5 py-0.5 text-[10px] text-cyan-300 hover:bg-zinc-800"
          >
            {showAll ? "Compatible only" : "Show all nodes"}
          </button>
        )}
      </div>
    </div>,
    document.body,
  );
}
