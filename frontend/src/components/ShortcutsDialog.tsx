import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { useUiStore } from "../store/useUiStore";
import { CloseIcon, KeyboardIcon } from "./icons";

const SHORTCUT_GROUPS: { title: string; items: { keys: string; action: string }[] }[] =
  [
    {
      title: "Workflow",
      items: [
        { keys: "Ctrl+S", action: "Save the active workflow" },
        { keys: "Ctrl+Z", action: "Undo" },
        { keys: "Ctrl+Y / Ctrl+Shift+Z", action: "Redo" },
      ],
    },
    {
      title: "Canvas",
      items: [
        { keys: "Double-click empty canvas", action: "Open node search" },
        { keys: "Right-click empty canvas", action: "Open the canvas menu" },
        { keys: "Double-click node", action: "Edit code or enter a subgraph" },
        { keys: "Ctrl+A", action: "Select all nodes" },
        { keys: "Ctrl+C / Ctrl+X / Ctrl+V", action: "Copy / cut / paste" },
        { keys: "Ctrl+D", action: "Duplicate the selection" },
        { keys: "Ctrl+G / Ctrl+Shift+G", action: "Group / ungroup in a frame" },
        { keys: "Delete / Backspace", action: "Delete the selection" },
        { keys: "Ctrl++ / Ctrl+-", action: "Zoom in / out" },
      ],
    },
  ];

export function ShortcutsDialog() {
  const closeShortcuts = useUiStore((state) => state.closeShortcuts);
  const dialogRef = useDialogA11y<HTMLDivElement>(closeShortcuts);

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        tabIndex={-1}
        className="w-full max-w-2xl overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
          <KeyboardIcon className="h-4 w-4 text-cyan-400" />
          <span className="text-sm font-semibold text-zinc-100">
            Keyboard shortcuts
          </span>
          <button
            type="button"
            onClick={closeShortcuts}
            aria-label="Close keyboard shortcuts"
            title="Close keyboard shortcuts"
            className="ml-auto rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
          >
            <CloseIcon />
          </button>
        </div>
        <div className="grid max-h-[70vh] gap-6 overflow-y-auto p-4 md:grid-cols-2">
          {SHORTCUT_GROUPS.map((group) => (
            <section key={group.title}>
              <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                {group.title}
              </h2>
              <dl className="space-y-1.5">
                {group.items.map((item) => (
                  <div
                    key={item.keys}
                    className="flex items-start justify-between gap-3 rounded-md border border-zinc-800/70 bg-zinc-950/40 px-2 py-1.5"
                  >
                    <dt className="text-[11px] text-zinc-300">{item.action}</dt>
                    <dd className="shrink-0 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                      {item.keys}
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
        <p className="border-t border-zinc-800 px-3 py-2 text-[10px] text-zinc-600">
          On macOS, Cmd works in place of Ctrl. Shortcuts are ignored while a
          dialog is open or focus is inside a text field (Ctrl+S still saves).
        </p>
      </div>
    </div>,
    document.body,
  );
}
