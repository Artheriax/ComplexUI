import { useState } from "react";
import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { useGraphStore } from "../store/useGraphStore";
import { CheckIcon } from "./icons";

interface RenameNodeDialogProps {
  nodeId: string;
  onClose: () => void;
}

export function RenameNodeDialog({ nodeId, onClose }: RenameNodeDialogProps) {
  const node = useGraphStore((state) =>
    state.nodes.find((candidate) => candidate.id === nodeId),
  );
  const renameNode = useGraphStore((state) => state.renameNode);
  const [label, setLabel] = useState(node?.data.label ?? "");
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useDialogA11y<HTMLFormElement>(onClose);

  if (!node) {
    return null;
  }

  const submit = () => {
    if (!renameNode(nodeId, label)) {
      setError("Enter a node name between 1 and 80 characters.");
      return;
    }
    onClose();
  };

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
      <form
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="rename-node-title"
        aria-describedby="rename-node-description"
        tabIndex={-1}
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className="w-full max-w-sm rounded-xl border border-zinc-700 bg-zinc-900 p-4 shadow-2xl outline-none"
      >
        <h2
          id="rename-node-title"
          className="text-sm font-semibold text-zinc-100"
        >
          Rename node
        </h2>
        <p
          id="rename-node-description"
          className="mt-1 text-[11px] text-zinc-500"
        >
          This changes the label shown on the canvas and execution panel.
        </p>

        <label className="mt-4 block">
          <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            Node name
          </span>
          <input
            type="text"
            value={label}
            maxLength={80}
            aria-invalid={error !== null}
            aria-describedby={error ? "rename-node-error" : undefined}
            onChange={(event) => {
              setLabel(event.target.value);
              setError(null);
            }}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600"
          />
        </label>

        {error && (
          <p id="rename-node-error" className="mt-1.5 text-[11px] text-red-300">
            {error}
          </p>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="flex items-center gap-1.5 rounded-md border border-cyan-700 bg-cyan-950/50 px-2.5 py-1 text-xs text-cyan-200 hover:bg-cyan-900/50"
          >
            <CheckIcon className="h-3 w-3" /> Save
          </button>
        </div>
      </form>
    </div>,
    document.body,
  );
}
