import { useMemo, useState } from "react";
import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { useGraphStore } from "../store/useGraphStore";
import { SaveIcon } from "./icons";

interface SaveTemplateDialogProps {
  nodeId: string;
  onClose: () => void;
}

const CATEGORIES = [
  "Custom",
  "Values",
  "Models",
  "Preprocessing",
  "Evaluation",
  "Computer Vision",
  "Image Generation",
];

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80);
}

export function SaveTemplateDialog({
  nodeId,
  onClose,
}: SaveTemplateDialogProps) {
  const node = useGraphStore((state) =>
    state.nodes.find((candidate) => candidate.id === nodeId),
  );
  const saveNodeAsTemplate = useGraphStore(
    (state) => state.saveNodeAsTemplate,
  );
  const [title, setTitle] = useState(node?.data.label ?? "Custom node");
  const [name, setName] = useState(() => slugify(node?.data.label ?? ""));
  const [nameTouched, setNameTouched] = useState(false);
  const [category, setCategory] = useState("Custom");
  const [description, setDescription] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useDialogA11y<HTMLFormElement>(onClose);
  const derivedName = useMemo(() => slugify(title), [title]);
  const effectiveName = nameTouched ? name : derivedName;

  if (!node) {
    return null;
  }

  const submit = async () => {
    if (!effectiveName || !title.trim() || busy) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await saveNodeAsTemplate(nodeId, {
        name: effectiveName,
        title: title.trim(),
        category,
        description: description.trim(),
        code: node.data.code,
        overwrite,
      });
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  return createPortal(
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/70 p-6">
      <form
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="save-template-title"
        aria-describedby="save-template-description"
        tabIndex={-1}
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
        className="w-full max-w-md rounded-xl border border-zinc-700 bg-zinc-900 p-4 shadow-2xl outline-none"
      >
        <h2
          id="save-template-title"
          className="text-sm font-semibold text-zinc-100"
        >
          Save as template
        </h2>
        <p
          id="save-template-description"
          className="mt-1 text-[11px] leading-snug text-zinc-500"
        >
          Save this node&apos;s code as a trusted local template. It will appear
          in the node library for future workflows.
        </p>

        <label className="mt-4 block">
          <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            Title
          </span>
          <input
            type="text"
            value={title}
            maxLength={120}
            autoComplete="off"
            onChange={(event) => setTitle(event.target.value)}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600"
          />
        </label>

        <label className="mt-3 block">
          <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            Template id
          </span>
          <input
            type="text"
            value={effectiveName}
            maxLength={80}
            autoComplete="off"
            onChange={(event) => {
              setName(event.target.value.toLowerCase().replace(/[^a-z0-9_]/g, "_"));
              setNameTouched(true);
            }}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-100 outline-none focus:border-cyan-600"
          />
          <span className="mt-1 block text-[10px] text-zinc-600">
            Lowercase letters, digits and underscores.
          </span>
        </label>

        <label className="mt-3 block">
          <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            Category
          </span>
          <select
            value={category}
            onChange={(event) => setCategory(event.target.value)}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600"
          >
            {CATEGORIES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <label className="mt-3 block">
          <span className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            Description
          </span>
          <textarea
            value={description}
            maxLength={200}
            rows={2}
            onChange={(event) => setDescription(event.target.value)}
            className="w-full resize-none rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600"
          />
        </label>

        <label className="mt-3 flex items-start gap-2 text-[11px] text-zinc-400">
          <input
            type="checkbox"
            checked={overwrite}
            onChange={(event) => setOverwrite(event.target.checked)}
            className="mt-0.5 accent-cyan-500"
          />
          <span>Replace an existing user template with this id.</span>
        </label>

        {error && (
          <p role="alert" className="mt-2 text-[11px] text-red-300">
            {error}
          </p>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-600"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={busy || !effectiveName || !title.trim()}
            className="flex items-center gap-1.5 rounded-md border border-cyan-700 bg-cyan-950/50 px-2.5 py-1 text-xs text-cyan-200 hover:bg-cyan-900/50 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600"
          >
            <SaveIcon className="h-3 w-3" />
            {busy ? "Saving…" : "Save template"}
          </button>
        </div>
      </form>
    </div>,
    document.body,
  );
}
