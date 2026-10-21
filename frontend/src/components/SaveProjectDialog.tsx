import { useState } from "react";
import { createPortal } from "react-dom";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { MAX_PROJECT_NAME_CHARS } from "../lib/persistence";
import { useGraphStore } from "../store/useGraphStore";
import { SaveIcon } from "./icons";

export function SaveProjectDialog() {
  const projectName = useGraphStore((state) => state.projectName);
  const projectSlug = useGraphStore((state) => state.projectSlug);
  const saveBusy = useGraphStore((state) => state.saveBusy);
  const saveError = useGraphStore((state) => state.saveError);
  const saveProject = useGraphStore((state) => state.saveProject);
  const closeSaveDialog = useGraphStore((state) => state.closeSaveDialog);
  const [name, setName] = useState(projectName ?? "");
  const [overwrite, setOverwrite] = useState(false);
  const dialogRef = useDialogA11y<HTMLFormElement>(closeSaveDialog);

  const submit = () => {
    void saveProject(name, overwrite);
  };

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
      <form
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="save-project-title"
        aria-describedby="save-project-description"
        tabIndex={-1}
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className="w-full max-w-md rounded-xl border border-zinc-700 bg-zinc-900 p-4 shadow-2xl outline-none"
      >
        <h2
          id="save-project-title"
          className="text-sm font-semibold text-zinc-100"
        >
          {projectSlug ? "Save project" : "Name this project"}
        </h2>
        <p
          id="save-project-description"
          className="mt-1 text-[11px] text-zinc-500"
        >
          Projects are stored in the local workspace with up to 30 version
          snapshots. Press Ctrl+S to save again without this dialog.
        </p>

        <label className="mt-4 block">
          <span className="mb-1 flex items-center justify-between text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            <span>Project name</span>
            <span>
              {name.trim().length}/{MAX_PROJECT_NAME_CHARS}
            </span>
          </span>
          <input
            type="text"
            aria-label="Project name"
            value={name}
            maxLength={MAX_PROJECT_NAME_CHARS}
            autoComplete="off"
            aria-invalid={saveError !== null}
            aria-describedby={saveError ? "save-project-error" : undefined}
            onChange={(event) => {
              setName(event.target.value);
              useGraphStore.setState({ saveError: null });
            }}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-100 outline-none focus:border-cyan-600"
          />
        </label>

        {!projectSlug && (
          <label className="mt-3 flex items-start gap-2 text-[11px] text-zinc-400">
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(event) => {
                setOverwrite(event.target.checked);
                useGraphStore.setState({ saveError: null });
              }}
              className="mt-0.5 accent-cyan-500"
            />
            <span>
              Replace an existing project with the same name. The previous
              contents are kept as a version snapshot.
            </span>
          </label>
        )}

        {saveError && (
          <p
            id="save-project-error"
            role="alert"
            className="mt-2 text-[11px] text-red-300"
          >
            {saveError}
          </p>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={closeSaveDialog}
            disabled={saveBusy}
            className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-600"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={saveBusy || name.trim().length === 0}
            className="flex items-center gap-1.5 rounded-md border border-cyan-700 bg-cyan-950/50 px-2.5 py-1 text-xs text-cyan-200 hover:bg-cyan-900/50 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600"
          >
            <SaveIcon className="h-3 w-3" />
            {saveBusy ? "Saving…" : "Save project"}
          </button>
        </div>
      </form>
    </div>,
    document.body,
  );
}
