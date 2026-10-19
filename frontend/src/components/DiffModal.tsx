import { DiffEditor } from "@monaco-editor/react";

import "../lib/monaco";
import { useDialogA11y } from "../hooks/useDialogA11y";
import { useGraphStore } from "../store/useGraphStore";
import { CheckIcon, CloseIcon, SpinnerIcon, WarningIcon } from "./icons";

export function DiffModal() {
  const pendingApproval = useGraphStore((state) => state.pendingApproval);
  const approvalBusy = useGraphStore((state) => state.approvalBusy);
  const approvalError = useGraphStore((state) => state.approvalError);
  const approvePending = useGraphStore((state) => state.approvePending);
  const rejectPending = useGraphStore((state) => state.rejectPending);

  const originalCode = useGraphStore(
    (state) =>
      state.nodes.find((node) => node.id === pendingApproval?.nodeId)?.data
        .templateCode ?? null,
  );
  const dialogRef = useDialogA11y<HTMLDivElement>(() => {
    if (!approvalBusy) {
      rejectPending();
    }
  }, pendingApproval !== null);

  if (!pendingApproval) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="diff-modal-title"
        aria-busy={approvalBusy}
        tabIndex={-1}
        className="flex max-h-full w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-amber-800/70 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-center gap-2 border-b border-zinc-800 bg-amber-950/40 px-3 py-2">
          <WarningIcon className="h-4 w-4 text-amber-400" />
          <span
            id="diff-modal-title"
            className="text-sm font-semibold text-amber-200"
          >
            Security review — modified node code
          </span>
          <span className="rounded bg-zinc-900 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
            sha256:{pendingApproval.codeHash.slice(0, 12)}…
          </span>
          <button
            type="button"
            className="ml-auto rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 disabled:cursor-not-allowed disabled:opacity-50"
            title="Reject changes"
            aria-label="Reject changes and close"
            disabled={approvalBusy}
            onClick={rejectPending}
          >
            <CloseIcon />
          </button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col">
          <div className="grid grid-cols-2 border-b border-zinc-800 bg-zinc-950/60 px-3 py-1 text-[10px] text-zinc-500">
            <span>
              {originalCode ? "official template" : "scratch node — no original"}
            </span>
            <span>modified code (will run with your privileges)</span>
          </div>
          <div className="min-h-0 flex-1">
            <DiffEditor
              height="46vh"
              language="python"
              theme="vs-dark"
              original={originalCode ?? ""}
              modified={pendingApproval.code}
              options={{
                readOnly: true,
                renderSideBySide: true,
                minimap: { enabled: false },
                fontSize: 12,
                automaticLayout: true,
              }}
            />
          </div>
        </div>

        <div className="border-t border-zinc-800 px-3 py-2">
          <p className="text-[11px] leading-snug text-zinc-400">
            This node differs from the official template. Approved code is
            transpiled and executed unsandboxed by the backend for this session.
            Only approve code you have read and understand.
          </p>
          {approvalError && (
            <div
              role="alert"
              className="mt-2 rounded border border-red-900/70 bg-red-950/40 p-2 text-[11px] text-red-300"
            >
              {approvalError}
            </div>
          )}
          <div className="mt-2 flex items-center justify-end gap-2">
            <button
              type="button"
              disabled={approvalBusy}
              onClick={rejectPending}
              className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
            >
              <CloseIcon /> Keep original
            </button>
            <button
              type="button"
              disabled={approvalBusy}
              onClick={() => {
                void approvePending();
              }}
              className="flex items-center gap-1.5 rounded-md border border-emerald-700 bg-emerald-950/60 px-3 py-1 text-xs font-medium text-emerald-200 hover:bg-emerald-900/60 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {approvalBusy ? (
                <SpinnerIcon className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <CheckIcon />
              )}
              Approve &amp; use modified code
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
