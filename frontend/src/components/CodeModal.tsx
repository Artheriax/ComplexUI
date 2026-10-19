import Editor from "@monaco-editor/react";
import { useEffect, useRef, useState } from "react";

import { parseCode } from "../lib/api";
import "../lib/monaco";
import { useDialogA11y } from "../hooks/useDialogA11y";
import { useGraphStore } from "../store/useGraphStore";
import type { NodeSchema } from "../types/graph";
import { SOCKET_COLORS } from "../types/graph";
import { CloseIcon, ResetIcon, SpinnerIcon, WarningIcon } from "./icons";

const PARSE_DEBOUNCE_MS = 400;

interface CodeModalProps {
  nodeId: string;
}

export function CodeModal({ nodeId }: CodeModalProps) {
  const node = useGraphStore(
    (state) => state.nodes.find((candidate) => candidate.id === nodeId) ?? null,
  );
  const closeEditor = useGraphStore((state) => state.closeEditor);
  const saveNodeCode = useGraphStore((state) => state.saveNodeCode);

  const [code, setCode] = useState(() => node?.data.code ?? "");
  const [parsedCode, setParsedCode] = useState(() => node?.data.code ?? "");
  const [draftSchema, setDraftSchema] = useState<NodeSchema | null>(
    () => node?.data.schema ?? null,
  );
  const [parsing, setParsing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const sequenceRef = useRef(0);
  const dialogRef = useDialogA11y<HTMLDivElement>(closeEditor, node !== null);

  useEffect(() => {
    const sequence = ++sequenceRef.current;
    const timer = window.setTimeout(async () => {
      setParsing(true);
      try {
        const schema = await parseCode(code, nodeId);
        if (sequenceRef.current === sequence) {
          setDraftSchema(schema);
          setParsedCode(code);
          setParseError(null);
        }
      } catch (error) {
        if (sequenceRef.current === sequence) {
          setParseError(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (sequenceRef.current === sequence) {
          setParsing(false);
        }
      }
    }, PARSE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [code, nodeId]);

  if (!node) {
    return null;
  }

  const schemaIsCurrent = parsedCode === code;
  const syntaxErrors = schemaIsCurrent ? (draftSchema?.syntax_errors ?? []) : [];
  const hasSyntaxErrors = syntaxErrors.length > 0;
  const busy = parsing || !schemaIsCurrent;
  const dirty = code !== node.data.code;

  const handleSave = () => {
    if (!dirty || !schemaIsCurrent || hasSyntaxErrors || draftSchema === null) {
      return;
    }
    setSaving(true);
    void saveNodeCode(nodeId, code, draftSchema)
      .catch((error: unknown) => {
        setParseError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => setSaving(false));
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-6">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="code-modal-title"
        aria-busy={busy}
        tabIndex={-1}
        className="flex h-full w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
          <span
            id="code-modal-title"
            className="text-sm font-semibold text-zinc-100"
          >
            {node.data.label}
          </span>
          <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
            {nodeId.slice(0, 8)}
          </span>
          {!node.data.isTrusted && (
            <span className="flex items-center gap-1 text-[11px] text-amber-400">
              <WarningIcon className="h-3.5 w-3.5" /> untrusted
            </span>
          )}
          <button
            type="button"
            className="ml-auto rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
            title="Close"
            aria-label="Close code editor"
            onClick={closeEditor}
          >
            <CloseIcon />
          </button>
        </div>

        <div className="min-h-0 flex-1">
          <Editor
            height="100%"
            language="python"
            theme="vs-dark"
            value={code}
            onChange={(value) => setCode(value ?? "")}
            onMount={(editor) => editor.focus()}
            options={{
              ariaLabel: `Python code for ${node.data.label}`,
              minimap: { enabled: false },
              fontSize: 13,
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 4,
            }}
          />
        </div>

        <div className="border-t border-zinc-800 px-3 py-2">
          {hasSyntaxErrors && (
            <div
              role="alert"
              className="mb-2 rounded border border-red-900/70 bg-red-950/40 p-2 text-[11px] text-red-300"
            >
              {syntaxErrors.map((error) => (
                <div key={`${error.line}-${error.column}-${error.message}`}>
                  line {error.line}:{error.column} — {error.message}
                </div>
              ))}
            </div>
          )}
          {parseError && (
            <div
              role="alert"
              className="mb-2 rounded border border-red-900/70 bg-red-950/40 p-2 text-[11px] text-red-300"
            >
              {parseError}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
            {busy ? (
              <span
                role="status"
                aria-live="polite"
                className="flex items-center gap-1.5 text-zinc-400"
              >
                <SpinnerIcon className="h-3 w-3 animate-spin" /> parsing…
              </span>
            ) : (
              <>
                {draftSchema?.inputs.map((socket) => (
                  <span
                    key={`in-${socket.name}`}
                    className="rounded px-1 py-px"
                    style={{
                      backgroundColor: `${SOCKET_COLORS[socket.data_type]}22`,
                      color: SOCKET_COLORS[socket.data_type],
                    }}
                  >
                    {socket.name} in
                  </span>
                ))}
                {draftSchema?.outputs.map((socket) => (
                  <span
                    key={`out-${socket.name}`}
                    className="rounded px-1 py-px"
                    style={{
                      backgroundColor: `${SOCKET_COLORS[socket.data_type]}22`,
                      color: SOCKET_COLORS[socket.data_type],
                    }}
                  >
                    {socket.name} out
                  </span>
                ))}
                {draftSchema && draftSchema.widgets.length > 0 && (
                  <span>{draftSchema.widgets.length} widget(s)</span>
                )}
                {draftSchema && draftSchema.dependencies.length > 0 && (
                  <span>deps: {draftSchema.dependencies.join(", ")}</span>
                )}
              </>
            )}
          </div>

          <div className="mt-2 flex items-center gap-2">
            {node.data.templateCode !== null && (
              <button
                type="button"
                className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
                title="Restore the official template code"
                onClick={() => setCode(node.data.templateCode ?? "")}
              >
                <ResetIcon /> Reset template
              </button>
            )}
            {!node.data.isTrusted && !dirty && (
              <button
                type="button"
                disabled={busy || draftSchema === null}
                title="Review the security diff and approve this exact code for the current session"
                onClick={() => {
                  if (draftSchema) {
                    void saveNodeCode(nodeId, code, draftSchema);
                  }
                }}
                className="flex items-center gap-1.5 rounded-md border border-amber-700 bg-amber-950/50 px-2.5 py-1 text-xs text-amber-200 hover:bg-amber-900/50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <WarningIcon className="h-3.5 w-3.5" /> Review &amp; approve…
              </button>
            )}
            <span className="ml-auto" />
            <button
              type="button"
              className="rounded-md border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
              onClick={closeEditor}
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={
                !dirty ||
                busy ||
                hasSyntaxErrors ||
                draftSchema === null ||
                saving
              }
              onClick={handleSave}
              className="rounded-md border border-cyan-700 bg-cyan-900/50 px-3 py-1 text-xs font-medium text-cyan-200 hover:bg-cyan-800/60 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-600"
            >
              Save &amp; apply
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
