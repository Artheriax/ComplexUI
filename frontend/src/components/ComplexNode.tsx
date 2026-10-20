import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { useState } from "react";
import type { CSSProperties } from "react";

import { useGraphStore } from "../store/useGraphStore";
import type {
  AppNode,
  NodeExecutionState,
  PreviewValue,
  SocketType,
} from "../types/graph";
import type { RichOutput } from "../types/graph";
import {
  NODE_COLORS,
  SOCKET_COLORS,
  SOCKET_LABELS,
  summarisePreview,
} from "../types/graph";
import { PencilIcon, DuplicateIcon, PlayIcon, TrashIcon, WarningIcon } from "./icons";
import { PreviewValueView } from "./PreviewValueView";
import { RichOutputView } from "./RichOutputView";
import { WidgetField } from "./WidgetField";

const statusBorder: Record<NodeExecutionState["status"] | "idle", string> = {
  idle: "border-zinc-700",
  running: "border-cyan-400 shadow-cyan-500/20",
  completed: "border-emerald-500/70",
  error: "border-red-500",
};

function socketStyle(type: SocketType): CSSProperties {
  return { "--cu-handle-color": SOCKET_COLORS[type] } as CSSProperties;
}

function SocketChip({ type }: { type: SocketType }) {
  return (
    <span
      className="rounded px-1 py-px text-[9px] font-medium uppercase tracking-wide"
      style={{ backgroundColor: `${SOCKET_COLORS[type]}22`, color: SOCKET_COLORS[type] }}
    >
      {SOCKET_LABELS[type]}
    </span>
  );
}

export function ComplexNode({ id, data, selected }: NodeProps<AppNode>) {
  const execution = useGraphStore((state) => state.execution.nodeStates[id]);
  const running = useGraphStore((state) => state.execution.status === "running");
  const openEditor = useGraphStore((state) => state.openEditor);
  const deleteNode = useGraphStore((state) => state.deleteNode);
  const duplicateNode = useGraphStore((state) => state.duplicateNode);
  const runGraph = useGraphStore((state) => state.runGraph);
  const updateWidgetValue = useGraphStore((state) => state.updateWidgetValue);

  const status = execution?.status ?? "idle";
  const borderClass = statusBorder[status];
  const accent = NODE_COLORS[data.color]?.accent ?? NODE_COLORS.default.accent;

  return (
    <div
      className={`min-w-[230px] max-w-[280px] rounded-lg border bg-zinc-900 shadow-xl ${borderClass} ${
        selected ? "ring-2 ring-cyan-400/60" : ""
      }`}
    >
      <div className="relative flex items-center gap-1.5 rounded-t-lg border-b border-zinc-800 bg-zinc-900/95 px-2 py-1.5">
        <span
          aria-hidden="true"
          data-node-color={data.color}
          className="pointer-events-none absolute inset-x-0 top-0 h-0.5 rounded-t-lg"
          style={{ backgroundColor: accent }}
        />
        {!data.isTrusted && (
          <span
            role="img"
            aria-label="Untrusted code"
            className="text-amber-400"
            title="Untrusted code — open the editor to review and approve"
          >
            <WarningIcon className="h-3.5 w-3.5" />
          </span>
        )}
        <span className="truncate text-xs font-semibold text-zinc-100" title={data.label}>
          {data.label}
        </span>
        <span className="ml-auto flex items-center gap-1">
          <button
            type="button"
            className="nodrag rounded p-1 text-cyan-300/90 hover:bg-zinc-700 hover:text-cyan-200 disabled:cursor-not-allowed disabled:text-zinc-600"
            title="Run workflow up to this node"
            aria-label={`Run workflow up to ${data.label}`}
            disabled={running}
            onClick={() => {
              void runGraph([id]);
            }}
          >
            <PlayIcon className="h-3 w-3" />
          </button>
          <button
            type="button"
            className="nodrag rounded p-1 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-100"
            title="Edit code"
            aria-label={`Edit code for ${data.label}`}
            onClick={() => openEditor(id)}
          >
            <PencilIcon />
          </button>
          <button
            type="button"
            className="nodrag rounded p-1 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-100"
            title="Duplicate node"
            aria-label={`Duplicate ${data.label}`}
            onClick={() => {
              void duplicateNode(id);
            }}
          >
            <DuplicateIcon />
          </button>
          <button
            type="button"
            className="nodrag rounded p-1 text-zinc-400 hover:bg-red-900/60 hover:text-red-300"
            title="Delete node"
            aria-label={`Delete ${data.label}`}
            onClick={() => deleteNode(id)}
          >
            <TrashIcon />
          </button>
        </span>
      </div>

      {data.schema.inputs.length > 0 && (
        <div className="border-b border-zinc-800 py-0.5">
          {data.schema.inputs.map((socket) => (
            <div key={socket.name} className="relative flex items-center gap-2 py-1 pl-3 pr-2">
              <Handle
                type="target"
                position={Position.Left}
                id={socket.name}
                style={socketStyle(socket.data_type)}
              />
              <span className="text-xs text-zinc-300">{socket.name}</span>
              <span className="ml-auto">
                <SocketChip type={socket.data_type} />
              </span>
            </div>
          ))}
        </div>
      )}

      {data.schema.widgets.length > 0 && (
        <div className="space-y-1 border-b border-zinc-800 px-2 py-1.5">
          {data.schema.widgets.map((widget) => (
            <WidgetField
              key={widget.name}
              widget={widget}
              value={data.widgetValues[widget.name]}
              onChange={(value) => updateWidgetValue(id, widget.name, value)}
            />
          ))}
        </div>
      )}

      {data.schema.outputs.length > 0 && (
        <div className="py-0.5">
          {data.schema.outputs.map((socket) => (
            <div key={socket.name} className="relative flex items-center gap-2 py-1 pl-2 pr-3">
              <span className="ml-auto text-xs text-zinc-300">{socket.name}</span>
              <SocketChip type={socket.data_type} />
              <Handle
                type="source"
                position={Position.Right}
                id={socket.name}
                style={socketStyle(socket.data_type)}
              />
            </div>
          ))}
        </div>
      )}

      {execution?.outputs && Object.keys(execution.outputs).length > 0 && (
        <NodePreviews outputs={execution.outputs} />
      )}

      {execution && execution.richOutputs && execution.richOutputs.length > 0 && (
        <NodeRichOutputs outputs={execution.richOutputs} />
      )}

      {execution && <NodeStatusFooter state={execution} />}
    </div>
  );
}

function NodePreviews({ outputs }: { outputs: Record<string, PreviewValue> }) {
  return (
    <div className="nodrag nowheel space-y-1.5 border-t border-zinc-800 px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-wide text-zinc-500">Preview</div>
      {Object.entries(outputs).map(([name, value]) => (
        <div key={name}>
          <div className="mb-0.5 font-mono text-[10px] text-cyan-300/80">{name}</div>
          <PreviewValueView value={value} compact />
        </div>
      ))}
    </div>
  );
}

function NodeRichOutputs({ outputs }: { outputs: RichOutput[] }) {
  const visible = outputs.slice(0, 2);
  const hidden = outputs.slice(2);
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="space-y-1.5 border-t border-zinc-800 px-2 py-1.5">
      <div className="flex items-center justify-between text-[9px] uppercase tracking-wide text-zinc-500">
        <span>Output</span>
        <span className="normal-case tracking-normal">
          {outputs.length} item{outputs.length === 1 ? "" : "s"}
        </span>
      </div>
      {visible.map((output, index) => (
        <RichOutputView key={index} output={output} compact />
      ))}
      {hidden.length > 0 && (
        <details
          className="nodrag"
          open={expanded}
          onToggle={(event) => setExpanded(event.currentTarget.open)}
        >
          <summary className="cursor-pointer text-[10px] text-cyan-400/80 hover:text-cyan-300">
            Show {hidden.length} more output{hidden.length === 1 ? "" : "s"}
          </summary>
          {expanded && (
            <div className="mt-1.5 space-y-1.5">
              {hidden.map((output, index) => (
                <RichOutputView key={index + visible.length} output={output} compact />
              ))}
            </div>
          )}
        </details>
      )}
    </div>
  );
}

function NodeStatusFooter({ state }: { state: NodeExecutionState }) {
  if (state.status === "running") {
    return (
      <div className="rounded-b-lg border-t border-zinc-800 bg-cyan-950/40 px-2 py-1 text-[10px] text-cyan-300">
        {state.installing && state.installing.length > 0
          ? `Installing: ${state.installing.join(", ")}`
          : "Running…"}
      </div>
    );
  }
  if (state.status === "completed") {
    const firstOutput = state.outputs ? Object.entries(state.outputs)[0] : undefined;
    return (
      <div className="flex items-center gap-1.5 rounded-b-lg border-t border-zinc-800 bg-emerald-950/40 px-2 py-1 text-[10px] text-emerald-300">
        <span className="font-semibold">{state.cached ? "cached" : "done"}</span>
        {firstOutput && (
          <span className="truncate text-emerald-200/80" title={JSON.stringify(firstOutput[1])}>
            {firstOutput[0]}: {summarisePreview(firstOutput[1])}
          </span>
        )}
      </div>
    );
  }
  if (state.status === "error" && state.error) {
    return (
      <div
        className="truncate rounded-b-lg border-t border-zinc-800 bg-red-950/50 px-2 py-1 text-[10px] text-red-300"
        title={state.error.traceback || state.error.message}
      >
        {state.error.errorType}
        {state.error.line !== null ? ` (line ${state.error.line})` : ""}: {state.error.message}
      </div>
    );
  }
  return null;
}
