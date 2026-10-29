import {
  Handle,
  NodeResizeControl,
  NodeResizer,
  Position,
  useConnection,
  useReactFlow,
} from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { useEffect, useRef, useState } from "react";
import type {
  CSSProperties,
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
} from "react";

import { classifyHandle, handleGuideClass } from "../lib/connectionGuide";
import { MAX_FRAME_DIMENSION } from "../lib/persistence";
import { deriveValueNodeOptions } from "../lib/valueNodeOptions";
import { isWidgetVisible } from "../lib/widgetVisibility";
import { useBundleDragStore } from "../store/useBundleDragStore";
import { useConnectionGuideStore } from "../store/useConnectionGuideStore";
import { useEdgeDragStore } from "../store/useEdgeDragStore";
import { useGraphStore } from "../store/useGraphStore";
import type {
  AppNode,
  NodeExecutionState,
  NodeSchema,
  PreviewValue,
  SocketType,
  SubgraphRole,
} from "../types/graph";
import type { RichOutput } from "../types/graph";
import {
  NEW_SUBGRAPH_INPUT_HANDLE,
  NEW_SUBGRAPH_OUTPUT_HANDLE,
  NODE_COLORS,
  SOCKET_COLORS,
  SOCKET_LABELS,
  summarisePreview,
  widgetSocketType,
} from "../types/graph";
import {
  DuplicateIcon,
  EnterIcon,
  PencilIcon,
  PlayIcon,
  PlusIcon,
  SubgraphIcon,
  TrashIcon,
  WarningIcon,
} from "./icons";
import { PreviewValueView } from "./PreviewValueView";
import { RichOutputView } from "./RichOutputView";
import { SaveMediaActions } from "./SaveMediaActions";
import { SaveMediaFields } from "./SaveMediaFields";
import { SwitchFields } from "./SwitchFields";
import { WidgetField } from "./WidgetField";

const SWITCH_WIDGETS = new Set([
  "input_1",
  "input_2",
  "input_3",
  "input_4",
  "input_5",
  "choices",
  "selected",
]);

const SAVE_MEDIA_WIDGETS = new Set([
  "folder",
  "filename",
  "model_name",
  "format",
  "quality",
  "compress_level",
  "optimize",
  "save_nonce",
]);

const statusBorder: Record<NodeExecutionState["status"] | "idle", string> = {
  idle: "border-zinc-700",
  running: "border-cyan-400 shadow-cyan-500/20",
  completed: "border-emerald-500/70",
  error: "border-red-500",
};

function socketStyle(type: SocketType): CSSProperties {
  return { "--cu-handle-color": SOCKET_COLORS[type] } as CSSProperties;
}

function startEdgeRetarget(
  event:
    | ReactMouseEvent<HTMLDivElement>
    | ReactPointerEvent<HTMLDivElement>,
  nodeId: string,
  handleId: string,
) {
  if (event.button !== 0) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  const graph = useGraphStore.getState();
  const edge = graph.edges.find(
    (candidate) =>
      candidate.target === nodeId && candidate.targetHandle === handleId,
  );
  const sourceHandleId = edge?.sourceHandle;
  if (!edge || !sourceHandleId) {
    return;
  }
  const sourceNode = graph.nodes.find((node) => node.id === edge.source);
  const socket = sourceNode?.data.schema.outputs.find(
    (candidate) => candidate.name === sourceHandleId,
  );
  const sourceElement = Array.from(
    document.querySelectorAll<HTMLElement>("[data-cu-output-handle='true']"),
  ).find(
    (candidate) =>
      candidate.dataset.nodeId === edge.source &&
      candidate.dataset.handleId === sourceHandleId,
  );
  if (!socket || !sourceElement) {
    return;
  }
  const rect = sourceElement.getBoundingClientRect();
  useEdgeDragStore.getState().press({
    edgeId: edge.id,
    sourceNodeId: edge.source,
    sourceHandleId,
    targetNodeId: nodeId,
    targetHandleId: handleId,
    dataType: socket.data_type,
    sourceScreen: {
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2,
    },
    start: { x: event.clientX, y: event.clientY },
  });
}

function SocketHandle({
  kind,
  nodeId,
  id,
  type,
  occupied = false,
  optional = false,
  bundleEdgeIds = [],
}: {
  kind: "source" | "target";
  nodeId: string;
  id: string;
  type: SocketType;
  occupied?: boolean;
  optional?: boolean;
  bundleEdgeIds?: string[];
}) {
  const guide = useConnectionGuideStore((state) => state.guide);
  const startBundle = useBundleDragStore((state) => state.start);
  const state = classifyHandle({
    guide,
    kind,
    dataType: type,
    nodeId,
    handleId: id,
    occupied,
  });
  const onMouseDownCapture = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (kind === "target" && occupied) {
      startEdgeRetarget(event, nodeId, id);
      return;
    }
    if (
      kind !== "source" ||
      !event.shiftKey ||
      bundleEdgeIds.length === 0
    ) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    useConnectionGuideStore.getState().begin({
      mode: "bundle",
      direction: "source",
      dataType: type,
      originNodeId: nodeId,
      originHandleId: id,
    });
    startBundle({
      edgeIds: bundleEdgeIds,
      sourceNodeId: nodeId,
      sourceHandleId: id,
      dataType: type,
      start: { x: event.clientX, y: event.clientY },
      current: { x: event.clientX, y: event.clientY },
    });
  };
  return (
    <Handle
      type={kind}
      position={kind === "source" ? Position.Right : Position.Left}
      id={id}
      style={socketStyle(type)}
      className={`nokey ${handleGuideClass(state)}${
        type === "IMAGE_BATCH" ? " cu-handle-batch" : ""
      }${optional && !occupied ? " cu-handle-optional" : ""}`}
      title={optional ? "Optional input — connect or leave empty" : undefined}
      onPointerDownCapture={
        kind === "target" && occupied
          ? (event) => startEdgeRetarget(event, nodeId, id)
          : undefined
      }
      onMouseDownCapture={onMouseDownCapture}
      onClickCapture={
        kind === "target" && occupied
          ? (event) => {
              event.preventDefault();
              event.stopPropagation();
            }
          : undefined
      }
      data-cu-handle={kind}
      data-cu-output-handle={kind === "source" ? "true" : undefined}
      data-node-id={nodeId}
      data-handle-id={id}
      data-socket-type={type}
      data-socket-optional={optional ? "true" : undefined}
      data-bundle-count={bundleEdgeIds.length}
      data-node-editor-ignore="true"
    />
  );
}

const WIDGET_HANDLE_NEAR_PX = 56;
const WIDGET_HANDLE_RELEASE_PX = 72;

function WidgetTargetHandle({
  nodeId,
  id,
  type,
  connected,
}: {
  nodeId: string;
  id: string;
  type: SocketType;
  connected: boolean;
}) {
  const handleRef = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);
  const connectionEnd = useConnection((connection) =>
    connection.inProgress ? connection.to : null,
  );
  const { flowToScreenPosition } = useReactFlow();
  const guide = useConnectionGuideStore((state) => state.guide);
  const guideState = classifyHandle({
    guide,
    kind: "target",
    dataType: type,
    nodeId,
    handleId: id,
    occupied: connected,
  });

  useEffect(() => {
    const handle = handleRef.current;
    if (!connectionEnd || !handle) {
      setNear(false);
      return;
    }
    const rect = handle.getBoundingClientRect();
    const point = flowToScreenPosition(connectionEnd);
    const distance = Math.hypot(
      point.x - (rect.left + rect.width / 2),
      point.y - (rect.top + rect.height / 2),
    );
    setNear((current) =>
      current
        ? distance <= WIDGET_HANDLE_RELEASE_PX
        : distance <= WIDGET_HANDLE_NEAR_PX,
    );
  }, [connectionEnd, flowToScreenPosition]);

  return (
    <Handle
      ref={handleRef}
      type="target"
      position={Position.Left}
      id={id}
      style={socketStyle(type)}
      className={`cu-widget-handle${connected ? " is-connected" : ""}${
        near ? " is-near" : ""
      } nokey ${handleGuideClass(guideState)}`}
      onPointerDownCapture={
        connected
          ? (event) => startEdgeRetarget(event, nodeId, id)
          : undefined
      }
      onMouseDownCapture={(event) => {
        if (connected) {
          startEdgeRetarget(event, nodeId, id);
        }
      }}
      onClickCapture={
        connected
          ? (event) => {
              event.preventDefault();
              event.stopPropagation();
            }
          : undefined
      }
      data-cu-handle="target"
      data-node-id={nodeId}
      data-handle-id={id}
      data-socket-type={type}
      data-node-editor-ignore="true"
    />
  );
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

export function ComplexNode({ id, data, selected, width, height }: NodeProps<AppNode>) {
  const execution = useGraphStore((state) => state.execution.nodeStates[id]);
  const running = useGraphStore((state) => state.execution.status === "running");
  const openEditor = useGraphStore((state) => state.openEditor);
  const deleteNode = useGraphStore((state) => state.deleteNode);
  const duplicateNode = useGraphStore((state) => state.duplicateNode);
  const runGraph = useGraphStore((state) => state.runGraph);
  const updateWidgetValue = useGraphStore((state) => state.updateWidgetValue);
  const checkpoint = useGraphStore((state) => state.checkpoint);
  const enterSubgraph = useGraphStore((state) => state.enterSubgraph);
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const valueOptions = deriveValueNodeOptions(id, data, nodes, edges);

  const connectedTargets = new Set(
    edges
      .filter((edge) => edge.target === id)
      .map((edge) => edge.targetHandle)
      .filter((handle): handle is string => Boolean(handle)),
  );
  const stepValue =
    typeof data.widgetValues.step === "number"
      ? data.widgetValues.step
      : undefined;

  const status = execution?.status ?? "idle";
  const borderClass = statusBorder[status];
  const accent = NODE_COLORS[data.color]?.accent ?? NODE_COLORS.default.accent;
  const surface =
    NODE_COLORS[data.color]?.surface ?? NODE_COLORS.default.surface;
  const isSaveMedia = [...SAVE_MEDIA_WIDGETS].every((name) =>
    data.schema.widgets.some((widget) => widget.name === name),
  );
  const isSwitch = [...SWITCH_WIDGETS].every((name) =>
    data.schema.widgets.some((widget) => widget.name === name),
  );
  const customWidgets = isSaveMedia
    ? SAVE_MEDIA_WIDGETS
    : isSwitch
      ? SWITCH_WIDGETS
      : null;
  const visibleWidgets = (
    customWidgets
      ? data.schema.widgets.filter(
          (widget) => !customWidgets.has(widget.name),
        )
      : data.schema.widgets
  ).filter((widget) =>
    isWidgetVisible(widget, data.widgetValues, connectedTargets),
  );
  const hasPromptWidget = visibleWidgets.some(
    (widget) => widget.data_type.toLowerCase() === "prompt",
  );
  const mediaEdge = edges.find(
    (edge) => edge.target === id && edge.targetHandle === "media",
  );
  const upstreamPreview = useGraphStore((state) => {
    if (!mediaEdge) {
      return null;
    }
    const source = state.execution.nodeStates[mediaEdge.source];
    return source?.outputs?.[mediaEdge.sourceHandle ?? ""] ?? null;
  });

  if (data.isFrame) {
    return <FrameShell id={id} label={data.label} accent={accent} selected={selected} />;
  }

  if (data.subgraphRole) {
    return (
      <SubgraphManagerShell
        id={id}
        role={data.subgraphRole}
        label={data.label}
        schema={data.schema}
        accent={accent}
        surface={surface}
        selected={selected}
      />
    );
  }

  if (data.isSubgraph) {
    return (
      <SubgraphInstanceShell
        id={id}
        width={width}
        height={height}
        label={data.label}
        schema={data.schema}
        accent={accent}
        surface={surface}
        selected={selected}
        onOpen={() => {
          enterSubgraph(id);
        }}
      />
    );
  }

  return (
    <div
      data-node-surface={data.color}
      className={`flex min-w-[230px] flex-col rounded-lg border shadow-xl ${
        width === undefined
          ? isSaveMedia || isSwitch
            ? "w-[280px]"
            : "max-w-[280px]"
          : ""
      } ${borderClass} ${selected ? "ring-2 ring-cyan-400/60" : ""}`}
      style={{
        width: width === undefined ? undefined : "100%",
        height: height === undefined ? undefined : "100%",
        backgroundColor: surface,
      }}
    >
      {selected && (
        <NodeResizeControl
          position="bottom-right"
          minWidth={230}
          minHeight={hasPromptWidget ? 240 : 160}
          maxWidth={MAX_FRAME_DIMENSION}
          maxHeight={MAX_FRAME_DIMENSION}
          color={accent}
          autoScale
          onResizeStart={() => checkpoint()}
          className="nodrag"
        />
      )}
      <div className="relative flex shrink-0 items-center gap-1.5 rounded-t-lg border-b border-zinc-800 bg-zinc-900/95 px-2 py-1.5">
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
        <div className="shrink-0 border-b border-zinc-800 py-0.5">
          {data.schema.inputs.map((socket) => (
            <div key={socket.name} className="relative flex items-center gap-2 py-1 pl-3 pr-2">
              <SocketHandle
                kind="target"
                nodeId={id}
                id={socket.name}
                type={socket.data_type}
                occupied={connectedTargets.has(socket.name)}
                optional={socket.optional === true}
              />
              <span
                className="text-xs text-zinc-300"
                title={
                  socket.optional
                    ? "Optional input — connect or leave empty"
                    : undefined
                }
              >
                {socket.name}
              </span>
              {socket.optional && (
                <span className="rounded bg-zinc-800/70 px-1 py-px text-[8px] uppercase tracking-wide text-zinc-500">
                  optional
                </span>
              )}
              <span className="ml-auto">
                <SocketChip type={socket.data_type} />
              </span>
            </div>
          ))}
        </div>
      )}

      {(visibleWidgets.length > 0 || isSaveMedia || isSwitch) && (
        <div
          className={`space-y-1 border-b border-zinc-800 py-1.5 pr-2 ${
            hasPromptWidget
              ? "flex min-h-0 flex-1 flex-col"
              : "shrink-0"
          }`}
        >
          {isSwitch && (
            <SwitchFields
              widgets={data.schema.widgets.filter((widget) =>
                SWITCH_WIDGETS.has(widget.name),
              )}
              values={data.widgetValues}
              connected={connectedTargets}
              onChange={(name, value) => updateWidgetValue(id, name, value)}
              renderHandle={(name, widget, connected) => (
                <WidgetTargetHandle
                  nodeId={id}
                  id={name}
                  type={widgetSocketType(widget.data_type)}
                  connected={connected}
                />
              )}
            />
          )}
          {isSaveMedia && (
            <SaveMediaFields
              widgets={data.schema.widgets.filter((widget) =>
                SAVE_MEDIA_WIDGETS.has(widget.name),
              )}
              values={data.widgetValues}
              connected={connectedTargets}
              onChange={(name, value) => updateWidgetValue(id, name, value)}
              renderHandle={(name, widget, connected) => (
                <WidgetTargetHandle
                  nodeId={id}
                  id={name}
                  type={widgetSocketType(widget.data_type)}
                  connected={connected}
                />
              )}
              upstreamPreview={upstreamPreview}
            />
          )}
          {visibleWidgets.map((widget) => {
            const effectiveWidget =
              widget.name === "value" && valueOptions !== null
                ? { ...widget, options: valueOptions }
                : widget;
            const isPromptWidget =
              widget.data_type.toLowerCase() === "prompt";
            return (
              <div
                key={widget.name}
                className={`cu-widget-row relative pl-3 ${
                  isPromptWidget
                    ? "flex min-h-0 flex-1 flex-col"
                    : "shrink-0"
                }`}
                data-prompt-row={isPromptWidget ? "true" : undefined}
                onDoubleClick={(event) => event.stopPropagation()}
              >
                <WidgetTargetHandle
                  nodeId={id}
                  id={widget.name}
                  type={widgetSocketType(widget.data_type)}
                  connected={connectedTargets.has(widget.name)}
                />
                <WidgetField
                  widget={effectiveWidget}
                  value={data.widgetValues[widget.name]}
                  step={widget.name === "value" ? stepValue : undefined}
                  connected={connectedTargets.has(widget.name)}
                  onChange={(value) =>
                    updateWidgetValue(id, widget.name, value)
                  }
                />
              </div>
            );
          })}
        </div>
      )}

      {data.schema.outputs.length > 0 && (
        <div className="shrink-0 py-0.5">
          {data.schema.outputs.map((socket) => {
            const bundleEdgeIds = edges
              .filter(
                (edge) =>
                  edge.source === id && edge.sourceHandle === socket.name,
              )
              .map((edge) => edge.id);
            return (
              <div
                key={socket.name}
                className="relative flex items-center gap-2 py-1 pl-2 pr-3"
              >
                <span className="ml-auto text-xs text-zinc-300">
                  {socket.name}
                </span>
                <SocketChip type={socket.data_type} />
                <SocketHandle
                  kind="source"
                  nodeId={id}
                  id={socket.name}
                  type={socket.data_type}
                  bundleEdgeIds={bundleEdgeIds}
                />
              </div>
            );
          })}
        </div>
      )}

      {execution?.outputs && Object.keys(execution.outputs).length > 0 && (
        <NodePreviews
          outputs={
            isSaveMedia
              ? Object.fromEntries(
                  Object.entries(execution.outputs).filter(
                    ([name]) => name !== "saved_path",
                  ),
                )
              : execution.outputs
          }
          fill={height !== undefined}
        />
      )}

      {execution && execution.richOutputs && execution.richOutputs.length > 0 && (
        <NodeRichOutputs outputs={execution.richOutputs} />
      )}

      {isSaveMedia && (
        <SaveMediaActions
          nodeId={id}
          values={data.widgetValues}
          execution={execution}
          running={running}
          isTrusted={data.isTrusted}
          hasMediaInput={connectedTargets.has("media")}
        />
      )}

      {execution && <NodeStatusFooter state={execution} />}
    </div>
  );
}

function PortRows({
  nodeId,
  sockets,
  direction,
}: {
  nodeId: string;
  sockets: NodeSchema["inputs"] | NodeSchema["outputs"];
  direction: "input" | "output";
}) {
  return (
    <div className="py-0.5">
      {sockets.map((socket) => {
        const optional = "optional" in socket && socket.optional === true;
        return (
          <div
            key={socket.name}
            className={`relative flex items-center gap-2 py-1 ${
              direction === "input" ? "pl-3 pr-2" : "pl-2 pr-3"
            }`}
          >
            {direction === "input" && (
              <SocketHandle
                kind="target"
                nodeId={nodeId}
                id={socket.name}
                type={socket.data_type}
                optional={optional}
              />
            )}
            {direction === "output" && (
              <span className="ml-auto text-xs text-zinc-300">
                {socket.name}
              </span>
            )}
            {direction === "input" && (
              <span
                className="text-xs text-zinc-300"
                title={
                  optional
                    ? "Optional input — connect or leave empty"
                    : undefined
                }
              >
                {socket.name}
              </span>
            )}
            {direction === "input" && optional && (
              <span className="rounded bg-zinc-800/70 px-1 py-px text-[8px] uppercase tracking-wide text-zinc-500">
                optional
              </span>
            )}
            <span className={direction === "input" ? "ml-auto" : ""}>
              <SocketChip type={socket.data_type} />
            </span>
            {direction === "output" && (
              <SocketHandle
                kind="source"
                nodeId={nodeId}
                id={socket.name}
                type={socket.data_type}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

function SubgraphManagerShell({
  id,
  role,
  label,
  schema,
  accent,
  surface,
  selected,
}: {
  id: string;
  role: SubgraphRole;
  label: string;
  schema: NodeSchema;
  accent: string;
  surface: string;
  selected: boolean;
}) {
  const addSubgraphPort = useGraphStore((state) => state.addSubgraphPort);

  return (
    <div
      className={`min-w-[190px] rounded-lg border shadow-xl ${
        selected ? "ring-2 ring-cyan-400/60" : ""
      }`}
      style={{ borderColor: accent, backgroundColor: surface }}
    >
      <div className="relative flex shrink-0 items-center gap-1.5 rounded-t-lg border-b border-zinc-800 bg-zinc-900/95 px-2 py-1.5">
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 h-0.5 rounded-t-lg"
          style={{ backgroundColor: accent }}
        />
        <SubgraphIcon className="h-3.5 w-3.5 text-cyan-300" />
        <span className="truncate text-xs font-semibold text-zinc-100">
          {label}
        </span>
        <span className="ml-auto rounded bg-zinc-800 px-1 py-px text-[9px] uppercase text-zinc-500">
          shared
        </span>
      </div>
      <PortRows nodeId={id} sockets={schema.inputs} direction="input" />
      <PortRows nodeId={id} sockets={schema.outputs} direction="output" />
      <div className="relative flex items-center gap-1.5 border-t border-dashed border-zinc-700 px-2 py-1.5 text-[10px] text-zinc-500">
        {role === "input" ? (
          <>
            <button
              type="button"
              className="nodrag flex items-center gap-1 rounded px-1 py-0.5 text-cyan-300 hover:bg-zinc-800"
              title="Add an ANY-typed input port; drag from the dashed handle to infer a type"
              aria-label="Add input port"
              onClick={() => {
                addSubgraphPort("input", "ANY");
              }}
            >
              <PlusIcon className="h-3 w-3" />
              <span>Add input</span>
            </button>
            <span className="ml-auto text-[9px] text-zinc-600">
              or drag + to a node input
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={NEW_SUBGRAPH_INPUT_HANDLE}
              style={socketStyle("ANY")}
            />
          </>
        ) : (
          <>
            <Handle
              type="target"
              position={Position.Left}
              id={NEW_SUBGRAPH_OUTPUT_HANDLE}
              style={socketStyle("ANY")}
            />
            <button
              type="button"
              className="nodrag flex items-center gap-1 rounded px-1 py-0.5 text-cyan-300 hover:bg-zinc-800"
              title="Add an ANY-typed output port; drop a wire on the dashed handle to infer a type"
              aria-label="Add output port"
              onClick={() => {
                addSubgraphPort("output", "ANY");
              }}
            >
              <PlusIcon className="h-3 w-3" />
              <span>Add output</span>
            </button>
            <span className="ml-auto text-[9px] text-zinc-600">
              or drop a node output on +
            </span>
          </>
        )}
      </div>
    </div>
  );
}

function SubgraphInstanceShell({
  id,
  width,
  height,
  label,
  schema,
  accent,
  surface,
  selected,
  onOpen,
}: {
  id: string;
  width: number | undefined;
  height: number | undefined;
  label: string;
  schema: NodeSchema;
  accent: string;
  surface: string;
  selected: boolean;
  onOpen: () => void;
}) {
  const duplicateNode = useGraphStore((state) => state.duplicateNode);
  const deleteNode = useGraphStore((state) => state.deleteNode);
  const checkpoint = useGraphStore((state) => state.checkpoint);
  const nodeCount = useGraphStore((state) => {
    const definitionId = state.nodes.find((node) => node.id === id)?.data
      .subgraphId;
    return definitionId
      ? (state.subgraphs[definitionId]?.nodes.length ?? 0)
      : 0;
  });

  return (
    <div
      className={`flex min-w-[230px] flex-col rounded-lg border shadow-xl ${
        width === undefined ? "max-w-[280px]" : ""
      } ${selected ? "ring-2 ring-cyan-400/60" : ""}`}
      style={{
        borderColor: accent,
        backgroundColor: surface,
        width: width === undefined ? undefined : "100%",
        height: height === undefined ? undefined : "100%",
      }}
    >
      {selected && (
        <NodeResizeControl
          position="bottom-right"
          minWidth={230}
          minHeight={160}
          maxWidth={MAX_FRAME_DIMENSION}
          maxHeight={MAX_FRAME_DIMENSION}
          color={accent}
          autoScale
          onResizeStart={() => checkpoint()}
          className="nodrag"
        />
      )}
      <div className="relative flex shrink-0 items-center gap-1.5 rounded-t-lg border-b border-zinc-800 bg-zinc-900/95 px-2 py-1.5">
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 h-0.5 rounded-t-lg"
          style={{ backgroundColor: accent }}
        />
        <SubgraphIcon className="h-3.5 w-3.5 text-cyan-300" />
        <span className="truncate text-xs font-semibold text-zinc-100" title={label}>
          {label}
        </span>
        <span className="rounded bg-cyan-950 px-1 py-px text-[9px] uppercase tracking-wide text-cyan-300">
          subgraph
        </span>
        <span className="ml-auto flex items-center gap-1">
          <button
            type="button"
            className="nodrag rounded p-1 text-cyan-300/90 hover:bg-zinc-700 hover:text-cyan-200"
            title="Open subgraph"
            aria-label={`Open subgraph ${label}`}
            onClick={onOpen}
          >
            <EnterIcon className="h-3 w-3" />
          </button>
          <button
            type="button"
            className="nodrag rounded p-1 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-100"
            title="Duplicate subgraph instance"
            aria-label={`Duplicate ${label}`}
            onClick={() => {
              void duplicateNode(id);
            }}
          >
            <DuplicateIcon />
          </button>
          <button
            type="button"
            className="nodrag rounded p-1 text-zinc-400 hover:bg-red-900/60 hover:text-red-300"
            title="Delete subgraph instance"
            aria-label={`Delete ${label}`}
            onClick={() => deleteNode(id)}
          >
            <TrashIcon />
          </button>
        </span>
      </div>
      <PortRows nodeId={id} sockets={schema.inputs} direction="input" />
      <PortRows nodeId={id} sockets={schema.outputs} direction="output" />
      <div className="rounded-b-lg border-t border-zinc-800 px-2 py-1 text-[9px] text-zinc-600">
        Double-click to open · {nodeCount} internal node
        {nodeCount === 1 ? "" : "s"}
      </div>
    </div>
  );
}

function FrameShell({
  id,
  label,
  accent,
  selected,
}: {
  id: string;
  label: string;
  accent: string;
  selected: boolean;
}) {
  const checkpoint = useGraphStore((state) => state.checkpoint);
  const childCount = useGraphStore(
    (state) => state.nodes.filter((node) => node.parentId === id).length,
  );

  return (
    <div
      className={`flex h-full w-full flex-col rounded-xl border-2 border-dashed transition-colors ${
        selected ? "ring-2 ring-cyan-400/60" : ""
      }`}
      style={{
        borderColor: selected ? accent : `${accent}88`,
        backgroundColor: `${accent}0d`,
      }}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={280}
        minHeight={180}
        maxWidth={MAX_FRAME_DIMENSION}
        maxHeight={MAX_FRAME_DIMENSION}
        color={accent}
        onResizeStart={() => checkpoint()}
      />
      <div
        className="flex h-10 shrink-0 items-center gap-2 px-3"
        style={{ backgroundColor: `${accent}14` }}
      >
        <span
          aria-hidden="true"
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: accent }}
        />
        <span className="truncate text-xs font-semibold text-zinc-200" title={label}>
          {label}
        </span>
        <span className="ml-auto shrink-0 text-[9px] uppercase tracking-wide text-zinc-500">
          {childCount} node{childCount === 1 ? "" : "s"}
        </span>
      </div>
      <div className="min-h-0 flex-1" />
    </div>
  );
}

function NodePreviews({
  outputs,
  fill,
}: {
  outputs: Record<string, PreviewValue>;
  fill: boolean;
}) {
  return (
    <div
      className={`nodrag nowheel space-y-1.5 border-t border-zinc-800 px-2 py-1.5 ${
        fill ? "flex min-h-0 flex-1 flex-col overflow-hidden" : ""
      }`}
    >
      <div className="shrink-0 text-[9px] uppercase tracking-wide text-zinc-500">
        Preview
      </div>
      {Object.entries(outputs).map(([name, value]) => (
        <div key={name} className={fill ? "flex min-h-0 flex-1 flex-col" : ""}>
          <div className="mb-0.5 shrink-0 font-mono text-[10px] text-cyan-300/80">
            {name}
          </div>
          <PreviewValueView value={value} compact fill={fill} />
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
    const progress =
      state.progress === undefined
        ? null
        : Math.round(Math.min(1, Math.max(0, state.progress)) * 100);
    return (
      <div className="rounded-b-lg border-t border-zinc-800 bg-cyan-950/40 px-2 py-1 text-[10px] text-cyan-300">
        <div className="flex items-center gap-2">
          <span className="min-w-0 truncate">
            {state.installing && state.installing.length > 0
              ? `Installing: ${state.installing.join(", ")}`
              : state.progressMessage || "Running…"}
          </span>
          {progress !== null && (
            <span className="ml-auto shrink-0 font-mono text-[9px] text-cyan-200/80">
              {progress}%
            </span>
          )}
        </div>
        {progress !== null && (
          <div
            role="progressbar"
            aria-label="Node progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress}
            className="mt-1 h-1 overflow-hidden rounded-full bg-zinc-800/80"
          >
            <div
              className="h-full rounded-full bg-cyan-400 transition-[width] duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
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
