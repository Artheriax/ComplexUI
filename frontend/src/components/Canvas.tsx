import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import type {
  FinalConnectionState,
  NodeMouseHandler,
  XYPosition,
} from "@xyflow/react";
import type {
  CSSProperties,
  DragEvent,
  MouseEvent as ReactMouseEvent,
} from "react";
import { useCallback, useMemo, useState } from "react";

import { useViewportShortcuts } from "../hooks/useViewportShortcuts";
import {
  buildDisplayEdges,
  CONNECTION_LINE_TYPE_BY_STYLE,
} from "../lib/edgePresentation";
import { useGraphStore } from "../store/useGraphStore";
import { HANDLE_SIZE_VALUES, useSettingsStore } from "../store/useSettingsStore";
import type { AppNode, PendingConnection } from "../types/graph";
import { NODE_COLORS } from "../types/graph";
import { ComplexNode } from "./ComplexNode";
import { NodeContextMenu } from "./NodeContextMenu";
import { NodeSearchPopup } from "./NodeSearchPopup";
import { RenameNodeDialog } from "./RenameNodeDialog";
import { EyeIcon, EyeOffIcon } from "./icons";

export const DRAG_MIME = "application/x-complexui-template";

const nodeTypes = { complex: ComplexNode };

interface SearchOverlay {
  anchor: { x: number; y: number };
  flowPosition: XYPosition;
  connection: PendingConnection | null;
}

interface MenuOverlay {
  nodeId: string;
  anchor: { x: number; y: number };
  flowPosition: XYPosition;
}

function pointerPosition(event: MouseEvent | TouchEvent): {
  x: number;
  y: number;
} {
  if ("changedTouches" in event) {
    const touch = event.changedTouches[0];
    if (touch) {
      return { x: touch.clientX, y: touch.clientY };
    }
  }
  return {
    x: (event as MouseEvent).clientX,
    y: (event as MouseEvent).clientY,
  };
}

function CanvasInner() {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const onNodesChange = useGraphStore((state) => state.onNodesChange);
  const onEdgesChange = useGraphStore((state) => state.onEdgesChange);
  const onConnect = useGraphStore((state) => state.onConnect);
  const openEditor = useGraphStore((state) => state.openEditor);
  const settings = useSettingsStore((state) => state.settings);
  const updateSetting = useSettingsStore((state) => state.updateSetting);
  const { screenToFlowPosition, zoomIn, zoomOut } = useReactFlow();
  const [searchOverlay, setSearchOverlay] = useState<SearchOverlay | null>(
    null,
  );
  const [menuOverlay, setMenuOverlay] = useState<MenuOverlay | null>(null);
  const [renameNodeId, setRenameNodeId] = useState<string | null>(null);
  const closeSearch = useCallback(() => setSearchOverlay(null), []);
  const closeMenu = useCallback(() => setMenuOverlay(null), []);
  const closeOverlays = useCallback(() => {
    setSearchOverlay(null);
    setMenuOverlay(null);
  }, []);
  const handleZoomIn = useCallback(() => {
    void zoomIn({ duration: 150 });
  }, [zoomIn]);
  const handleZoomOut = useCallback(() => {
    void zoomOut({ duration: 150 });
  }, [zoomOut]);

  useViewportShortcuts(handleZoomIn, handleZoomOut);

  const openSearchAt = useCallback(
    (point: { x: number; y: number }) => {
      setMenuOverlay(null);
      setSearchOverlay({
        anchor: point,
        flowPosition: screenToFlowPosition(point),
        connection: null,
      });
    },
    [screenToFlowPosition],
  );

  const handlePaneContextMenu = useCallback(
    (event: ReactMouseEvent | MouseEvent) => {
      event.preventDefault();
      openSearchAt({ x: event.clientX, y: event.clientY });
    },
    [openSearchAt],
  );

  const handleNodeContextMenu = useCallback<NodeMouseHandler<AppNode>>(
    (event, node) => {
      event.preventDefault();
      const graph = useGraphStore.getState();
      const target = graph.nodes.find((candidate) => candidate.id === node.id);
      if (!target?.selected) {
        graph.selectOnlyNode(node.id);
      }
      setSearchOverlay(null);
      const point = { x: event.clientX, y: event.clientY };
      setMenuOverlay({
        nodeId: node.id,
        anchor: point,
        flowPosition: screenToFlowPosition(point),
      });
    },
    [screenToFlowPosition],
  );

  const handleNodeClick = useCallback<NodeMouseHandler<AppNode>>((event, node) => {
    setSearchOverlay(null);
    setMenuOverlay(null);
    if (event.ctrlKey || event.metaKey) {
      return;
    }
    const graph = useGraphStore.getState();
    const selected = graph.nodes.filter((candidate) => candidate.selected);
    const target = graph.nodes.find((candidate) => candidate.id === node.id);
    if (selected.length > 1 || !target?.selected) {
      graph.selectOnlyNode(node.id);
    }
  }, []);

  const handleConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, connectionState: FinalConnectionState) => {
      if (!useSettingsStore.getState().settings.connectionDropSearch) {
        return;
      }
      if (
        connectionState.toHandle !== null ||
        connectionState.toNode !== null
      ) {
        return;
      }
      const fromHandle = connectionState.fromHandle;
      if (!fromHandle?.id) {
        return;
      }
      const graph = useGraphStore.getState();
      const sourceNode = graph.nodes.find(
        (candidate) => candidate.id === fromHandle.nodeId,
      );
      if (!sourceNode) {
        return;
      }
      const socket =
        fromHandle.type === "source"
          ? sourceNode.data.schema.outputs.find(
              (candidate) => candidate.name === fromHandle.id,
            )
          : sourceNode.data.schema.inputs.find(
              (candidate) => candidate.name === fromHandle.id,
            );
      if (!socket) {
        return;
      }
      const point = pointerPosition(event);
      const flowPosition = screenToFlowPosition(point);
      if (fromHandle.type === "target") {
        flowPosition.x -= 240;
      }
      setMenuOverlay(null);
      setSearchOverlay({
        anchor: point,
        flowPosition,
        connection: {
          nodeId: fromHandle.nodeId,
          handleId: fromHandle.id,
          handleType: fromHandle.type,
          dataType: socket.data_type,
        },
      });
    },
    [screenToFlowPosition],
  );

  const displayEdges = useMemo(
    () => buildDisplayEdges(edges, nodes, settings),
    [edges, nodes, settings],
  );
  const handleStyle = useMemo(() => {
    const size = HANDLE_SIZE_VALUES[settings.handleSize];
    return {
      "--cu-handle-hit-size": `${size.hit}px`,
      "--cu-handle-dot-size": `${size.dot}px`,
    } as CSSProperties;
  }, [settings.handleSize]);
  const backgroundVariant =
    settings.backgroundVariant === "lines"
      ? BackgroundVariant.Lines
      : settings.backgroundVariant === "cross"
        ? BackgroundVariant.Cross
        : BackgroundVariant.Dots;

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const templateName = event.dataTransfer.getData(DRAG_MIME);
      if (!templateName) {
        return;
      }
      const template = useGraphStore
        .getState()
        .templates.find((candidate) => candidate.name === templateName);
      if (!template) {
        return;
      }
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      useGraphStore.getState().addNodeFromTemplate(template, position);
    },
    [screenToFlowPosition],
  );

  return (
    <div
      className="relative h-full w-full"
      style={handleStyle}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <ReactFlow
        nodes={nodes}
        edges={displayEdges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onConnectEnd={handleConnectEnd}
        onClickConnectEnd={handleConnectEnd}
        onNodeDoubleClick={(_, node) => openEditor(node.id)}
        onNodeClick={handleNodeClick}
        onNodeContextMenu={handleNodeContextMenu}
        onPaneContextMenu={handlePaneContextMenu}
        onPaneClick={closeOverlays}
        onMoveStart={closeOverlays}
        onNodeDragStart={closeOverlays}
        selectionKeyCode={["Control", "Meta"]}
        multiSelectionKeyCode={["Control", "Meta"]}
        onBeforeDelete={async () => {
          useGraphStore.getState().checkpoint();
          return true;
        }}
        deleteKeyCode={["Backspace", "Delete"]}
        connectionRadius={32}
        connectionLineType={CONNECTION_LINE_TYPE_BY_STYLE[settings.edgeStyle]}
        connectionLineStyle={{
          stroke: settings.accentColor,
          strokeWidth: settings.edgeWidth,
        }}
        snapToGrid={settings.snapToGrid}
        snapGrid={[settings.gridSpacing, settings.gridSpacing]}
        fitView
        proOptions={{ hideAttribution: false }}
      >
        {settings.backgroundVariant !== "none" && (
          <Background
            variant={backgroundVariant}
            gap={settings.gridSpacing}
            size={1.5}
            color="#27272a"
          />
        )}
        {settings.showMiniMap && (
          <MiniMap
            pannable
            zoomable
            style={{ background: "#18181b" }}
            nodeColor={(node) =>
              NODE_COLORS[(node as AppNode).data.color]?.accent ?? "#3f3f46"
            }
            maskColor="rgba(9, 9, 11, 0.7)"
          />
        )}
        <Controls showInteractive={false} />
        <Panel position="top-right">
          <button
            type="button"
            onClick={() => updateSetting("showEdges", !settings.showEdges)}
            aria-pressed={settings.showEdges}
            aria-label={
              settings.showEdges
                ? "Hide connection lines"
                : "Show connection lines"
            }
            title={
              settings.showEdges
                ? "Hide connection lines"
                : "Show connection lines"
            }
            className="flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-900/90 px-2 py-1 text-[10px] text-zinc-300 shadow-md transition-colors hover:bg-zinc-800"
          >
            {settings.showEdges ? <EyeIcon /> : <EyeOffIcon />}
            <span>{settings.showEdges ? "Lines" : "Lines hidden"}</span>
          </button>
        </Panel>
      </ReactFlow>
      {nodes.length === 0 && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="rounded-lg border border-dashed border-zinc-700 bg-zinc-900/80 px-6 py-4 text-center">
            <p className="text-sm text-zinc-300">
              Drag a template from the Node Library — or add a scratch node
            </p>
            <p className="mt-1 text-xs text-zinc-500">
              Connect typed sockets, edit the Python code per node, then Run
              workflow.
            </p>
          </div>
        </div>
      )}
      {searchOverlay && (
        <NodeSearchPopup
          anchor={searchOverlay.anchor}
          flowPosition={searchOverlay.flowPosition}
          connection={searchOverlay.connection}
          onClose={closeSearch}
        />
      )}
      {menuOverlay && (
        <NodeContextMenu
          nodeId={menuOverlay.nodeId}
          anchor={menuOverlay.anchor}
          flowPosition={menuOverlay.flowPosition}
          onClose={closeMenu}
          onRename={(nodeId) => setRenameNodeId(nodeId)}
        />
      )}
      {renameNodeId && (
        <RenameNodeDialog
          nodeId={renameNodeId}
          onClose={() => setRenameNodeId(null)}
        />
      )}
    </div>
  );
}

export function Canvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
