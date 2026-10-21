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
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useViewportShortcuts } from "../hooks/useViewportShortcuts";
import { useWorkflowFileActions } from "../hooks/useWorkflowFileActions";
import {
  buildDisplayEdges,
  CONNECTION_LINE_TYPE_BY_STYLE,
} from "../lib/edgePresentation";
import { resolveTargetSocket } from "../lib/socketCompatibility";
import { useGraphStore } from "../store/useGraphStore";
import { HANDLE_SIZE_VALUES, useSettingsStore } from "../store/useSettingsStore";
import type { AppNode, PendingConnection } from "../types/graph";
import {
  isFrameNode,
  isSubgraphInstanceNode,
  isSubgraphManagerNode,
  NEW_SUBGRAPH_INPUT_HANDLE,
  NEW_SUBGRAPH_OUTPUT_HANDLE,
  NODE_COLORS,
} from "../types/graph";
import { CanvasContextMenu } from "./CanvasContextMenu";
import { CanvasStatusBar } from "./CanvasStatusBar";
import { ComplexNode } from "./ComplexNode";
import { NodeContextMenu } from "./NodeContextMenu";
import { NodeSearchPopup } from "./NodeSearchPopup";
import { RenameNodeDialog } from "./RenameNodeDialog";
import { SelectionToolbar } from "./SelectionToolbar";
import { ArrowLeftIcon, SubgraphIcon } from "./icons";

export const DRAG_MIME = "application/x-complexui-template";
export const DRAG_SUBGRAPH_MIME = "application/x-complexui-subgraph";

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

interface CanvasMenuOverlay {
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
  const enterSubgraph = useGraphStore((state) => state.enterSubgraph);
  const navigateToScope = useGraphStore((state) => state.navigateToScope);
  const scopeStack = useGraphStore((state) => state.scopeStack);
  const settings = useSettingsStore((state) => state.settings);
  const { fileInputRef, onFileInputChange, openImport } =
    useWorkflowFileActions();
  const { screenToFlowPosition, zoomIn, zoomOut, fitView } = useReactFlow();
  const fitScopeRef = useRef<string | null>(null);
  const [searchOverlay, setSearchOverlay] = useState<SearchOverlay | null>(
    null,
  );
  const [menuOverlay, setMenuOverlay] = useState<MenuOverlay | null>(null);
  const [canvasMenu, setCanvasMenu] = useState<CanvasMenuOverlay | null>(null);
  const [renameNodeId, setRenameNodeId] = useState<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const closeMenu = useCallback(() => setMenuOverlay(null), []);
  const closeCanvasMenu = useCallback(() => setCanvasMenu(null), []);
  const closeOverlays = useCallback(() => {
    setSearchOverlay(null);
    setMenuOverlay(null);
    setCanvasMenu(null);
  }, []);
  const handleZoomIn = useCallback(() => {
    void zoomIn({ duration: 150 });
  }, [zoomIn]);
  const handleZoomOut = useCallback(() => {
    void zoomOut({ duration: 150 });
  }, [zoomOut]);
  const handleFitView = useCallback(() => {
    void fitView({ padding: 0.2, duration: 200 });
  }, [fitView]);

  useViewportShortcuts(handleZoomIn, handleZoomOut);

  const scopeKey =
    scopeStack.length === 0
      ? "root"
      : scopeStack.map((scope) => scope.instanceId).join("/");
  const allNodesMeasured =
    nodes.length > 0 &&
    nodes.every(
      (node) => node.measured?.width && node.measured?.height,
    );

  useEffect(() => {
    if (!allNodesMeasured || fitScopeRef.current === scopeKey) {
      return;
    }
    fitScopeRef.current = scopeKey;
    const timer = setTimeout(() => {
      void fitView({ padding: 0.2, duration: 200 });
    }, 50);
    return () => clearTimeout(timer);
  }, [allNodesMeasured, fitView, scopeKey]);

  const openSearchAt = useCallback(
    (point: { x: number; y: number }) => {
      const flowPosition = screenToFlowPosition(point);
      setMenuOverlay(null);
      setCanvasMenu(null);
      setSearchOverlay({
        anchor: point,
        flowPosition,
        connection: null,
      });
    },
    [screenToFlowPosition],
  );

  const handlePaneContextMenu = useCallback(
    (event: ReactMouseEvent | MouseEvent) => {
      event.preventDefault();
      const point = { x: event.clientX, y: event.clientY };
      setSearchOverlay(null);
      setMenuOverlay(null);
      setCanvasMenu({
        anchor: point,
        flowPosition: screenToFlowPosition(point),
      });
    },
    [screenToFlowPosition],
  );

  const handleNodeContextMenu = useCallback<NodeMouseHandler<AppNode>>(
    (event, node) => {
      event.preventDefault();
      if (isSubgraphManagerNode(node)) {
        closeOverlays();
        return;
      }
      const graph = useGraphStore.getState();
      const target = graph.nodes.find((candidate) => candidate.id === node.id);
      if (!target?.selected) {
        graph.selectOnlyNode(node.id);
      }
      setSearchOverlay(null);
      setCanvasMenu(null);
      const point = { x: event.clientX, y: event.clientY };
      setMenuOverlay({
        nodeId: node.id,
        anchor: point,
        flowPosition: screenToFlowPosition(point),
      });
    },
    [closeOverlays, screenToFlowPosition],
  );

  const handleNodeClick = useCallback<NodeMouseHandler<AppNode>>((event, node) => {
    setSearchOverlay(null);
    setMenuOverlay(null);
    setCanvasMenu(null);
    if (event.ctrlKey || event.metaKey || event.shiftKey) {
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
      const droppedNode = connectionState.toNode
        ? graph.nodes.find(
            (candidate) => candidate.id === connectionState.toNode?.id,
          )
        : undefined;
      if (droppedNode && !connectionState.toHandle) {
        if (droppedNode.data.subgraphRole === "input") {
          if (fromHandle.type !== "target") {
            return;
          }
          const socket = sourceNode.data.schema.inputs.find(
            (candidate) => candidate.name === fromHandle.id,
          );
          if (!socket) {
            return;
          }
          const port = graph.addSubgraphPort(
            "input",
            socket.data_type,
            fromHandle.id,
          );
          if (!port) {
            return;
          }
          graph.onConnect({
            source: droppedNode.id,
            sourceHandle: port,
            target: fromHandle.nodeId,
            targetHandle: fromHandle.id,
          });
          closeOverlays();
          return;
        }
        if (droppedNode.data.subgraphRole === "output") {
          if (fromHandle.type !== "source") {
            return;
          }
          const socket = sourceNode.data.schema.outputs.find(
            (candidate) => candidate.name === fromHandle.id,
          );
          if (!socket) {
            return;
          }
          const port = graph.addSubgraphPort(
            "output",
            socket.data_type,
            fromHandle.id,
          );
          if (!port) {
            return;
          }
          graph.onConnect({
            source: fromHandle.nodeId,
            sourceHandle: fromHandle.id,
            target: droppedNode.id,
            targetHandle: port,
          });
          closeOverlays();
          return;
        }
      }
      if (!useSettingsStore.getState().settings.connectionDropSearch) {
        return;
      }
      if (
        connectionState.toHandle !== null ||
        connectionState.toNode !== null
      ) {
        return;
      }
      const point = pointerPosition(event);
      const flowPosition = screenToFlowPosition(point);
      if (fromHandle.type === "target") {
        flowPosition.x -= 240;
      }
      const newManagerHandle =
        (sourceNode.data.subgraphRole === "input" &&
          fromHandle.type === "source" &&
          fromHandle.id === NEW_SUBGRAPH_INPUT_HANDLE) ||
        (sourceNode.data.subgraphRole === "output" &&
          fromHandle.type === "target" &&
          fromHandle.id === NEW_SUBGRAPH_OUTPUT_HANDLE);
      if (newManagerHandle) {
        setMenuOverlay(null);
        setCanvasMenu(null);
        setSearchOverlay({
          anchor: point,
          flowPosition,
          connection: {
            nodeId: fromHandle.nodeId,
            handleId: fromHandle.id,
            handleType: fromHandle.type,
            dataType: "ANY",
          },
        });
        return;
      }
      const socket =
        fromHandle.type === "source"
          ? sourceNode.data.schema.outputs.find(
              (candidate) => candidate.name === fromHandle.id,
            )
          : resolveTargetSocket(sourceNode.data.schema, fromHandle.id);
      if (!socket) {
        return;
      }
      setMenuOverlay(null);
      setCanvasMenu(null);
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
    [closeOverlays, screenToFlowPosition],
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
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      const subgraphId = event.dataTransfer.getData(DRAG_SUBGRAPH_MIME);
      if (subgraphId) {
        useGraphStore.getState().addSubgraphInstance(subgraphId, position);
        return;
      }
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
      useGraphStore.getState().addNodeFromTemplate(template, position);
    },
    [screenToFlowPosition],
  );

  const handleCanvasDoubleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      if (!target.closest(".react-flow__pane")) {
        return;
      }
      if (
        target.closest(
          ".react-flow__node, .react-flow__controls, .react-flow__minimap, .react-flow__panel",
        )
      ) {
        return;
      }
      event.preventDefault();
      openSearchAt({ x: event.clientX, y: event.clientY });
    },
    [openSearchAt],
  );

  return (
    <div
      className="relative h-full w-full"
      style={handleStyle}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onDoubleClick={handleCanvasDoubleClick}
    >
      <ReactFlow
        key={scopeKey}
        nodes={nodes}
        edges={displayEdges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onConnectEnd={handleConnectEnd}
        onClickConnectEnd={handleConnectEnd}
        onNodeDoubleClick={(_, node) => {
          if (isSubgraphManagerNode(node)) {
            return;
          }
          if (isSubgraphInstanceNode(node)) {
            enterSubgraph(node.id);
            return;
          }
          if (node.data.isFrame) {
            setRenameNodeId(node.id);
            return;
          }
          openEditor(node.id);
        }}
        onNodeClick={handleNodeClick}
        onNodeContextMenu={handleNodeContextMenu}
        onPaneContextMenu={handlePaneContextMenu}
        onPaneClick={closeOverlays}
        onNodeDragStart={closeOverlays}
        selectionKeyCode={["Control", "Meta", "Shift"]}
        multiSelectionKeyCode={["Control", "Meta", "Shift"]}
        onBeforeDelete={async ({ nodes: deletingNodes }) => {
          const graph = useGraphStore.getState();
          if (deletingNodes.some(isSubgraphManagerNode)) {
            graph.deleteSelection();
            return false;
          }
          if (deletingNodes.some(isFrameNode)) {
            graph.deleteSelection();
            return false;
          }
          graph.checkpoint();
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
        zoomOnDoubleClick={false}
        minZoom={0.2}
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
        {scopeStack.length > 0 && (
          <Panel position="top-left">
            <nav
              aria-label="Canvas hierarchy"
              className="flex max-w-[min(70vw,42rem)] items-center gap-1 rounded-lg border border-zinc-700 bg-zinc-900/95 px-2 py-1 text-[11px] shadow-xl backdrop-blur"
            >
              <button
                type="button"
                title="Back to the root canvas"
                onClick={() => navigateToScope(-1)}
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-zinc-300 hover:bg-zinc-800 hover:text-cyan-200"
              >
                <ArrowLeftIcon className="h-3 w-3" />
                Root
              </button>
              {scopeStack.map((scope, index) => {
                const current = index === scopeStack.length - 1;
                return (
                  <span key={`${scope.definitionId}-${index}`} className="flex items-center gap-1">
                    <span className="text-zinc-600">/</span>
                    <button
                      type="button"
                      disabled={current}
                      onClick={() => navigateToScope(index)}
                      className={`flex max-w-44 items-center gap-1 truncate rounded px-1.5 py-0.5 ${
                        current
                          ? "text-cyan-200"
                          : "text-zinc-300 hover:bg-zinc-800 hover:text-cyan-200"
                      }`}
                    >
                      <SubgraphIcon className="h-3 w-3 shrink-0" />
                      <span className="truncate">{scope.name}</span>
                    </button>
                  </span>
                );
              })}
              <span className="ml-1 whitespace-nowrap text-[10px] text-zinc-600">
                nested canvas — run from Root
              </span>
            </nav>
          </Panel>
        )}
      </ReactFlow>
      <CanvasStatusBar />
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        aria-label="Import workflow file"
        className="hidden"
        onChange={onFileInputChange}
      />
      <SelectionToolbar />
      {nodes.every(isFrameNode) && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="rounded-lg border border-dashed border-zinc-700 bg-zinc-900/80 px-6 py-4 text-center">
            <p className="text-sm text-zinc-300">
              Double-click the canvas to search nodes — or open Nodes in the
              left rail
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
          onClose={closeOverlays}
          inputRef={searchInputRef}
        />
      )}
      {canvasMenu && (
        <CanvasContextMenu
          anchor={canvasMenu.anchor}
          flowPosition={canvasMenu.flowPosition}
          onClose={closeCanvasMenu}
          onDismissAll={closeOverlays}
          onFocusSearch={() => {
            if (searchInputRef.current) {
              searchInputRef.current.focus();
            } else if (canvasMenu) {
              openSearchAt(canvasMenu.anchor);
            }
          }}
          onFitView={handleFitView}
          onImport={openImport}
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
