import {
  Background,
  BackgroundVariant,
  Controls,
  getBezierPath,
  getSmoothStepPath,
  getStraightPath,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import type {
  Connection,
  FinalConnectionState,
  NodeMouseHandler,
  OnConnectStart,
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
import { isInteractiveControlTarget } from "../lib/interactiveTarget";
import { resolveTargetSocket } from "../lib/socketCompatibility";
import { useBundleDragStore } from "../store/useBundleDragStore";
import { useConnectionGuideStore } from "../store/useConnectionGuideStore";
import { useEdgeDragStore } from "../store/useEdgeDragStore";
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

const EMPTY_EDGE_IDS: string[] = [];

function BundleDragOverlay() {
  const drag = useBundleDragStore((state) => state.drag);
  const edges = useGraphStore((state) => state.edges);
  if (!drag) {
    return null;
  }
  const handles = Array.from(
    document.querySelectorAll<HTMLElement>("[data-cu-handle='target']"),
  );
  const lines = drag.edgeIds.flatMap((edgeId) => {
    const edge = edges.find((candidate) => candidate.id === edgeId);
    if (!edge) {
      return [];
    }
    const handle = handles.find(
      (candidate) =>
        candidate.dataset.nodeId === edge.target &&
        candidate.dataset.handleId === edge.targetHandle,
    );
    if (!handle) {
      return [];
    }
    const rect = handle.getBoundingClientRect();
    return [
      {
        id: edge.id,
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
      },
    ];
  });
  return (
    <svg className="pointer-events-none fixed inset-0 z-[55] h-full w-full">
      {lines.map((line) => (
        <line
          key={line.id}
          x1={drag.current.x}
          y1={drag.current.y}
          x2={line.x}
          y2={line.y}
          stroke="#22d3ee"
          strokeWidth={2}
          strokeOpacity={0.85}
          strokeDasharray="5 4"
        />
      ))}
      <circle
        cx={drag.current.x}
        cy={drag.current.y}
        r={4}
        fill="#22d3ee"
      />
    </svg>
  );
}

function EdgeDragOverlay() {
  const drag = useEdgeDragStore((state) => state.drag);
  const edgeStyle = useSettingsStore((state) => state.settings.edgeStyle);
  const accentColor = useSettingsStore(
    (state) => state.settings.accentColor,
  );
  const edgeWidth = useSettingsStore((state) => state.settings.edgeWidth);
  const path = useMemo(() => {
    if (!drag) {
      return "";
    }
    const sourceX = drag.sourceScreen.x;
    const sourceY = drag.sourceScreen.y;
    const targetX = drag.current.x;
    const targetY = drag.current.y;
    if (edgeStyle === "straight") {
      return getStraightPath({ sourceX, sourceY, targetX, targetY })[0];
    }
    if (edgeStyle === "smoothstep") {
      return getSmoothStepPath({
        sourceX,
        sourceY,
        sourcePosition: Position.Right,
        targetX,
        targetY,
        targetPosition: Position.Left,
      })[0];
    }
    if (edgeStyle === "step") {
      return getSmoothStepPath({
        sourceX,
        sourceY,
        sourcePosition: Position.Right,
        targetX,
        targetY,
        targetPosition: Position.Left,
        borderRadius: 0,
      })[0];
    }
    return getBezierPath({
      sourceX,
      sourceY,
      sourcePosition: Position.Right,
      targetX,
      targetY,
      targetPosition: Position.Left,
    })[0];
  }, [drag, edgeStyle]);
  if (!drag) {
    return null;
  }
  return (
    <svg className="pointer-events-none fixed inset-0 z-[55] h-full w-full">
      <path
        className="cu-edge-retarget-line"
        d={path}
        fill="none"
        stroke={accentColor}
        strokeWidth={edgeWidth}
        strokeLinecap="round"
      />
    </svg>
  );
}

function CanvasInner() {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const onNodesChange = useGraphStore((state) => state.onNodesChange);
  const onEdgesChange = useGraphStore((state) => state.onEdgesChange);
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
  const suppressPaneClickRef = useRef(false);
  const suppressNextPaneClick = useCallback(() => {
    suppressPaneClickRef.current = true;
    window.setTimeout(() => {
      suppressPaneClickRef.current = false;
    }, 0);
  }, []);
  const bundleEdgeIds = useBundleDragStore(
    (state) => state.drag?.edgeIds ?? EMPTY_EDGE_IDS,
  );
  const bundleActive = useBundleDragStore((state) => state.drag !== null);
  const edgeDragArmed = useEdgeDragStore(
    (state) => state.pending !== null || state.drag !== null,
  );
  const closeMenu = useCallback(() => setMenuOverlay(null), []);
  const closeCanvasMenu = useCallback(() => setCanvasMenu(null), []);
  const closeOverlays = useCallback(() => {
    useBundleDragStore.getState().clear();
    useEdgeDragStore.getState().clear();
    useConnectionGuideStore.getState().clear();
    setSearchOverlay(null);
    setMenuOverlay(null);
    setCanvasMenu(null);
  }, []);
  const handlePaneClick = useCallback(() => {
    if (suppressPaneClickRef.current) {
      return;
    }
    closeOverlays();
  }, [closeOverlays]);
  useEffect(
    () => () => {
      useBundleDragStore.getState().clear();
      useEdgeDragStore.getState().clear();
      useConnectionGuideStore.getState().clear();
    },
    [],
  );

  useEffect(() => {
    if (!bundleActive) {
      return;
    }
    const onPointerMove = (event: PointerEvent) => {
      useBundleDragStore.getState().move(event.clientX, event.clientY);
    };
    const finish = (event: PointerEvent) => {
      const drag = useBundleDragStore.getState().drag;
      if (!drag) {
        return;
      }
      event.preventDefault();
      const element = document.elementFromPoint(event.clientX, event.clientY);
      const outputElement =
        element instanceof Element
          ? element.closest<HTMLElement>("[data-cu-output-handle='true']")
          : null;
      if (outputElement) {
        const nodeId = outputElement.dataset.nodeId ?? "";
        const handleId = outputElement.dataset.handleId ?? "";
        const socketType = outputElement.dataset.socketType ?? "";
        if (
          nodeId &&
          handleId &&
          socketType === drag.dataType &&
          !(nodeId === drag.sourceNodeId && handleId === drag.sourceHandleId)
        ) {
          useGraphStore
            .getState()
            .reassignOutputBundle(drag.edgeIds, nodeId, handleId);
          useBundleDragStore.getState().clear();
          useConnectionGuideStore.getState().clear();
          return;
        }
      }
      const anyHandle =
        element instanceof Element
          ? element.closest<HTMLElement>("[data-cu-handle]")
          : null;
      useBundleDragStore.getState().clear();
      useConnectionGuideStore.getState().clear();
      if (anyHandle) {
        return;
      }
      const point = { x: event.clientX, y: event.clientY };
      suppressNextPaneClick();
      setMenuOverlay(null);
      setCanvasMenu(null);
      setSearchOverlay({
        anchor: point,
        flowPosition: screenToFlowPosition(point),
        connection: {
          nodeId: drag.sourceNodeId,
          handleId: drag.sourceHandleId,
          handleType: "source",
          dataType: drag.dataType,
          bundleEdgeIds: drag.edgeIds,
        },
      });
    };
    const cancel = () => {
      useBundleDragStore.getState().clear();
      useConnectionGuideStore.getState().clear();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        cancel();
      }
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", cancel);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [bundleActive, screenToFlowPosition, suppressNextPaneClick]);

  useEffect(() => {
    if (!edgeDragArmed) {
      return;
    }
    const onPointerMove = (event: PointerEvent) => {
      const state = useEdgeDragStore.getState();
      if (state.drag) {
        state.move(event.clientX, event.clientY);
        return;
      }
      const pending = state.pending;
      if (!pending) {
        return;
      }
      const distance = Math.hypot(
        event.clientX - pending.start.x,
        event.clientY - pending.start.y,
      );
      if (distance < 4) {
        return;
      }
      state.activate({ x: event.clientX, y: event.clientY });
      useConnectionGuideStore.getState().begin({
        mode: "connect",
        direction: "source",
        dataType: pending.dataType,
        originNodeId: pending.sourceNodeId,
        originHandleId: pending.sourceHandleId,
        ignoredTargetNodeId: pending.targetNodeId,
        ignoredTargetHandleId: pending.targetHandleId,
      });
    };
    const finish = (event: PointerEvent) => {
      const state = useEdgeDragStore.getState();
      const drag = state.drag;
      if (!drag) {
        state.clear();
        return;
      }
      event.preventDefault();
      const element = document.elementFromPoint(event.clientX, event.clientY);
      const targetElement =
        element instanceof Element
          ? element.closest<HTMLElement>("[data-cu-handle='target']")
          : null;
      if (targetElement) {
        const targetNodeId = targetElement.dataset.nodeId ?? "";
        const targetHandleId = targetElement.dataset.handleId ?? "";
        if (targetNodeId && targetHandleId) {
          useGraphStore
            .getState()
            .reconnectEdgeTarget(
              drag.edgeId,
              targetNodeId,
              targetHandleId,
            );
        }
        state.clear();
        useConnectionGuideStore.getState().clear();
        return;
      }
      const anyHandle =
        element instanceof Element
          ? element.closest<HTMLElement>("[data-cu-handle]")
          : null;
      state.clear();
      useConnectionGuideStore.getState().clear();
      if (anyHandle) {
        return;
      }
      useGraphStore.getState().deleteEdge(drag.edgeId);
      if (!useSettingsStore.getState().settings.connectionDropSearch) {
        return;
      }
      const point = { x: event.clientX, y: event.clientY };
      suppressNextPaneClick();
      setMenuOverlay(null);
      setCanvasMenu(null);
      setSearchOverlay({
        anchor: point,
        flowPosition: screenToFlowPosition(point),
        connection: {
          nodeId: drag.sourceNodeId,
          handleId: drag.sourceHandleId,
          handleType: "source",
          dataType: drag.dataType,
        },
      });
    };
    const cancel = () => {
      useEdgeDragStore.getState().clear();
      useConnectionGuideStore.getState().clear();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        cancel();
      }
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", cancel);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [edgeDragArmed, screenToFlowPosition, suppressNextPaneClick]);

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

  const handleConnectStart = useCallback<OnConnectStart>((_event, params) => {
    const graph = useGraphStore.getState();
    const nodeId = params.nodeId ?? "";
    const node = graph.nodes.find((candidate) => candidate.id === nodeId);
    const handleType = params.handleType;
    const handleId = params.handleId ?? "";
    if (
      !node ||
      !handleId ||
      (handleType !== "source" && handleType !== "target")
    ) {
      useConnectionGuideStore.getState().clear();
      return;
    }
    const socket =
      handleType === "source"
        ? node.data.schema.outputs.find(
            (candidate) => candidate.name === handleId,
          )
        : resolveTargetSocket(node.data.schema, handleId);
    if (!socket) {
      useConnectionGuideStore.getState().clear();
      return;
    }
    useConnectionGuideStore.getState().begin({
      mode: "connect",
      direction: handleType,
      dataType: socket.data_type,
      originNodeId: nodeId,
      originHandleId: handleId,
    });
  }, []);

  const handleConnect = useCallback((connection: Connection) => {
    useConnectionGuideStore.getState().clear();
    useGraphStore.getState().onConnect(connection);
  }, []);

  const handleConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, connectionState: FinalConnectionState) => {
      const fromHandle = connectionState.fromHandle;
      if (!fromHandle?.id) {
        return;
      }
      useConnectionGuideStore.getState().clear();
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

  const displayEdges = useMemo(() => {
    const hidden = new Set(bundleEdgeIds);
    return buildDisplayEdges(edges, nodes, settings).filter(
      (edge) => !hidden.has(edge.id),
    );
  }, [bundleEdgeIds, edges, nodes, settings]);
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
        onConnect={handleConnect}
        onConnectStart={handleConnectStart}
        onConnectEnd={handleConnectEnd}
        onClickConnectEnd={handleConnectEnd}
        onNodeDoubleClick={(event, node) => {
          if (isInteractiveControlTarget(event.target)) {
            return;
          }
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
        onPaneClick={handlePaneClick}
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
      <BundleDragOverlay />
      <EdgeDragOverlay />
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
