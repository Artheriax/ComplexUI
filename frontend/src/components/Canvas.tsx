import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import type { DragEvent } from "react";
import { useCallback } from "react";

import { useGraphStore } from "../store/useGraphStore";
import { useViewportShortcuts } from "../hooks/useViewportShortcuts";
import { ComplexNode } from "./ComplexNode";

export const DRAG_MIME = "application/x-complexui-template";

const nodeTypes = { complex: ComplexNode };

function CanvasInner() {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const onNodesChange = useGraphStore((state) => state.onNodesChange);
  const onEdgesChange = useGraphStore((state) => state.onEdgesChange);
  const onConnect = useGraphStore((state) => state.onConnect);
  const openEditor = useGraphStore((state) => state.openEditor);
  const { screenToFlowPosition, zoomIn, zoomOut } = useReactFlow();
  const handleZoomIn = useCallback(() => {
    void zoomIn({ duration: 150 });
  }, [zoomIn]);
  const handleZoomOut = useCallback(() => {
    void zoomOut({ duration: 150 });
  }, [zoomOut]);

  useViewportShortcuts(handleZoomIn, handleZoomOut);

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
    <div className="relative h-full w-full" onDragOver={onDragOver} onDrop={onDrop}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDoubleClick={(_, node) => openEditor(node.id)}
        selectionKeyCode={["Control", "Meta"]}
        multiSelectionKeyCode={["Control", "Meta"]}
        onBeforeDelete={async () => {
          useGraphStore.getState().checkpoint();
          return true;
        }}
        deleteKeyCode={["Backspace", "Delete"]}
        fitView
        proOptions={{ hideAttribution: false }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.5} color="#27272a" />
        <MiniMap
          pannable
          zoomable
          style={{ background: "#18181b" }}
          nodeColor="#3f3f46"
          maskColor="rgba(9, 9, 11, 0.7)"
        />
        <Controls showInteractive={false} />
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
