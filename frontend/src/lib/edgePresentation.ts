import { ConnectionLineType, MarkerType } from "@xyflow/react";

import type {
  AppSettings,
  EdgeLineStyle,
} from "../store/useSettingsStore";
import type { AppEdge, AppNode, SocketType } from "../types/graph";
import { SOCKET_COLORS } from "../types/graph";

export const EDGE_TYPE_BY_STYLE: Record<EdgeLineStyle, string> = {
  bezier: "default",
  smoothstep: "smoothstep",
  step: "step",
  straight: "straight",
};

export const CONNECTION_LINE_TYPE_BY_STYLE: Record<
  EdgeLineStyle,
  ConnectionLineType
> = {
  bezier: ConnectionLineType.Bezier,
  smoothstep: ConnectionLineType.SmoothStep,
  step: ConnectionLineType.Step,
  straight: ConnectionLineType.Straight,
};

function socketKey(nodeId: string, handle: string): string {
  return `${nodeId}\u0000${handle}`;
}

function collectOutputTypes(nodes: AppNode[]): Map<string, SocketType> {
  const types = new Map<string, SocketType>();
  for (const node of nodes) {
    for (const socket of node.data.schema.outputs) {
      types.set(socketKey(node.id, socket.name), socket.data_type);
    }
  }
  return types;
}

export function buildDisplayEdges(
  edges: AppEdge[],
  nodes: AppNode[],
  settings: AppSettings,
): AppEdge[] {
  const outputTypes = collectOutputTypes(nodes);
  return edges.map((edge) => {
    const sourceType =
      outputTypes.get(socketKey(edge.source, edge.sourceHandle ?? "")) ?? "ANY";
    const color =
      settings.edgeColorMode === "single"
        ? settings.accentColor
        : SOCKET_COLORS[sourceType];
    const markerSize = Math.max(10, settings.edgeWidth * 4);
    return {
      ...edge,
      type: EDGE_TYPE_BY_STYLE[settings.edgeStyle],
      hidden: !settings.showEdges || edge.hidden === true,
      animated: settings.edgeAnimated,
      interactionWidth: Math.max(20, settings.edgeWidth * 6),
      style: {
        ...edge.style,
        stroke: color,
        strokeWidth: settings.edgeWidth,
      },
      markerEnd: settings.showArrowheads
        ? {
            type: MarkerType.ArrowClosed,
            color,
            width: markerSize,
            height: markerSize,
          }
        : undefined,
    };
  });
}
