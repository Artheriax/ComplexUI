import type {
  AppEdge,
  ConnectionHandleType,
  InputSocket,
  NodeSchema,
  OutputSocket,
  SocketType,
  TemplateInfo,
  WidgetParam,
} from "../types/graph";
import { widgetSocketType } from "../types/graph";

export interface TemplateConnectionMatch {
  socketName: string;
  socketType: SocketType;
  targetKind: "input" | "widget" | "output";
}

export function socketTypesCompatible(
  source: SocketType,
  target: SocketType,
): boolean {
  return (
    source === target ||
    source === "ANY" ||
    target === "ANY" ||
    (source === "INT" && target === "FLOAT") ||
    (source === "IMAGE" && target === "IMAGE_BATCH") ||
    (source === "IMAGE_BATCH" && target === "IMAGE") ||
    ((source === "IMAGE" ||
      source === "IMAGE_BATCH" ||
      source === "VIDEO") &&
      target === "MEDIA")
  );
}

export function findCompatibleInput(
  schema: NodeSchema,
  sourceType: SocketType,
  isTaken: (socketName: string) => boolean = () => false,
): InputSocket | null {
  return (
    schema.inputs.find(
      (socket) =>
        !isTaken(socket.name) &&
        socketTypesCompatible(sourceType, socket.data_type),
    ) ?? null
  );
}

export function findCompatibleOutput(
  schema: NodeSchema,
  targetType: SocketType,
): OutputSocket | null {
  return (
    schema.outputs.find((socket) =>
      socketTypesCompatible(socket.data_type, targetType),
    ) ?? null
  );
}

export function resolveTargetSocket(
  schema: NodeSchema,
  handle: string,
): InputSocket | null {
  const input = schema.inputs.find((socket) => socket.name === handle);
  if (input) {
    return input;
  }
  const widget = schema.widgets.find(
    (candidate) => candidate.name === handle,
  );
  return widget
    ? { name: widget.name, data_type: widgetSocketType(widget.data_type) }
    : null;
}

export function findCompatibleWidget(
  schema: NodeSchema,
  sourceType: SocketType,
): WidgetParam | null {
  return (
    schema.widgets.find((widget) =>
      socketTypesCompatible(sourceType, widgetSocketType(widget.data_type)),
    ) ?? null
  );
}

export function findCompatibleTarget(
  schema: NodeSchema,
  sourceType: SocketType,
): TemplateConnectionMatch | null {
  const input = findCompatibleInput(schema, sourceType);
  if (input) {
    return {
      socketName: input.name,
      socketType: input.data_type,
      targetKind: "input",
    };
  }
  const widget = findCompatibleWidget(schema, sourceType);
  return widget
    ? {
        socketName: widget.name,
        socketType: widgetSocketType(widget.data_type),
        targetKind: "widget",
      }
    : null;
}

export function isInputTaken(
  edges: AppEdge[],
  targetNodeId: string,
  targetHandle: string,
): boolean {
  return edges.some(
    (edge) =>
      edge.target === targetNodeId && edge.targetHandle === targetHandle,
  );
}

export function matchTemplateConnection(
  template: TemplateInfo,
  handleType: ConnectionHandleType,
  dataType: SocketType,
): TemplateConnectionMatch | null {
  if (handleType === "source") {
    return findCompatibleTarget(template.node_schema, dataType);
  }
  const output = findCompatibleOutput(template.node_schema, dataType);
  return output
    ? {
        socketName: output.name,
        socketType: output.data_type,
        targetKind: "output",
      }
    : null;
}
