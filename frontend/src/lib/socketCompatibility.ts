import type {
  AppEdge,
  ConnectionHandleType,
  InputSocket,
  NodeSchema,
  OutputSocket,
  SocketType,
  TemplateInfo,
} from "../types/graph";

export interface TemplateConnectionMatch {
  socketName: string;
  socketType: SocketType;
}

export function socketTypesCompatible(
  source: SocketType,
  target: SocketType,
): boolean {
  return source === target || source === "ANY" || target === "ANY";
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
    const input = findCompatibleInput(template.node_schema, dataType);
    return input
      ? { socketName: input.name, socketType: input.data_type }
      : null;
  }
  const output = findCompatibleOutput(template.node_schema, dataType);
  return output
    ? { socketName: output.name, socketType: output.data_type }
    : null;
}
