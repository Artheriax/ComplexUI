import type { Edge, Node, XYPosition } from "@xyflow/react";

export type SocketType =
  | "DATAFRAME"
  | "DATASET"
  | "ML_MODEL"
  | "METRICS"
  | "TENSOR"
  | "ANY";

export interface InputSocket {
  name: string;
  data_type: SocketType;
}

export interface OutputSocket {
  name: string;
  data_type: SocketType;
}

export interface WidgetParam {
  name: string;
  data_type: string;
  default: unknown;
}

export interface SyntaxErrorInfo {
  line: number;
  column: number;
  message: string;
}

export interface NodeSchema {
  node_id: string;
  inputs: InputSocket[];
  widgets: WidgetParam[];
  outputs: OutputSocket[];
  dependencies: string[];
  syntax_errors: SyntaxErrorInfo[];
}

export interface TemplateInfo {
  name: string;
  title: string;
  code: string;
  code_hash: string;
  node_schema: NodeSchema;
}

export interface GraphNodePayload {
  id: string;
  code: string;
  widget_values: Record<string, unknown>;
}

export interface GraphEdgePayload {
  source_node_id: string;
  source_handle: string;
  target_node_id: string;
  target_handle: string;
}

export interface WorkflowGraphPayload {
  nodes: GraphNodePayload[];
  edges: GraphEdgePayload[];
}

export interface ExecuteResponse {
  status: string;
  run_id: string;
}

export type EventType =
  | "EXECUTION_STARTED"
  | "EXECUTION_COMPLETED"
  | "EXECUTION_FAILED"
  | "EXECUTION_CANCELLED"
  | "NODE_STARTED"
  | "NODE_COMPLETED"
  | "NODE_ERROR"
  | "INSTALLING_DEPS";

export interface ExecutionEvent {
  type: EventType;
  node_id: string;
  data: Record<string, unknown>;
}

export interface UntrustedNodeRef {
  node_id: string;
  code_hash: string;
}

export interface DataFramePreview {
  rows: number;
  cols: number;
  head: Record<string, unknown>[];
}

export interface DatasetPreview {
  X_shape: [number, number];
  y_shape: [number];
  dtype: string;
}

export interface ModelPreview {
  class: string;
  params?: Record<string, unknown>;
  parameters?: number;
  layers?: number;
}

export interface TensorPreview {
  shape: number[];
  dtype: string;
  device: string;
  sample: number[];
}

export type PreviewValue =
  | DataFramePreview
  | DatasetPreview
  | ModelPreview
  | TensorPreview
  | Record<string, unknown>
  | string;

export type NodeExecutionStatus = "idle" | "running" | "completed" | "error";

export interface NodeExecutionState {
  status: NodeExecutionStatus;
  cached?: boolean;
  installing?: string[];
  outputs?: Record<string, PreviewValue>;
  error?: {
    line: number | null;
    message: string;
    errorType: string;
    traceback: string;
  };
}

export interface ExecutionError {
  message: string;
  errorType: string;
  nodeId: string | null;
  line: number | null;
  traceback: string;
}

export type ExecutionStatus = "idle" | "running" | "completed" | "failed" | "cancelled";

export type ComplexNodeData = {
  label: string;
  templateName: string | null;
  templateCode: string | null;
  code: string;
  schema: NodeSchema;
  widgetValues: Record<string, unknown>;
  isTrusted: boolean;
};

export type AppNode = Node<ComplexNodeData, "complex">;
export type AppEdge = Edge;

export interface PendingApproval {
  nodeId: string;
  code: string;
  codeHash: string;
  schema: NodeSchema;
}

export interface NodePositionUpdate {
  id: string;
  position: XYPosition;
}

export function defaultWidgetValues(schema: NodeSchema): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const widget of schema.widgets) {
    values[widget.name] = widget.default;
  }
  return values;
}

export const SOCKET_COLORS: Record<SocketType, string> = {
  DATAFRAME: "#38bdf8",
  DATASET: "#a78bfa",
  ML_MODEL: "#fbbf24",
  METRICS: "#34d399",
  TENSOR: "#fb7185",
  ANY: "#a1a1aa",
};

export const SOCKET_LABELS: Record<SocketType, string> = {
  DATAFRAME: "DataFrame",
  DATASET: "Dataset",
  ML_MODEL: "Model",
  METRICS: "Metrics",
  TENSOR: "Tensor",
  ANY: "Any",
};

export function summarisePreview(value: PreviewValue): string {
  if (value === null || value === undefined) {
    return "null";
  }
  if (typeof value === "string") {
    return value.length > 48 ? `${value.slice(0, 45)}...` : value;
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    if ("rows" in record && "cols" in record) {
      return `${record.rows as number} rows x ${record.cols as number} cols`;
    }
    if ("X_shape" in record) {
      return `X${JSON.stringify(record.X_shape)} y${JSON.stringify(record.y_shape)}`;
    }
    if ("shape" in record && "dtype" in record) {
      return `tensor ${JSON.stringify(record.shape)} ${String(record.dtype)}`;
    }
    if ("class" in record) {
      return String(record.class);
    }
    if ("accuracy" in record) {
      return `accuracy ${Number(record.accuracy).toFixed(4)}`;
    }
    return JSON.stringify(record).slice(0, 60);
  }
  return String(value);
}
