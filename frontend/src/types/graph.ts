import type { Edge, Node, XYPosition } from "@xyflow/react";

export type SocketType =
  | "DATAFRAME"
  | "DATASET"
  | "ML_MODEL"
  | "METRICS"
  | "TENSOR"
  | "ANY";

export type NodeColor =
  | "default"
  | "sky"
  | "emerald"
  | "amber"
  | "rose"
  | "violet"
  | "zinc";

export type ConnectionHandleType = "source" | "target";

export interface PendingConnection {
  nodeId: string;
  handleId: string;
  handleType: ConnectionHandleType;
  dataType: SocketType;
}

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
  category: string;
  description: string;
  hidden: boolean;
  code: string;
  code_hash: string;
  node_schema: NodeSchema;
}

export interface WorkflowNodeSpec {
  id: string;
  template: string;
  position: { x: number; y: number };
  widget_values: Record<string, unknown>;
}

export interface WorkflowEdgeSpec {
  source_node_id: string;
  source_handle: string;
  target_node_id: string;
  target_handle: string;
}

export interface WorkflowTemplateInfo {
  name: string;
  title: string;
  category: string;
  description: string;
  nodes: WorkflowNodeSpec[];
  edges: WorkflowEdgeSpec[];
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

export interface BackendRunStatus {
  status: "idle" | "running";
  run_id: string | null;
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

export interface TextRichOutput {
  kind: "text";
  source: string;
  text: string;
  truncated: boolean;
}

export interface TableRichOutput {
  kind: "table";
  source: string;
  columns: string[];
  rows: unknown[][];
  total_rows: number;
  total_cols: number;
  truncated: boolean;
}

export interface ImageRichOutput {
  kind: "image";
  source: string;
  mime: string;
  data: string;
  alt: string;
}

export interface HtmlRichOutput {
  kind: "html";
  source: string;
  html: string;
  truncated: boolean;
}

export interface PlotlyRichOutput {
  kind: "plotly";
  source: string;
  figure: {
    data?: unknown[];
    layout?: Record<string, unknown>;
    frames?: unknown[];
  };
}

export type RichOutput =
  | TextRichOutput
  | TableRichOutput
  | ImageRichOutput
  | HtmlRichOutput
  | PlotlyRichOutput;

const RICH_OUTPUT_KINDS = new Set<RichOutput["kind"]>([
  "text",
  "table",
  "image",
  "html",
  "plotly",
]);

export function parseRichOutputs(value: unknown): RichOutput[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is RichOutput => {
    if (typeof item !== "object" || item === null) {
      return false;
    }
    const kind = (item as Record<string, unknown>).kind;
    return typeof kind === "string" && RICH_OUTPUT_KINDS.has(kind as RichOutput["kind"]);
  });
}

export type NodeExecutionStatus = "idle" | "running" | "completed" | "error";

export interface NodeExecutionState {
  status: NodeExecutionStatus;
  cached?: boolean;
  installing?: string[];
  outputs?: Record<string, PreviewValue>;
  richOutputs?: RichOutput[];
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

export interface NodeColorMeta {
  label: string;
  accent: string;
}

export const NODE_COLOR_ORDER: NodeColor[] = [
  "default",
  "sky",
  "emerald",
  "amber",
  "rose",
  "violet",
  "zinc",
];

export const NODE_COLORS: Record<NodeColor, NodeColorMeta> = {
  default: { label: "Default", accent: "#52525b" },
  sky: { label: "Sky", accent: "#38bdf8" },
  emerald: { label: "Emerald", accent: "#34d399" },
  amber: { label: "Amber", accent: "#fbbf24" },
  rose: { label: "Rose", accent: "#fb7185" },
  violet: { label: "Violet", accent: "#a78bfa" },
  zinc: { label: "Zinc", accent: "#a1a1aa" },
};

export function isNodeColor(value: unknown): value is NodeColor {
  return (
    typeof value === "string" &&
    (NODE_COLOR_ORDER as readonly string[]).includes(value)
  );
}

export type ComplexNodeData = {
  label: string;
  templateName: string | null;
  templateCode: string | null;
  code: string;
  schema: NodeSchema;
  widgetValues: Record<string, unknown>;
  isTrusted: boolean;
  color: NodeColor;
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
