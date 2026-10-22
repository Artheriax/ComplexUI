import type { Edge, Node, XYPosition } from "@xyflow/react";

export type SocketType =
  | "DATAFRAME"
  | "DATASET"
  | "ML_MODEL"
  | "METRICS"
  | "TENSOR"
  | "IMAGE"
  | "IMAGE_BATCH"
  | "VIDEO"
  | "MEDIA"
  | "MODEL"
  | "CLIP"
  | "VAE"
  | "CONDITIONING"
  | "LATENT"
  | "SIGMAS"
  | "SAMPLER"
  | "NOISE"
  | "GUIDER"
  | "DETECTION_MODEL"
  | "SEGMENTATION_MODEL"
  | "CLASSIFICATION_MODEL"
  | "DETECTIONS"
  | "MASKS"
  | "CLASSIFICATIONS"
  | "INT"
  | "FLOAT"
  | "BOOL"
  | "STRING"
  | "ANY";

const SOCKET_TYPE_NAMES = new Set<SocketType>([
  "DATAFRAME",
  "DATASET",
  "ML_MODEL",
  "METRICS",
  "TENSOR",
  "IMAGE",
  "IMAGE_BATCH",
  "VIDEO",
  "MEDIA",
  "MODEL",
  "CLIP",
  "VAE",
  "CONDITIONING",
  "LATENT",
  "SIGMAS",
  "SAMPLER",
  "NOISE",
  "GUIDER",
  "DETECTION_MODEL",
  "SEGMENTATION_MODEL",
  "CLASSIFICATION_MODEL",
  "DETECTIONS",
  "MASKS",
  "CLASSIFICATIONS",
  "INT",
  "FLOAT",
  "BOOL",
  "STRING",
  "ANY",
]);

export function widgetSocketType(dataType: string): SocketType {
  const kind = dataType.trim().toLowerCase();
  if (kind === "int") {
    return "INT";
  }
  if (kind === "float") {
    return "FLOAT";
  }
  if (kind === "bool") {
    return "BOOL";
  }
  if (
    kind === "str" ||
    kind === "string" ||
    kind === "file" ||
    kind === "path" ||
    kind === "folder" ||
    kind === "directory" ||
    kind === "modelpath" ||
    kind === "imagefile" ||
    kind === "videofile" ||
    kind === "outputfile"
  ) {
    return "STRING";
  }
  const normalized = dataType
    .trim()
    .replace(/^['"]|['"]$/g, "")
    .toUpperCase() as SocketType;
  return SOCKET_TYPE_NAMES.has(normalized) ? normalized : "ANY";
}

export function isPrimitiveSocketType(type: SocketType): boolean {
  return type === "INT" || type === "FLOAT" || type === "BOOL" || type === "STRING";
}

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
  bundleEdgeIds?: string[];
}

export interface InputSocket {
  name: string;
  data_type: SocketType;
}

export interface OutputSocket {
  name: string;
  data_type: SocketType;
}

export type WidgetChoice = string | number | boolean;

export interface WidgetParam {
  name: string;
  data_type: string;
  default: unknown;
  options?: WidgetChoice[];
  model_type?: string | null;
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
  source?: "builtin" | "user";
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
  approval_id?: string;
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

export type ExportFormat = "python" | "notebook";

export interface ExportedWorkflow {
  filename: string;
  media_type: string;
  content: string;
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
  | "NODE_PROGRESS"
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

export interface ImagePreview {
  kind: "image";
  width: number;
  height: number;
  mode: string;
  thumbnail: string | null;
}

export interface VideoPreview {
  kind: "video";
  path: string | null;
  url: string | null;
  fps: number | null;
  frames: number | null;
  width: number | null;
  height: number | null;
  duration: number | null;
  poster: string | null;
}

export interface ImageBatchPreview {
  kind: "image_batch";
  count: number;
  width: number | null;
  height: number | null;
  mode: string;
  thumbnail: string | null;
}

export interface DiffusionModelPreview {
  kind: "diffusion_model";
  arch: string;
  name: string;
  class: string;
  parameters: number | null;
  device: string | null;
  dtype: string;
  source: string;
  model_type: string | null;
  path: string | null;
  format: string | null;
  fingerprint: string | null;
  preset: string | null;
  recommended: Record<string, unknown>;
}

export interface ClipPreview {
  kind: "clip";
  arch: string;
  encoders: number;
  device: string | null;
  dtype: string;
}

export interface VaePreview {
  kind: "vae";
  arch: string;
  scaling_factor: number | null;
  shift_factor: number | null;
  device: string | null;
  dtype: string;
}

export interface ConditioningPreview {
  kind: "conditioning";
  arch: string;
  shape: number[];
  pooled_shape: number[] | null;
  text: string;
}

export interface LatentPreview {
  kind: "latent";
  shape: number[];
  dtype: string;
  device: string;
  width: number | null;
  height: number | null;
  batch_size: number | null;
}

export interface SigmasPreview {
  kind: "sigmas";
  sigmas: number[];
  steps: number;
  scheduler: string;
  denoise: number;
  sigma_max: number | null;
  sigma_min: number | null;
}

export interface SamplerPreview {
  kind: "sampler";
  name: string;
  custom: boolean;
}

export interface NoisePreview {
  kind: "noise";
  noise_type: string;
  seed: number | null;
  alpha: number;
  k: number;
  scale: number;
}

export interface GuiderPreview {
  kind: "guider";
  cfg: number | null;
  arch: string;
  model: string;
  positive_text: string;
  negative_text: string;
}

export interface VisionModelPreview {
  kind: "vision_model";
  backend: string;
  task: string;
  model: string;
  weights: unknown;
  device: unknown;
  classes: unknown[];
  source?: "builtin" | "local" | string;
  model_type?: string | null;
  path?: string | null;
  format?: string | null;
  fingerprint?: string | null;
}

export interface DetectionsPreview {
  kind: "detections";
  count: number;
  items: unknown[];
}

export interface MasksPreview {
  kind: "masks";
  count: number;
  thumbnail: string | null;
}

export interface ClassificationsPreview {
  kind: "classifications";
  items: unknown[];
}

export type PreviewValue =
  | DataFramePreview
  | DatasetPreview
  | ModelPreview
  | TensorPreview
  | ImagePreview
  | ImageBatchPreview
  | VideoPreview
  | DiffusionModelPreview
  | ClipPreview
  | VaePreview
  | ConditioningPreview
  | LatentPreview
  | SigmasPreview
  | SamplerPreview
  | NoisePreview
  | GuiderPreview
  | VisionModelPreview
  | DetectionsPreview
  | MasksPreview
  | ClassificationsPreview
  | Record<string, unknown>
  | string
  | number
  | boolean;

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
  progress?: number;
  progressMessage?: string;
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

export type SubgraphRole = "input" | "output";

export const SUBGRAPH_INPUT_NODE_ID = "__subgraph_input__";
export const SUBGRAPH_OUTPUT_NODE_ID = "__subgraph_output__";
export const NEW_SUBGRAPH_INPUT_HANDLE = "__new_subgraph_input__";
export const NEW_SUBGRAPH_OUTPUT_HANDLE = "__new_subgraph_output__";

export type ComplexNodeData = {
  label: string;
  templateName: string | null;
  templateCode: string | null;
  code: string;
  schema: NodeSchema;
  widgetValues: Record<string, unknown>;
  isTrusted: boolean;
  color: NodeColor;
  isFrame?: boolean;
  isSubgraph?: boolean;
  subgraphId?: string;
  subgraphRole?: SubgraphRole;
};

export type AppNode = Node<ComplexNodeData, "complex">;
export type AppEdge = Edge;

export interface SubgraphDefinition {
  id: string;
  name: string;
  inputs: InputSocket[];
  outputs: OutputSocket[];
  nodes: AppNode[];
  edges: AppEdge[];
  inputPosition: XYPosition;
  outputPosition: XYPosition;
}

export type SubgraphMap = Record<string, SubgraphDefinition>;

export interface SubgraphScope {
  definitionId: string;
  instanceId: string;
  name: string;
}

export function isFrameNode(node: AppNode): boolean {
  return node.data.isFrame === true;
}

export function isSubgraphInstanceNode(node: AppNode): boolean {
  return node.data.isSubgraph === true;
}

export function isSubgraphManagerNode(node: AppNode): boolean {
  return node.data.subgraphRole !== undefined;
}

export function isRunnableNode(node: AppNode): boolean {
  return !isFrameNode(node) && !isSubgraphManagerNode(node);
}

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
  IMAGE: "#f472b6",
  IMAGE_BATCH: "#ec4899",
  VIDEO: "#fb7185",
  MEDIA: "#c084fc",
  MODEL: "#facc15",
  CLIP: "#38bdf8",
  VAE: "#2dd4bf",
  CONDITIONING: "#c4b5fd",
  LATENT: "#e879f9",
  SIGMAS: "#22d3ee",
  SAMPLER: "#fbbf24",
  NOISE: "#a3e635",
  GUIDER: "#fb7185",
  DETECTION_MODEL: "#f59e0b",
  SEGMENTATION_MODEL: "#10b981",
  CLASSIFICATION_MODEL: "#60a5fa",
  DETECTIONS: "#f97316",
  MASKS: "#2dd4bf",
  CLASSIFICATIONS: "#818cf8",
  INT: "#fb923c",
  FLOAT: "#60a5fa",
  BOOL: "#f472b6",
  STRING: "#a3e635",
  ANY: "#a1a1aa",
};

export const SOCKET_LABELS: Record<SocketType, string> = {
  DATAFRAME: "DataFrame",
  DATASET: "Dataset",
  ML_MODEL: "Model",
  METRICS: "Metrics",
  TENSOR: "Tensor",
  IMAGE: "Image",
  IMAGE_BATCH: "Image Batch",
  VIDEO: "Video",
  MEDIA: "Media",
  MODEL: "Diffusion Model",
  CLIP: "CLIP",
  VAE: "VAE",
  CONDITIONING: "Conditioning",
  LATENT: "Latent",
  SIGMAS: "Sigmas",
  SAMPLER: "Sampler",
  NOISE: "Noise",
  GUIDER: "Guider",
  DETECTION_MODEL: "Detection Model",
  SEGMENTATION_MODEL: "Segmentation Model",
  CLASSIFICATION_MODEL: "Classification Model",
  DETECTIONS: "Detections",
  MASKS: "Masks",
  CLASSIFICATIONS: "Classifications",
  INT: "Integer",
  FLOAT: "Float",
  BOOL: "Boolean",
  STRING: "String",
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
    if (record.kind === "image") {
      return `image ${String(record.width)}x${String(record.height)}`;
    }
    if (record.kind === "image_batch") {
      return `${String(record.count)} images ${String(record.width ?? "?")}x${String(record.height ?? "?")}`;
    }
    if (record.kind === "video") {
      return `video ${String(record.width ?? "?")}x${String(record.height ?? "?")}`;
    }
    if (record.kind === "diffusion_model") {
      return `${String(record.arch)} ${String(record.name)}`;
    }
    if (record.kind === "clip") {
      return `CLIP ${String(record.arch)}`;
    }
    if (record.kind === "vae") {
      return `VAE ${String(record.arch)}`;
    }
    if (record.kind === "conditioning") {
      return `conditioning ${JSON.stringify(record.shape)}`;
    }
    if (record.kind === "latent") {
      return `latent ${JSON.stringify(record.shape)}`;
    }
    if (record.kind === "sigmas") {
      return `${String(record.steps)} steps ${String(record.scheduler)}`;
    }
    if (record.kind === "sampler") {
      return `sampler ${String(record.name)}`;
    }
    if (record.kind === "noise") {
      return `noise ${String(record.noise_type)}`;
    }
    if (record.kind === "guider") {
      return `CFG ${String(record.cfg)}`;
    }
    if (record.kind === "vision_model") {
      return `${String(record.task)} ${String(record.model)}`;
    }
    if (record.kind === "detections") {
      return `${String(record.count)} detections`;
    }
    if (record.kind === "masks") {
      return `${String(record.count)} masks`;
    }
    if (record.kind === "classifications") {
      return `${Array.isArray(record.items) ? record.items.length : 0} classes`;
    }
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
