import type {
  BackendRunStatus,
  ExecuteResponse,
  ExportFormat,
  ExportedWorkflow,
  ModelMemoryEstimate,
  ModelQuantizationInfo,
  NodeSchema,
  PreviewValue,
  RichOutput,
  TemplateInfo,
  UntrustedNodeRef,
  WorkflowGraphPayload,
  WorkflowTemplateInfo,
} from "../types/graph";
import type { SavedWorkflow } from "./persistence";

export type { ModelMemoryEstimate, ModelQuantizationInfo } from "../types/graph";

const API_BASE = "/api";

export interface ProjectSummary {
  slug: string;
  name: string;
  saved_at: string;
  node_count: number;
  edge_count: number;
  version_count: number;
}

export interface ProjectDetail extends ProjectSummary {
  workflow: SavedWorkflow;
}

export interface ProjectSaveResult {
  status: "created" | "updated";
  project: ProjectSummary;
}

export interface ProjectVersionInfo {
  version: string;
  saved_at: string;
  node_count: number;
  edge_count: number;
  size_bytes: number;
}

export interface ProjectVersionDetail extends ProjectVersionInfo {
  workflow: SavedWorkflow;
}

export interface WorkspaceSession {
  saved_at: string;
  project_slug: string | null;
  project_name: string | null;
  workflow: SavedWorkflow;
}

export type RunStatus = "completed" | "failed" | "cancelled";
export type RunNodeStatus = "pending" | "completed" | "error";

export interface RunErrorRecord {
  message: string;
  error_type: string;
  node_id: string | null;
  line: number | null;
  traceback: string;
}

export interface RunNodeRecord {
  node_id: string;
  label: string;
  status: RunNodeStatus;
  duration_ms: number | null;
  cached: boolean;
  previews: Record<string, PreviewValue>;
  rich_outputs: RichOutput[];
  rich_outputs_truncated: boolean;
  metrics?: Record<string, unknown> | null;
  error: RunErrorRecord | null;
}

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  started_at: string;
  finished_at: string;
  duration_ms: number;
  node_count: number;
  completed_count: number;
  cached_count: number;
  error_type: string | null;
}

export interface RunDetail extends RunSummary {
  order: string[];
  targets: string[] | null;
  node_labels: Record<string, string>;
  nodes: RunNodeRecord[];
  error: RunErrorRecord | null;
}

export type AssetKind = "table" | "image" | "video";

export interface DatasetSummary {
  id: string;
  name: string;
  path: string;
  format: string;
  kind: AssetKind;
  size_bytes: number;
  modified_at: string;
}

export function datasetRawUrl(datasetId: string): string {
  return `${API_BASE}/uploads/${encodeURIComponent(datasetId)}/raw`;
}

function encodeOutputPath(value: string): string {
  return value
    .replace(/\\/g, "/")
    .replace(/^outputs\//, "")
    .split("/")
    .filter(Boolean)
    .map(encodeURIComponent)
    .join("/");
}

export function outputRawUrl(path: string): string {
  return `${API_BASE}/outputs/${encodeOutputPath(path)}/raw`;
}

function isOutputsPath(path: string): boolean {
  return /^outputs[\\/]/.test(path);
}

export function savedMediaRawUrl(path: string): string {
  if (isOutputsPath(path)) {
    return outputRawUrl(path);
  }
  return `${API_BASE}/artifacts/raw?path=${encodeURIComponent(path)}`;
}

export interface OutputSummary {
  id: string;
  name: string;
  path: string;
  format: string;
  kind: Exclude<AssetKind, "table">;
  size_bytes: number;
  modified_at: string;
}

export interface OutputDeleteResult {
  status: "deleted";
  output: OutputSummary;
}

export interface OutputActionResponse {
  status: "opened" | "revealed";
}

export interface FolderPickResponse {
  path: string | null;
}

export function fetchOutputs(limit = 200): Promise<OutputSummary[]> {
  return requestJson<OutputSummary[]>(`/outputs?limit=${limit}`);
}

export function deleteOutput(outputId: string): Promise<OutputDeleteResult> {
  return requestJson<OutputDeleteResult>(
    `/outputs/${encodeOutputPath(outputId)}`,
    { method: "DELETE" },
  );
}

export function openOutput(outputId: string): Promise<OutputActionResponse> {
  return requestJson<OutputActionResponse>(
    `/outputs/${encodeOutputPath(outputId)}/open`,
    { method: "POST" },
  );
}

export function revealOutput(outputId: string): Promise<OutputActionResponse> {
  return requestJson<OutputActionResponse>(
    `/outputs/${encodeOutputPath(outputId)}/reveal`,
    { method: "POST" },
  );
}

export async function openSavedOutput(
  path: string,
): Promise<OutputActionResponse> {
  if (isOutputsPath(path)) {
    return openOutput(path);
  }
  return requestJson<OutputActionResponse>("/artifacts/open", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

export async function revealSavedOutput(
  path: string,
): Promise<OutputActionResponse> {
  if (isOutputsPath(path)) {
    return revealOutput(path);
  }
  return requestJson<OutputActionResponse>("/artifacts/reveal", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

export async function pickFolder(initial?: string): Promise<string | null> {
  const response = await requestJson<FolderPickResponse>("/fs/pick-folder", {
    method: "POST",
    body: JSON.stringify({ initial: initial ?? null }),
  });
  return response.path;
}

export interface ModelSummary {
  id: string;
  name: string;
  path: string;
  model_type: string;
  format: string;
  size_bytes: number;
  modified_at: string;
  mtime_ns: number;
  fingerprint: string;
  executable: boolean;
  is_directory: boolean;
  parameter_count?: number | null;
  quantization?: ModelQuantizationInfo | null;
  memory?: ModelMemoryEstimate | null;
}

export interface WorkflowModelEstimate {
  node_id: string;
  widget: string;
  value: string;
  model_id?: string | null;
  model_type?: string | null;
  memory?: ModelMemoryEstimate | null;
  error?: string | null;
}

export interface WorkflowMemoryEstimate {
  models: WorkflowModelEstimate[];
  weights_bytes: number;
  resident_estimate_bytes: number;
  offload_ram_estimate_bytes: number;
  unknown: boolean;
  warnings: string[];
}

export function estimateWorkflowMemory(
  workflow: WorkflowGraphPayload,
): Promise<WorkflowMemoryEstimate> {
  return requestJson<WorkflowMemoryEstimate>("/estimate/workflow", {
    method: "POST",
    body: JSON.stringify({ workflow }),
  });
}

export interface ModelTypeSummary {
  name: string;
  path: string;
  count: number;
  default: boolean;
}

export interface DatasetPreview {
  dataset: DatasetSummary;
  columns: string[];
  rows: unknown[][];
  truncated: boolean;
  row_limit: number;
}

export interface DatasetUploadResult {
  status: "uploaded";
  dataset: DatasetSummary;
}

export type AcceleratorMode = "auto" | "cpu" | "cuda";

export interface CpuMetrics {
  utilization: number | null;
  cores: number | null;
}

export interface MemoryMetrics {
  total_bytes: number;
  used_bytes: number;
  utilization: number;
}

export interface GpuMetrics {
  index: number;
  name: string;
  utilization: number | null;
  vram_total_bytes: number | null;
  vram_used_bytes: number | null;
  temperature_c: number | null;
}

export interface SystemMetrics {
  cpu: CpuMetrics;
  memory: MemoryMetrics;
  gpus: GpuMetrics[];
  timestamp: string;
}

export interface RuntimeLimits {
  node_timeout_seconds: number;
  run_timeout_seconds: number;
  max_memory_mb: number;
}

export interface OptimizationHealth {
  attention: {
    configured: string;
    benchmark?: boolean;
    backends: string[];
    packages: Record<string, string | null>;
  };
  precision: {
    default: string;
    fp16_accumulation: boolean;
    quantization_formats: string[];
    quantized_runtime: boolean;
  };
  memory: {
    profile: string;
    allocator: string;
    alloc_conf: string | null;
    pinned_memory: boolean;
  };
  compile: {
    mode: string;
    cache_dir: string | null;
  };
}

export interface QuantizationFormatHealth {
  label: string;
  bits: number | null;
  runtime: string;
  package: string | null;
  installed: boolean;
  components: string[];
  notes: string;
}

export interface QuantizationHealth {
  formats: Record<string, QuantizationFormatHealth>;
}

export interface HealthInfo {
  status: string;
  version: string;
  templates: number;
  workflows: number;
  models_root: string;
  accelerator: AcceleratorMode;
  effective_accelerator: "cpu" | "cuda";
  torch: string | null;
  torch_cuda: boolean;
  nvidia: boolean;
  limits: RuntimeLimits;
  optimizations?: OptimizationHealth;
  quantization?: QuantizationHealth;
}

export function fetchSystemMetrics(signal?: AbortSignal): Promise<SystemMetrics> {
  return requestJson<SystemMetrics>("/system/metrics", { signal });
}

export function fetchHealth(): Promise<HealthInfo> {
  return requestJson<HealthInfo>("/health");
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (cause) {
    throw new ApiError(0, "Backend unreachable — is uvicorn running?", cause);
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, describeError(response.status, body), body);
  }
  return body as T;
}

function describeError(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return `Request failed with status ${status}`;
}

export function parseCode(code: string, nodeId = ""): Promise<NodeSchema> {
  return requestJson<NodeSchema>("/parse", {
    method: "POST",
    body: JSON.stringify({ code, node_id: nodeId }),
  });
}

export interface TemplateCreatePayload {
  name: string;
  title: string;
  category: string;
  description: string;
  code: string;
  overwrite?: boolean;
}

export function fetchTemplates(): Promise<TemplateInfo[]> {
  return requestJson<TemplateInfo[]>("/templates");
}

export function createTemplate(
  payload: TemplateCreatePayload,
): Promise<TemplateInfo> {
  return requestJson<TemplateInfo>("/templates", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteTemplate(name: string): Promise<{ status: string; name: string }> {
  return requestJson(`/templates/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export function fetchWorkflowTemplates(): Promise<WorkflowTemplateInfo[]> {
  return requestJson<WorkflowTemplateInfo[]>("/workflows");
}

export function approveNode(
  nodeId: string,
  codeHash: string,
): Promise<{ status: string; node_id: string; code_hash: string }> {
  return requestJson("/approve", {
    method: "POST",
    body: JSON.stringify({ node_id: nodeId, code_hash: codeHash }),
  });
}

export function executeGraph(
  graph: WorkflowGraphPayload,
  targets?: string[],
  nodeLabels?: Record<string, string>,
): Promise<ExecuteResponse> {
  return requestJson<ExecuteResponse>("/execute", {
    method: "POST",
    body: JSON.stringify({
      graph,
      targets: targets ?? null,
      node_labels: nodeLabels ?? null,
    }),
  });
}

export function exportWorkflow(
  graph: WorkflowGraphPayload,
  format: ExportFormat,
  targets?: string[],
): Promise<ExportedWorkflow> {
  return requestJson<ExportedWorkflow>("/export", {
    method: "POST",
    body: JSON.stringify({ graph, format, targets: targets ?? null }),
  });
}

export function fetchProjects(): Promise<ProjectSummary[]> {
  return requestJson<ProjectSummary[]>("/projects");
}

export function createProject(
  name: string,
  workflow: SavedWorkflow,
  overwrite = false,
): Promise<ProjectSaveResult> {
  return requestJson<ProjectSaveResult>("/projects", {
    method: "POST",
    body: JSON.stringify({ name, workflow, overwrite }),
  });
}

export function updateProject(
  slug: string,
  name: string,
  workflow: SavedWorkflow,
): Promise<ProjectSaveResult> {
  return requestJson<ProjectSaveResult>(`/projects/${encodeURIComponent(slug)}`, {
    method: "PUT",
    body: JSON.stringify({ name, workflow }),
  });
}

export function deleteProject(slug: string): Promise<{ status: string; slug: string }> {
  return requestJson(`/projects/${encodeURIComponent(slug)}`, {
    method: "DELETE",
  });
}

export function fetchProject(slug: string): Promise<ProjectDetail> {
  return requestJson<ProjectDetail>(`/projects/${encodeURIComponent(slug)}`);
}

export function fetchProjectVersions(
  slug: string,
): Promise<ProjectVersionInfo[]> {
  return requestJson<ProjectVersionInfo[]>(
    `/projects/${encodeURIComponent(slug)}/versions`,
  );
}

export function fetchProjectVersion(
  slug: string,
  version: string,
): Promise<ProjectVersionDetail> {
  return requestJson<ProjectVersionDetail>(
    `/projects/${encodeURIComponent(slug)}/versions/${encodeURIComponent(version)}`,
  );
}

export function fetchWorkspaceSession(): Promise<WorkspaceSession | null> {
  return requestJson<WorkspaceSession | null>("/session");
}

export function saveWorkspaceSession(
  workflow: SavedWorkflow,
  projectSlug: string | null,
  projectName: string | null,
  options: { keepalive?: boolean } = {},
): Promise<void> {
  const keepalive = options.keepalive ?? false;
  const body = JSON.stringify({
    workflow,
    project_slug: projectSlug,
    project_name: projectName,
  });
  if (keepalive) {
    try {
      void fetch(`${API_BASE}/session`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body,
        keepalive: true,
      }).catch(() => undefined);
    } catch {
      // The localStorage autosave remains the primary recovery path.
    }
    return Promise.resolve();
  }
  return requestJson("/session", { method: "PUT", body }).then(() => undefined);
}

export function fetchDatasets(): Promise<DatasetSummary[]> {
  return requestJson<DatasetSummary[]>("/uploads");
}

export function fetchModelTypes(): Promise<ModelTypeSummary[]> {
  return requestJson<ModelTypeSummary[]>("/models/types");
}

export function fetchModels(modelType?: string | null): Promise<ModelSummary[]> {
  const query = modelType ? `?type=${encodeURIComponent(modelType)}` : "";
  return requestJson<ModelSummary[]>(`/models${query}`);
}

export function fetchDatasetPreview(
  datasetId: string,
  limit = 20,
): Promise<DatasetPreview> {
  return requestJson<DatasetPreview>(
    `/uploads/${encodeURIComponent(datasetId)}/preview?limit=${limit}`,
  );
}

export function deleteDataset(
  datasetId: string,
): Promise<{ status: string; dataset: DatasetSummary }> {
  return requestJson(`/uploads/${encodeURIComponent(datasetId)}`, {
    method: "DELETE",
  });
}

export async function uploadDataset(file: File): Promise<DatasetUploadResult> {
  const form = new FormData();
  form.append("file", file);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/uploads`, {
      method: "POST",
      body: form,
    });
  } catch (cause) {
    throw new ApiError(0, "Backend unreachable — is uvicorn running?", cause);
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, describeError(response.status, body), body);
  }
  return body as DatasetUploadResult;
}

export function cancelExecution(): Promise<{ status: string }> {
  return requestJson<{ status: string }>("/cancel", { method: "POST" });
}

export function fetchRunStatus(): Promise<BackendRunStatus> {
  return requestJson<BackendRunStatus>("/status");
}

export function fetchRuns(limit = 50): Promise<RunSummary[]> {
  return requestJson<RunSummary[]>(`/runs?limit=${limit}`);
}

export function fetchRun(runId: string): Promise<RunDetail> {
  return requestJson<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
}

export function deleteRun(runId: string): Promise<{ status: string; run_id: string }> {
  return requestJson(`/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
}

export function untrustedNodesFrom(error: ApiError): UntrustedNodeRef[] {
  const detail = error.detail as { detail?: unknown } | null;
  const inner = (detail as { detail?: unknown } | null)?.detail;
  if (inner && typeof inner === "object" && "untrusted_nodes" in inner) {
    const nodes = (inner as { untrusted_nodes: unknown }).untrusted_nodes;
    if (Array.isArray(nodes)) {
      return nodes as UntrustedNodeRef[];
    }
  }
  return [];
}

export interface ModelSelectionIssue {
  node_id: string;
  widget: string;
  model_type: string;
  path: string;
  message: string;
}

export function modelSelectionIssuesFrom(error: ApiError): ModelSelectionIssue[] {
  const outer = error.detail as { detail?: unknown } | null;
  const inner = outer?.detail;
  if (
    inner &&
    typeof inner === "object" &&
    "type" in inner &&
    (inner as { type?: unknown }).type === "ModelSelection" &&
    "models" in inner &&
    Array.isArray((inner as { models?: unknown }).models)
  ) {
    return (inner as { models: ModelSelectionIssue[] }).models;
  }
  return [];
}

export function syntaxErrorFrom(error: ApiError): {
  nodeId: string;
  line: number;
  message: string;
} | null {
  const outer = error.detail as { detail?: unknown } | null;
  const detail = outer?.detail;
  if (
    detail &&
    typeof detail === "object" &&
    "syntax_errors" in detail &&
    Array.isArray((detail as { syntax_errors: unknown[] }).syntax_errors)
  ) {
    const errors = (detail as { syntax_errors: { line: number; message: string }[] })
      .syntax_errors;
    const first = errors[0];
    if (first) {
      const nodeId = (detail as { node_id?: string }).node_id ?? "";
      return { nodeId, line: first.line, message: first.message };
    }
  }
  return null;
}
