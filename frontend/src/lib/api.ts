import type {
  BackendRunStatus,
  ExecuteResponse,
  NodeSchema,
  TemplateInfo,
  UntrustedNodeRef,
  WorkflowGraphPayload,
  WorkflowTemplateInfo,
} from "../types/graph";

const API_BASE = "/api";

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

export function fetchTemplates(): Promise<TemplateInfo[]> {
  return requestJson<TemplateInfo[]>("/templates");
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
): Promise<ExecuteResponse> {
  return requestJson<ExecuteResponse>("/execute", {
    method: "POST",
    body: JSON.stringify({ graph, targets: targets ?? null }),
  });
}

export function cancelExecution(): Promise<{ status: string }> {
  return requestJson<{ status: string }>("/cancel", { method: "POST" });
}

export function fetchRunStatus(): Promise<BackendRunStatus> {
  return requestJson<BackendRunStatus>("/status");
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
