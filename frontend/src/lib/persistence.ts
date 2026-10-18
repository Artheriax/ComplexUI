/**
 * Workflow persistence: `.cuigraph.json` documents, localStorage autosave
 * and file import/export. Schemas are intentionally NOT stored — they are
 * re-parsed from code on load so a restored graph always matches the current
 * transpiler and trust registry.
 */
import type { AppEdge, AppNode } from "../types/graph";

export const WORKFLOW_FORMAT_VERSION = 1;
export const STORAGE_KEY = "complexui.workflow.v1";
export const AUTOSAVE_DELAY_MS = 800;

export interface SavedNode {
  id: string;
  label: string;
  position: { x: number; y: number };
  code: string;
  widgetValues: Record<string, unknown>;
  templateName: string | null;
}

export interface SavedEdge {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

export interface SavedWorkflow {
  version: number;
  savedAt: string;
  nodes: SavedNode[];
  edges: SavedEdge[];
}

export function serialiseWorkflow(
  nodes: AppNode[],
  edges: AppEdge[],
): SavedWorkflow {
  return {
    version: WORKFLOW_FORMAT_VERSION,
    savedAt: new Date().toISOString(),
    nodes: nodes.map((node) => ({
      id: node.id,
      label: node.data.label,
      position: { x: node.position.x, y: node.position.y },
      code: node.data.code,
      widgetValues: node.data.widgetValues,
      templateName: node.data.templateName,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      sourceHandle: edge.sourceHandle ?? "",
      target: edge.target,
      targetHandle: edge.targetHandle ?? "",
    })),
  };
}

export function parseWorkflowJson(text: string): SavedWorkflow {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error("Workflow file is not valid JSON.");
  }
  return validateWorkflow(raw);
}

function validateWorkflow(value: unknown): SavedWorkflow {
  if (typeof value !== "object" || value === null) {
    throw new Error("Workflow file must contain a JSON object.");
  }
  const record = value as Record<string, unknown>;
  if (record.version !== WORKFLOW_FORMAT_VERSION) {
    throw new Error(
      `Unsupported workflow version: ${JSON.stringify(record.version)}`,
    );
  }
  if (!Array.isArray(record.nodes) || !Array.isArray(record.edges)) {
    throw new Error("Workflow file is missing the nodes/edges arrays.");
  }
  return {
    version: WORKFLOW_FORMAT_VERSION,
    savedAt: typeof record.savedAt === "string" ? record.savedAt : "",
    nodes: record.nodes.map(validateNode),
    edges: record.edges.map(validateEdge),
  };
}

function validateNode(value: unknown): SavedNode {
  if (typeof value !== "object" || value === null) {
    throw new Error("Workflow contains an invalid node entry.");
  }
  const record = value as Record<string, unknown>;
  const position = record.position as Record<string, unknown> | undefined;
  if (
    typeof record.id !== "string" ||
    typeof record.code !== "string" ||
    !position ||
    typeof position.x !== "number" ||
    typeof position.y !== "number"
  ) {
    throw new Error(
      `Workflow node entries require id/code/position: ${JSON.stringify(value).slice(0, 80)}`,
    );
  }
  return {
    id: record.id,
    label:
      typeof record.label === "string" && record.label.length > 0
        ? record.label
        : "Node",
    position: { x: position.x, y: position.y },
    code: record.code,
    widgetValues:
      typeof record.widgetValues === "object" &&
      record.widgetValues !== null &&
      !Array.isArray(record.widgetValues)
        ? (record.widgetValues as Record<string, unknown>)
        : {},
    templateName:
      typeof record.templateName === "string" ? record.templateName : null,
  };
}

function validateEdge(value: unknown): SavedEdge {
  if (typeof value !== "object" || value === null) {
    throw new Error("Workflow contains an invalid edge entry.");
  }
  const record = value as Record<string, unknown>;
  if (
    typeof record.source !== "string" ||
    typeof record.target !== "string" ||
    typeof record.sourceHandle !== "string" ||
    typeof record.targetHandle !== "string"
  ) {
    throw new Error("Workflow edge entries require string endpoints/handles.");
  }
  const id =
    typeof record.id === "string"
      ? record.id
      : `${record.source}:${record.sourceHandle}->${record.target}:${record.targetHandle}`;
  return {
    id,
    source: record.source,
    sourceHandle: record.sourceHandle,
    target: record.target,
    targetHandle: record.targetHandle,
  };
}

interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

const memoryStore = new Map<string, string>();

function storage(): KeyValueStorage {
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      return window.localStorage;
    }
  } catch {
    // Accessing localStorage can throw in sandboxed iframes; fall back.
  }
  return {
    getItem: (key) => memoryStore.get(key) ?? null,
    setItem: (key, value) => {
      memoryStore.set(key, value);
    },
    removeItem: (key) => {
      memoryStore.delete(key);
    },
  };
}

export function saveLocalWorkflow(saved: SavedWorkflow): void {
  try {
    storage().setItem(STORAGE_KEY, JSON.stringify(saved));
  } catch {
    // Quota exceeded or storage disabled — autosave is best effort.
  }
}

export function loadLocalWorkflow(): SavedWorkflow | null {
  try {
    const raw = storage().getItem(STORAGE_KEY);
    return raw === null ? null : parseWorkflowJson(raw);
  } catch {
    return null;
  }
}

export function clearLocalWorkflow(): void {
  try {
    storage().removeItem(STORAGE_KEY);
  } catch {
    // Nothing to clean up when storage is unavailable.
  }
}

export function downloadWorkflow(saved: SavedWorkflow): void {
  const blob = new Blob([JSON.stringify(saved, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `complexui-${saved.savedAt.replace(/[:.]/g, "-")}.cuigraph.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export async function readWorkflowFile(file: File): Promise<SavedWorkflow> {
  return parseWorkflowJson(await file.text());
}
