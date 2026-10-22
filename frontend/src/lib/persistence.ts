/**
 * Workflow persistence: `.cuigraph.json` documents, localStorage autosave
 * and file import/export. Schemas are intentionally NOT stored — they are
 * re-parsed from code on load so a restored graph always matches the current
 * transpiler and trust registry.
 */
import type {
  AppEdge,
  AppNode,
  InputSocket,
  NodeColor,
  OutputSocket,
  SocketType,
  SubgraphMap,
} from "../types/graph";
import {
  isFrameNode,
  isNodeColor,
  isSubgraphInstanceNode,
  isSubgraphManagerNode,
  SOCKET_TYPE_NAMES,
  SUBGRAPH_INPUT_NODE_ID,
  SUBGRAPH_OUTPUT_NODE_ID,
} from "../types/graph";

export const WORKFLOW_FORMAT_VERSION = 1;
export const STORAGE_KEY = "complexui.workflow.v1";
export const WORKSPACE_STORAGE_KEY = "complexui.workspace.v1";
export const PROJECT_STORAGE_KEY = "complexui.project.v1";
export const MAX_WORKSPACE_WINDOWS = 24;
export const AUTOSAVE_DELAY_MS = 800;
export const MAX_WORKFLOW_NODES = 200;
export const MAX_WORKFLOW_EDGES = 1_000;
export const MAX_NODE_CODE_CHARS = 200_000;
export const MAX_PROJECT_NAME_CHARS = 80;
export const MAX_SUBGRAPHS = 50;
const MAX_IDENTIFIER_CHARS = 200;
export const MAX_FRAME_DIMENSION = 100_000;

export interface SavedNode {
  id: string;
  label: string;
  position: { x: number; y: number };
  code: string;
  widgetValues: Record<string, unknown>;
  templateName: string | null;
  color?: NodeColor;
  isFrame?: boolean;
  width?: number;
  height?: number;
  parentId?: string;
  isSubgraph?: boolean;
  subgraphId?: string;
}

export interface SavedEdge {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

export interface SavedSubgraph {
  id: string;
  name: string;
  inputs: InputSocket[];
  outputs: OutputSocket[];
  nodes: SavedNode[];
  edges: SavedEdge[];
  inputPosition: { x: number; y: number };
  outputPosition: { x: number; y: number };
}

export interface SavedWorkflow {
  name?: string | null;
  version: number;
  savedAt: string;
  nodes: SavedNode[];
  edges: SavedEdge[];
  subgraphs?: Record<string, SavedSubgraph>;
}

export interface SavedWorkspaceWindow {
  id: string;
  title: string;
  projectName: string | null;
  projectSlug: string | null;
  workflow: SavedWorkflow;
}

export interface SavedWorkspace {
  version: 1;
  activeWindowId: string | null;
  windows: SavedWorkspaceWindow[];
}

export interface LocalProjectIdentity {
  name: string | null;
  slug: string | null;
}

function serialiseNode(node: AppNode): SavedNode {
  const saved: SavedNode = {
    id: node.id,
    label: node.data.label,
    position: { x: node.position.x, y: node.position.y },
    code: node.data.code,
    widgetValues: node.data.widgetValues,
    templateName: node.data.templateName,
    color: node.data.color,
  };
  if (isFrameNode(node)) {
    saved.isFrame = true;
    saved.width = node.width ?? node.measured?.width;
    saved.height = node.height ?? node.measured?.height;
  } else if (!isSubgraphManagerNode(node)) {
    if (node.width !== undefined) {
      saved.width = node.width;
    }
    if (node.height !== undefined) {
      saved.height = node.height;
    }
  }
  if (isSubgraphInstanceNode(node) && node.data.subgraphId) {
    saved.isSubgraph = true;
    saved.subgraphId = node.data.subgraphId;
  }
  if (node.parentId) {
    saved.parentId = node.parentId;
  }
  return saved;
}

function serialiseEdges(edges: AppEdge[]): SavedEdge[] {
  return edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    sourceHandle: edge.sourceHandle ?? "",
    target: edge.target,
    targetHandle: edge.targetHandle ?? "",
  }));
}

function serialiseSubgraphs(
  subgraphs: SubgraphMap,
): Record<string, SavedSubgraph> {
  return Object.fromEntries(
    Object.entries(subgraphs).map(([id, definition]) => [
      id,
      {
        id: definition.id,
        name: definition.name,
        inputs: definition.inputs,
        outputs: definition.outputs,
        nodes: stripManagers(definition.nodes).map(serialiseNode),
        edges: serialiseEdges(definition.edges),
        inputPosition: definition.inputPosition,
        outputPosition: definition.outputPosition,
      },
    ]),
  );
}

function stripManagers(nodes: AppNode[]): AppNode[] {
  return nodes.filter((node) => !isSubgraphManagerNode(node));
}

export function serialiseWorkflow(
  nodes: AppNode[],
  edges: AppEdge[],
  name?: string | null,
  subgraphs?: SubgraphMap,
): SavedWorkflow {
  const savedSubgraphs = subgraphs ? serialiseSubgraphs(subgraphs) : {};
  return {
    ...(name ? { name } : {}),
    version: WORKFLOW_FORMAT_VERSION,
    savedAt: new Date().toISOString(),
    nodes: stripManagers(nodes).map(serialiseNode),
    edges: serialiseEdges(edges),
    ...(Object.keys(savedSubgraphs).length > 0
      ? { subgraphs: savedSubgraphs }
      : {}),
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
  if (record.nodes.length > MAX_WORKFLOW_NODES) {
    throw new Error(
      `Workflow has too many nodes (${record.nodes.length} > ${MAX_WORKFLOW_NODES}).`,
    );
  }
  if (record.edges.length > MAX_WORKFLOW_EDGES) {
    throw new Error(
      `Workflow has too many edges (${record.edges.length} > ${MAX_WORKFLOW_EDGES}).`,
    );
  }

  const nodes = record.nodes.map(validateNode);
  const edges = record.edges.map(validateEdge);
  const subgraphs = validateSubgraphs(record.subgraphs);

  validateGraphStructure(nodes, edges, subgraphs, null);
  for (const definition of Object.values(subgraphs)) {
    validateGraphStructure(
      definition.nodes,
      definition.edges,
      subgraphs,
      definition,
    );
  }
  validateSubgraphReferences(nodes, subgraphs);

  const name =
    typeof record.name === "string" && record.name.trim().length > 0
      ? record.name.trim().slice(0, MAX_PROJECT_NAME_CHARS)
      : undefined;

  return {
    ...(name ? { name } : {}),
    version: WORKFLOW_FORMAT_VERSION,
    savedAt: typeof record.savedAt === "string" ? record.savedAt : "",
    nodes,
    edges,
    ...(Object.keys(subgraphs).length > 0 ? { subgraphs } : {}),
  };
}

function validatePosition(
  value: unknown,
  fallback: { x: number; y: number },
): { x: number; y: number } {
  if (typeof value !== "object" || value === null) {
    return fallback;
  }
  const record = value as Record<string, unknown>;
  return typeof record.x === "number" &&
    Number.isFinite(record.x) &&
    typeof record.y === "number" &&
    Number.isFinite(record.y)
    ? { x: record.x, y: record.y }
    : fallback;
}

function validateSockets(value: unknown, label: string): OutputSocket[] {
  if (!Array.isArray(value)) {
    return [];
  }
  if (value.length > MAX_WORKFLOW_NODES) {
    throw new Error(`${label} contains too many sockets.`);
  }
  const names = new Set<string>();
  return value.map((item) => {
    if (typeof item !== "object" || item === null) {
      throw new Error(`${label} contains an invalid socket.`);
    }
    const record = item as Record<string, unknown>;
    if (
      typeof record.name !== "string" ||
      record.name.length === 0 ||
      record.name.length > MAX_IDENTIFIER_CHARS ||
      typeof record.data_type !== "string" ||
      !SOCKET_TYPE_NAMES.has(record.data_type as SocketType)
    ) {
      throw new Error(`${label} contains an invalid socket.`);
    }
    if (names.has(record.name)) {
      throw new Error(`${label} contains duplicate socket '${record.name}'.`);
    }
    names.add(record.name);
    return {
      name: record.name,
      data_type: record.data_type as SocketType,
    };
  });
}

function validateSubgraphs(raw: unknown): Record<string, SavedSubgraph> {
  if (raw === undefined || raw === null) {
    return {};
  }
  if (typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("Workflow subgraphs must be an object keyed by id.");
  }
  const entries = Object.entries(raw as Record<string, unknown>);
  if (entries.length > MAX_SUBGRAPHS) {
    throw new Error(
      `Workflow has too many subgraphs (${entries.length} > ${MAX_SUBGRAPHS}).`,
    );
  }
  const subgraphs: Record<string, SavedSubgraph> = {};
  for (const [key, value] of entries) {
    if (typeof value !== "object" || value === null) {
      throw new Error(`Subgraph '${key}' is not a valid object.`);
    }
    const record = value as Record<string, unknown>;
    const id =
      typeof record.id === "string" && record.id.length > 0
        ? record.id
        : key;
    if (id !== key || id.length > MAX_IDENTIFIER_CHARS) {
      throw new Error(`Subgraph '${key}' has an invalid id.`);
    }
    if (!Array.isArray(record.nodes) || !Array.isArray(record.edges)) {
      throw new Error(`Subgraph '${key}' is missing nodes/edges arrays.`);
    }
    if (record.nodes.length > MAX_WORKFLOW_NODES) {
      throw new Error(`Subgraph '${key}' has too many nodes.`);
    }
    if (record.edges.length > MAX_WORKFLOW_EDGES) {
      throw new Error(`Subgraph '${key}' has too many edges.`);
    }
    const nodes = record.nodes.map(validateNode);
    subgraphs[key] = {
      id,
      name:
        typeof record.name === "string" && record.name.trim().length > 0
          ? record.name.trim().slice(0, MAX_PROJECT_NAME_CHARS)
          : `Subgraph ${key.slice(0, 8)}`,
      inputs: validateSockets(record.inputs, `Subgraph '${key}' inputs`),
      outputs: validateSockets(record.outputs, `Subgraph '${key}' outputs`),
      nodes,
      edges: record.edges.map(validateEdge),
      inputPosition: validatePosition(record.inputPosition, { x: -260, y: 0 }),
      outputPosition: validatePosition(record.outputPosition, { x: 300, y: 0 }),
    };
  }
  return subgraphs;
}

function validateGraphStructure(
  nodes: SavedNode[],
  edges: SavedEdge[],
  subgraphs: Record<string, SavedSubgraph>,
  definition: SavedSubgraph | null,
): void {
  const scope = definition ? `Subgraph '${definition.id}'` : "Workflow";
  const nodeIds = new Set<string>();
  for (const node of nodes) {
    if (nodeIds.has(node.id)) {
      throw new Error(`${scope} contains duplicate node id '${node.id}'.`);
    }
    nodeIds.add(node.id);
  }
  const nodesById = new Map(nodes.map((node) => [node.id, node]));
  const allowedEndpoints = new Set(nodeIds);
  if (definition) {
    allowedEndpoints.add(SUBGRAPH_INPUT_NODE_ID);
    allowedEndpoints.add(SUBGRAPH_OUTPUT_NODE_ID);
  }

  for (const node of nodes) {
    if (node.parentId && !nodeIds.has(node.parentId)) {
      throw new Error(
        `${scope} node '${node.id}' references a missing parent '${node.parentId}'.`,
      );
    }
    if (node.parentId === node.id) {
      throw new Error(`${scope} node '${node.id}' cannot be its own parent.`);
    }
    if (node.parentId && !nodesById.get(node.parentId)?.isFrame) {
      throw new Error(
        `${scope} node '${node.id}' must be parented to a frame.`,
      );
    }
    if (node.isFrame && node.isSubgraph) {
      throw new Error(`${scope} node '${node.id}' has conflicting types.`);
    }
    if (node.isSubgraph) {
      const target = subgraphs[node.subgraphId ?? ""];
      if (!target) {
        throw new Error(
          `${scope} node '${node.id}' references missing subgraph '${node.subgraphId ?? ""}'.`,
        );
      }
    }
  }

  const frameIds = new Set(
    nodes.filter((node) => node.isFrame).map((node) => node.id),
  );
  const edgeIds = new Set<string>();
  const inputNames = new Set(definition?.inputs.map((socket) => socket.name));
  const outputNames = new Set(
    definition?.outputs.map((socket) => socket.name),
  );
  for (const edge of edges) {
    if (edgeIds.has(edge.id)) {
      throw new Error(`${scope} contains duplicate edge id '${edge.id}'.`);
    }
    edgeIds.add(edge.id);
    if (frameIds.has(edge.source) || frameIds.has(edge.target)) {
      throw new Error("Workflow edges cannot reference frame nodes.");
    }
    if (
      !allowedEndpoints.has(edge.source) ||
      !allowedEndpoints.has(edge.target)
    ) {
      throw new Error(
        `${scope} edge '${edge.id}' references an unknown node.`,
      );
    }
    if (definition) {
      if (edge.target === SUBGRAPH_INPUT_NODE_ID) {
        throw new Error("Subgraph input managers cannot receive edges.");
      }
      if (edge.source === SUBGRAPH_OUTPUT_NODE_ID) {
        throw new Error("Subgraph output managers cannot emit edges.");
      }
      if (edge.source === SUBGRAPH_INPUT_NODE_ID) {
        if (!inputNames.has(edge.sourceHandle)) {
          throw new Error(
            `Subgraph '${definition.id}' edge references unknown input '${edge.sourceHandle}'.`,
          );
        }
        if (edge.target === SUBGRAPH_OUTPUT_NODE_ID) {
          throw new Error(
            "Subgraph inputs cannot connect directly to outputs.",
          );
        }
      }
      if (edge.target === SUBGRAPH_OUTPUT_NODE_ID) {
        if (!outputNames.has(edge.targetHandle)) {
          throw new Error(
            `Subgraph '${definition.id}' edge references unknown output '${edge.targetHandle}'.`,
          );
        }
      }
    }
    const sourceNode = nodesById.get(edge.source);
    const targetNode = nodesById.get(edge.target);
    if (sourceNode?.isSubgraph) {
      const ports = subgraphs[sourceNode.subgraphId ?? ""]?.outputs ?? [];
      if (!ports.some((socket) => socket.name === edge.sourceHandle)) {
        throw new Error(
          `${scope} edge '${edge.id}' references unknown subgraph output '${edge.sourceHandle}'.`,
        );
      }
    }
    if (targetNode?.isSubgraph) {
      const ports = subgraphs[targetNode.subgraphId ?? ""]?.inputs ?? [];
      if (!ports.some((socket) => socket.name === edge.targetHandle)) {
        throw new Error(
          `${scope} edge '${edge.id}' references unknown subgraph input '${edge.targetHandle}'.`,
        );
      }
    }
  }
}

function validateSubgraphReferences(
  rootNodes: SavedNode[],
  subgraphs: Record<string, SavedSubgraph>,
): void {
  const visiting = new Set<string>();
  const visited = new Set<string>();

  const visit = (definitionId: string): void => {
    if (visited.has(definitionId)) {
      return;
    }
    if (visiting.has(definitionId)) {
      throw new Error(
        `Workflow contains a subgraph cycle involving '${definitionId}'.`,
      );
    }
    visiting.add(definitionId);
    const definition = subgraphs[definitionId];
    for (const node of definition?.nodes ?? []) {
      if (node.isSubgraph) {
        const child = node.subgraphId ?? "";
        if (!subgraphs[child]) {
          throw new Error(
            `Subgraph '${definitionId}' references missing subgraph '${child}'.`,
          );
        }
        visit(child);
      }
    }
    visiting.delete(definitionId);
    visited.add(definitionId);
  };

  for (const definitionId of Object.keys(subgraphs)) {
    visit(definitionId);
  }
  for (const node of rootNodes) {
    if (node.isSubgraph && !subgraphs[node.subgraphId ?? ""]) {
      throw new Error(
        `Workflow node '${node.id}' references missing subgraph '${node.subgraphId ?? ""}'.`,
      );
    }
  }
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
  if (record.id.length === 0 || record.id.length > MAX_IDENTIFIER_CHARS) {
    throw new Error(
      `Workflow node id must be 1-${MAX_IDENTIFIER_CHARS} characters.`,
    );
  }
  if (record.code.length > MAX_NODE_CODE_CHARS) {
    throw new Error(
      `Workflow node '${record.id}' exceeds the ${MAX_NODE_CODE_CHARS}-character code limit.`,
    );
  }
  if (!Number.isFinite(position.x) || !Number.isFinite(position.y)) {
    throw new Error(`Workflow node '${record.id}' has a non-finite position.`);
  }
  const saved: SavedNode = {
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
    color: isNodeColor(record.color) ? record.color : "default",
  };
  if (record.isFrame === true) {
    saved.isFrame = true;
  }
  if (record.isSubgraph === true) {
    saved.isSubgraph = true;
    if (
      typeof record.subgraphId === "string" &&
      record.subgraphId.length > 0 &&
      record.subgraphId.length <= MAX_IDENTIFIER_CHARS
    ) {
      saved.subgraphId = record.subgraphId;
    }
  }
  for (const dimension of ["width", "height"] as const) {
    const value = record[dimension];
    if (
      typeof value === "number" &&
      Number.isFinite(value) &&
      value > 0 &&
      value <= MAX_FRAME_DIMENSION
    ) {
      saved[dimension] = value;
    }
  }
  if (
    typeof record.parentId === "string" &&
    record.parentId.length > 0 &&
    record.parentId.length <= MAX_IDENTIFIER_CHARS
  ) {
    saved.parentId = record.parentId;
  }
  return saved;
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
  for (const endpoint of [
    record.source,
    record.target,
    record.sourceHandle,
    record.targetHandle,
  ]) {
    if (endpoint.length === 0 || endpoint.length > MAX_IDENTIFIER_CHARS) {
      throw new Error(
        `Workflow edge endpoints/handles must be 1-${MAX_IDENTIFIER_CHARS} characters.`,
      );
    }
  }
  const id =
    typeof record.id === "string" && record.id.length > 0
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
    if (raw === null) {
      return null;
    }
    try {
      return parseWorkflowJson(raw);
    } catch {
      // Corrupt autosave payloads would otherwise fail on every reload.
      clearLocalWorkflow();
      return null;
    }
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

export function saveLocalWorkspace(workspace: SavedWorkspace): void {
  try {
    storage().setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(workspace));
  } catch {
    // Quota exceeded or storage disabled — autosave is best effort.
  }
}

function validateWorkspace(value: unknown): SavedWorkspace | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const record = value as Record<string, unknown>;
  if (record.version !== 1 || !Array.isArray(record.windows)) {
    return null;
  }
  const windows: SavedWorkspaceWindow[] = [];
  for (const entry of record.windows.slice(0, MAX_WORKSPACE_WINDOWS)) {
    if (typeof entry !== "object" || entry === null) {
      continue;
    }
    const window = entry as Record<string, unknown>;
    if (
      typeof window.id !== "string" ||
      window.id.length === 0 ||
      window.id.length > MAX_IDENTIFIER_CHARS
    ) {
      continue;
    }
    let workflow: SavedWorkflow;
    try {
      workflow = validateWorkflow(window.workflow);
    } catch {
      continue;
    }
    const title =
      typeof window.title === "string" && window.title.trim().length > 0
        ? window.title.trim().slice(0, MAX_PROJECT_NAME_CHARS)
        : "Untitled";
    const projectName =
      typeof window.projectName === "string" &&
      window.projectName.trim().length > 0
        ? window.projectName.trim().slice(0, MAX_PROJECT_NAME_CHARS)
        : null;
    const projectSlug =
      typeof window.projectSlug === "string" && window.projectSlug
        ? window.projectSlug
        : null;
    windows.push({
      id: window.id,
      title,
      projectName,
      projectSlug,
      workflow,
    });
  }
  if (windows.length === 0) {
    return null;
  }
  const activeWindowId =
    typeof record.activeWindowId === "string" &&
    windows.some((window) => window.id === record.activeWindowId)
      ? record.activeWindowId
      : windows[0]?.id ?? null;
  return { version: 1, activeWindowId, windows };
}

export function loadLocalWorkspace(): SavedWorkspace | null {
  try {
    const raw = storage().getItem(WORKSPACE_STORAGE_KEY);
    if (raw === null) {
      return null;
    }
    try {
      const parsed: unknown = JSON.parse(raw);
      const workspace = validateWorkspace(parsed);
      if (!workspace) {
        clearLocalWorkspace();
      }
      return workspace;
    } catch {
      clearLocalWorkspace();
      return null;
    }
  } catch {
    return null;
  }
}

export function clearLocalWorkspace(): void {
  try {
    storage().removeItem(WORKSPACE_STORAGE_KEY);
  } catch {
    // Nothing to clean up when storage is unavailable.
  }
}

export function saveLocalProjectIdentity(identity: LocalProjectIdentity): void {
  try {
    storage().setItem(PROJECT_STORAGE_KEY, JSON.stringify(identity));
  } catch {
    // Identity is a convenience; the workflow autosave still recovers.
  }
}

export function loadLocalProjectIdentity(): LocalProjectIdentity | null {
  try {
    const raw = storage().getItem(PROJECT_STORAGE_KEY);
    if (raw === null) {
      return null;
    }
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) {
      return null;
    }
    const record = parsed as Record<string, unknown>;
    return {
      name:
        typeof record.name === "string" && record.name.trim().length > 0
          ? record.name.trim().slice(0, MAX_PROJECT_NAME_CHARS)
          : null,
      slug: typeof record.slug === "string" && record.slug ? record.slug : null,
    };
  } catch {
    return null;
  }
}

export function clearLocalProjectIdentity(): void {
  try {
    storage().removeItem(PROJECT_STORAGE_KEY);
  } catch {
    // Nothing to clean up when storage is unavailable.
  }
}

export function downloadTextFile(
  filename: string,
  content: string,
  mediaType: string,
): void {
  const blob = new Blob([content], { type: mediaType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function downloadWorkflow(saved: SavedWorkflow): void {
  const stem = saved.name
    ? saved.name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 80) || "complexui"
    : `complexui-${saved.savedAt.replace(/[:.]/g, "-")}`;
  downloadTextFile(
    `${stem}.cuigraph.json`,
    JSON.stringify(saved, null, 2),
    "application/json",
  );
}

export async function readWorkflowFile(file: File): Promise<SavedWorkflow> {
  return parseWorkflowJson(await file.text());
}
