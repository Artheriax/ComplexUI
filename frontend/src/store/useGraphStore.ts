import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import type {
  Connection,
  EdgeChange,
  NodeChange,
  XYPosition,
} from "@xyflow/react";
import { create } from "zustand";

import {
  ApiError,
  approveNode,
  cancelExecution,
  createProject as createProjectApi,
  createTemplate as createTemplateApi,
  deleteProject as deleteProjectApi,
  executeGraph,
  fetchProject as fetchProjectApi,
  fetchProjectVersion as fetchProjectVersionApi,
  fetchProjects as fetchProjectsApi,
  fetchTemplates,
  fetchWorkflowTemplates,
  fetchWorkspaceSession,
  modelSelectionIssuesFrom,
  parseCode,
  syntaxErrorFrom,
  untrustedNodesFrom,
  updateProject as updateProjectApi,
} from "../lib/api";
import type {
  ProjectSummary,
  TemplateCreatePayload,
  WorkspaceSession,
} from "../lib/api";
import {
  idleExecution,
  pruneExecution,
  reduceExecutionEvent,
  reduceRunStatus,
} from "../lib/executionEvents";
import type { GraphExecutionState } from "../lib/executionEvents";
import { mergeWidgetValues, reconcileEdges } from "../lib/graphEditing";
import { sha256Hex } from "../lib/hash";
import { advanceSeedNodes } from "../lib/seedNodes";
import {
  loadLocalProjectIdentity,
  loadLocalWorkflow,
  loadLocalWorkspace,
  saveLocalProjectIdentity,
  saveLocalWorkflow,
  serialiseWorkflow,
} from "../lib/persistence";
import type { SavedWorkflow } from "../lib/persistence";
import {
  findCompatibleOutput,
  findCompatibleTarget,
  isInputTaken,
  resolveTargetSocket,
  socketTypesCompatible,
} from "../lib/socketCompatibility";
import type { TemplateConnectionMatch } from "../lib/socketCompatibility";
import {
  isOfficialValueNode,
  normalizeValueChoice,
  normalizeValueNodeValues,
} from "../lib/valueNodeOptions";
import {
  buildSubgraphFromSelection,
  createSubgraphInstanceNode,
  createSubgraphManagerNodes,
  flattenWorkflow,
  stripSubgraphManagers,
  subgraphDependsOn,
  syncSubgraphDefinitions,
  syncSubgraphInstanceSchemas,
  uniqueSubgraphPortName,
} from "../lib/subgraph";
import type { FlattenedWorkflow } from "../lib/subgraph";
import type {
  AppEdge,
  AppNode,
  BackendRunStatus,
  ComplexNodeData,
  ExecuteResponse,
  ExecutionEvent,
  ExecutionStatus,
  NodeColor,
  NodeSchema,
  PendingApproval,
  PendingConnection,
  SocketType,
  SubgraphDefinition,
  SubgraphMap,
  SubgraphScope,
  TemplateInfo,
  WorkflowGraphPayload,
  WorkflowTemplateInfo,
} from "../types/graph";
import {
  defaultWidgetValues,
  isFrameNode,
  isRunnableNode,
  isSubgraphInstanceNode,
  isSubgraphManagerNode,
  NEW_SUBGRAPH_INPUT_HANDLE,
  NEW_SUBGRAPH_OUTPUT_HANDLE,
  SUBGRAPH_INPUT_NODE_ID,
  SUBGRAPH_OUTPUT_NODE_ID,
} from "../types/graph";

export type { GraphExecutionState };

export interface GraphSnapshot {
  nodes: AppNode[];
  edges: AppEdge[];
  subgraphs: SubgraphMap;
  scopeStack: SubgraphScope[];
  rootGraph: RootGraph | null;
}

export interface RootGraph {
  nodes: AppNode[];
  edges: AppEdge[];
}

export interface WorkflowWindowState {
  nodes: AppNode[];
  edges: AppEdge[];
  subgraphs: SubgraphMap;
  scopeStack: SubgraphScope[];
  rootGraph: RootGraph | null;
  past: GraphSnapshot[];
  future: GraphSnapshot[];
  execution: GraphExecutionState;
  projectName: string | null;
  projectSlug: string | null;
}

export interface WorkflowWindow extends WorkflowWindowState {
  id: string;
  title: string;
}

interface HistoryMeta {
  coalesceKey: string | null;
  coalesceAt: number;
  dragging: boolean;
}

const HISTORY_LIMIT = 100;
const HISTORY_COALESCE_MS = 800;
const PASTE_OFFSET = 40;
const MAX_NODE_LABEL_CHARS = 80;
const ARRANGE_GAP_X = 48;
const ARRANGE_GAP_Y = 32;
const FALLBACK_NODE_WIDTH = 250;
const FALLBACK_NODE_HEIGHT = 160;
const FRAME_PADDING_X = 24;
const FRAME_HEADER_HEIGHT = 40;
const FRAME_PADDING_BOTTOM = 24;
const FRAME_MIN_WIDTH = 280;
const FRAME_MIN_HEIGHT = 180;

export type ArrangeMode =
  | "align-left"
  | "align-center-x"
  | "align-right"
  | "align-top"
  | "align-center-y"
  | "align-bottom"
  | "distribute-x"
  | "distribute-y"
  | "grid";

export interface ConnectedNodeResult {
  nodeId: string;
  edgeId: string;
}

export interface WindowMeta {
  title?: string;
  projectName?: string | null;
  projectSlug?: string | null;
}

interface GraphStore {
  templates: TemplateInfo[];
  templatesLoaded: boolean;
  templatesError: string | null;

  workflows: WorkflowTemplateInfo[];
  workflowsLoaded: boolean;
  workflowsError: string | null;

  projects: ProjectSummary[];
  projectsLoaded: boolean;
  projectsError: string | null;
  projectName: string | null;
  projectSlug: string | null;
  saveDialogOpen: boolean;
  saveBusy: boolean;
  saveError: string | null;

  windows: WorkflowWindow[];
  activeWindowId: string;
  executionWindowId: string | null;

  nodes: AppNode[];
  edges: AppEdge[];
  subgraphs: SubgraphMap;
  scopeStack: SubgraphScope[];
  rootGraph: RootGraph | null;

  past: GraphSnapshot[];
  future: GraphSnapshot[];
  clipboard: GraphSnapshot | null;
  historyMeta: HistoryMeta;

  editingNodeId: string | null;
  pendingApproval: PendingApproval | null;
  approvalBusy: boolean;
  approvalError: string | null;
  approvedHashes: Set<string>;

  wsConnected: boolean;
  execution: GraphExecutionState;

  newWorkflowWindow: () => void;
  activateWorkflowWindow: (windowId: string) => boolean;
  closeWorkflowWindow: (windowId: string) => boolean;
  openWorkflowInWindow: (
    saved: SavedWorkflow,
    meta?: WindowMeta,
  ) => Promise<boolean>;
  openProjectInWindow: (slug: string) => Promise<boolean>;
  openWorkflowTemplateInWindow: (
    workflow: WorkflowTemplateInfo,
  ) => boolean;
  getWindowDocument: (windowId: string) => SavedWorkflow | null;

  loadTemplates: () => Promise<void>;
  saveNodeAsTemplate: (
    nodeId: string,
    payload: TemplateCreatePayload,
  ) => Promise<TemplateInfo>;
  loadWorkflowTemplates: () => Promise<void>;
  loadWorkflowTemplate: (workflow: WorkflowTemplateInfo) => boolean;
  loadProjects: () => Promise<void>;
  openProject: (slug: string) => Promise<boolean>;
  requestSave: () => void;
  openSaveDialog: () => void;
  closeSaveDialog: () => void;
  saveProject: (name?: string, overwrite?: boolean) => Promise<boolean>;
  deleteProject: (slug: string) => Promise<boolean>;
  restoreProjectVersion: (slug: string, version: string) => Promise<boolean>;
  addNodeFromTemplate: (template: TemplateInfo, position: XYPosition) => string;
  addConnectedNodeFromTemplate: (
    template: TemplateInfo,
    position: XYPosition,
    connection: PendingConnection,
  ) => ConnectedNodeResult | null;
  addScratchNode: (position: XYPosition) => string;
  duplicateNode: (nodeId: string) => Promise<string | null>;
  deleteNode: (nodeId: string) => void;
  deleteSelection: () => boolean;
  clearCanvas: () => void;
  checkpoint: () => void;
  undo: () => void;
  redo: () => void;
  copySelection: () => boolean;
  cutSelection: () => boolean;
  pasteClipboard: (anchor?: XYPosition) => Promise<string[]>;
  duplicateSelection: () => Promise<string[]>;
  selectOnlyNode: (nodeId: string) => void;
  selectAllNodes: () => void;
  clearSelection: () => void;
  arrangeSelection: (mode: ArrangeMode) => boolean;
  groupSelection: () => string | null;
  ungroupSelection: () => boolean;
  createSubgraphFromSelection: () => string | null;
  addSubgraphInstance: (
    definitionId: string,
    position: XYPosition,
  ) => string | null;
  addSubgraphPort: (
    kind: "input" | "output",
    dataType: SocketType,
    suggestedName?: string,
  ) => string | null;
  enterSubgraph: (nodeId: string) => boolean;
  exitSubgraph: () => boolean;
  navigateToScope: (depth: number) => boolean;
  getWorkflowDocument: (name?: string | null) => SavedWorkflow;
  getExecutionGraph: () => FlattenedWorkflow;
  renameNode: (nodeId: string, label: string) => boolean;
  setNodeColor: (nodeIds: string[], color: NodeColor) => boolean;
  resetNode: (nodeId: string) => boolean;
  onNodesChange: (changes: NodeChange<AppNode>[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  reassignOutputBundle: (
    edgeIds: string[],
    sourceNodeId: string,
    sourceHandleId: string,
  ) => boolean;
  reconnectEdgeTarget: (
    edgeId: string,
    targetNodeId: string,
    targetHandleId: string,
  ) => boolean;
  deleteEdge: (edgeId: string) => boolean;
  openEditor: (nodeId: string) => void;
  closeEditor: () => void;
  updateWidgetValue: (nodeId: string, name: string, value: unknown) => void;
  saveNodeCode: (
    nodeId: string,
    code: string,
    schema: NodeSchema,
  ) => Promise<void>;
  approvePending: () => Promise<void>;
  rejectPending: () => void;
  restoreWorkflow: (saved: SavedWorkflow) => Promise<void>;
  restoreFromStorage: () => Promise<void>;
  runGraph: (targets?: string[]) => Promise<void>;
  runSweep: (
    request: SweepRunRequest,
    onProgress?: (completed: number, total: number) => void,
  ) => Promise<SweepRunResult>;
  setWidgetValueForSweep: (nodeId: string, name: string, value: unknown) => void;
  cancelRun: () => Promise<void>;
  applyExecutionEvent: (event: ExecutionEvent) => void;
  syncRunStatus: (status: BackendRunStatus) => void;
  setWsConnected: (connected: boolean) => void;
}

export interface SweepRunRequest {
  nodeId: string;
  widget: string;
  values: Array<string | number>;
  suffixOutputs?: boolean;
}

export interface SweepRunResult {
  completed: number;
  total: number;
  status: ExecutionStatus;
}

export const SCRATCH_NODE_CODE = `# Scratch node — edit freely.
# - bare annotations create input sockets  (e.g. \\\`train: DATASET\\\`)
# - annotated defaults create parameter widgets  (e.g. \\\`k: int = 5\\\`)
# - named returns create output sockets  (e.g. \\\`return result: ANY\\\`)

factor: float = 2.0
result = factor * 21
return result: ANY
`;

function emptySchema(
  nodeId: string,
  inputs: NodeSchema["inputs"] = [],
  outputs: NodeSchema["outputs"] = [],
): NodeSchema {
  return {
    node_id: nodeId,
    inputs,
    widgets: [],
    outputs,
    dependencies: [],
    syntax_errors: [],
  };
}

interface SizedNode {
  node: AppNode;
  width: number;
  height: number;
}

function sizedNodes(nodes: AppNode[]): SizedNode[] {
  return nodes.map((node) => ({
    node,
    width:
      node.width ?? node.measured?.width ?? FALLBACK_NODE_WIDTH,
    height:
      node.height ?? node.measured?.height ?? FALLBACK_NODE_HEIGHT,
  }));
}

function absolutePosition(
  node: AppNode,
  nodesById: Map<string, AppNode>,
): XYPosition {
  let position = node.position;
  let parentId = node.parentId;
  const visited = new Set<string>([node.id]);
  while (parentId && !visited.has(parentId)) {
    visited.add(parentId);
    const parent = nodesById.get(parentId);
    if (!parent) {
      break;
    }
    position = {
      x: position.x + parent.position.x,
      y: position.y + parent.position.y,
    };
    parentId = parent.parentId;
  }
  return position;
}

function frameLabel(nodes: AppNode[]): string {
  const used = new Set(
    nodes.filter(isFrameNode).map((node) => node.data.label),
  );
  let index = 1;
  while (used.has(`Frame ${index}`)) {
    index += 1;
  }
  return `Frame ${index}`;
}

function arrangementPositions(
  nodes: AppNode[],
  mode: ArrangeMode,
): Map<string, XYPosition> {
  const items = sizedNodes(nodes);
  const positions = new Map<string, XYPosition>();
  if (items.length < 2) {
    return positions;
  }
  const left = Math.min(...items.map((item) => item.node.position.x));
  const top = Math.min(...items.map((item) => item.node.position.y));
  const right = Math.max(
    ...items.map((item) => item.node.position.x + item.width),
  );
  const bottom = Math.max(
    ...items.map((item) => item.node.position.y + item.height),
  );
  const centerX = (left + right) / 2;
  const centerY = (top + bottom) / 2;

  for (const item of items) {
    const position = item.node.position;
    let x = position.x;
    let y = position.y;
    if (mode === "align-left") {
      x = left;
    } else if (mode === "align-center-x") {
      x = centerX - item.width / 2;
    } else if (mode === "align-right") {
      x = right - item.width;
    } else if (mode === "align-top") {
      y = top;
    } else if (mode === "align-center-y") {
      y = centerY - item.height / 2;
    } else if (mode === "align-bottom") {
      y = bottom - item.height;
    }
    positions.set(item.node.id, { x, y });
  }

  if (mode === "distribute-x") {
    if (items.length < 3) {
      return new Map();
    }
    const ordered = [...items].sort(
      (a, b) =>
        a.node.position.x +
        a.width / 2 -
        (b.node.position.x + b.width / 2),
    );
    const totalWidth = ordered.reduce((sum, item) => sum + item.width, 0);
    const gap = (right - left - totalWidth) / (ordered.length - 1);
    let cursor = left;
    for (const item of ordered) {
      positions.set(item.node.id, { x: cursor, y: item.node.position.y });
      cursor += item.width + gap;
    }
  } else if (mode === "distribute-y") {
    if (items.length < 3) {
      return new Map();
    }
    const ordered = [...items].sort(
      (a, b) =>
        a.node.position.y +
        a.height / 2 -
        (b.node.position.y + b.height / 2),
    );
    const totalHeight = ordered.reduce((sum, item) => sum + item.height, 0);
    const gap = (bottom - top - totalHeight) / (ordered.length - 1);
    let cursor = top;
    for (const item of ordered) {
      positions.set(item.node.id, { x: item.node.position.x, y: cursor });
      cursor += item.height + gap;
    }
  } else if (mode === "grid") {
    const ordered = [...items].sort(
      (a, b) =>
        a.node.position.y - b.node.position.y ||
        a.node.position.x - b.node.position.x ||
        a.node.id.localeCompare(b.node.id),
    );
    const columnCount = Math.ceil(Math.sqrt(ordered.length));
    const rowCount = Math.ceil(ordered.length / columnCount);
    const columnWidths = Array<number>(columnCount).fill(0);
    const rowHeights = Array<number>(rowCount).fill(0);
    ordered.forEach((item, index) => {
      const column = index % columnCount;
      const row = Math.floor(index / columnCount);
      columnWidths[column] = Math.max(columnWidths[column] ?? 0, item.width);
      rowHeights[row] = Math.max(rowHeights[row] ?? 0, item.height);
    });
    const columnOffsets: number[] = [];
    let x = left;
    for (const width of columnWidths) {
      columnOffsets.push(x);
      x += width + ARRANGE_GAP_X;
    }
    ordered.forEach((item, index) => {
      const column = index % columnCount;
      const row = Math.floor(index / columnCount);
      const y =
        top +
        rowHeights.slice(0, row).reduce((sum, height) => sum + height, 0) +
        row * ARRANGE_GAP_Y;
      positions.set(item.node.id, {
        x: columnOffsets[column] ?? left,
        y,
      });
    });
  }

  return positions;
}

export function serialiseGraph(
  nodes: AppNode[],
  edges: AppEdge[],
  subgraphs: SubgraphMap = {},
): WorkflowGraphPayload {
  return flattenWorkflow(nodes, edges, subgraphs).graph;
}

function mergeActiveSubgraph(
  state: Pick<GraphStore, "nodes" | "edges" | "subgraphs" | "scopeStack">,
): SubgraphMap {
  const active = state.scopeStack[state.scopeStack.length - 1];
  if (!active) {
    return state.subgraphs;
  }
  const definition = state.subgraphs[active.definitionId];
  if (!definition) {
    return state.subgraphs;
  }
  const inputManager = state.nodes.find(
    (node) => node.id === SUBGRAPH_INPUT_NODE_ID,
  );
  const outputManager = state.nodes.find(
    (node) => node.id === SUBGRAPH_OUTPUT_NODE_ID,
  );
  return {
    ...state.subgraphs,
    [definition.id]: {
      ...definition,
      inputs: inputManager?.data.schema.outputs ?? definition.inputs,
      outputs: outputManager?.data.schema.inputs ?? definition.outputs,
      nodes: stripSubgraphManagers(state.nodes),
      edges: state.edges,
      inputPosition: inputManager?.position ?? definition.inputPosition,
      outputPosition: outputManager?.position ?? definition.outputPosition,
    },
  };
}

function pruneUnusedSubgraphs(
  state: Pick<
    GraphStore,
    "nodes" | "edges" | "subgraphs" | "scopeStack" | "rootGraph"
  >,
  nodes: AppNode[],
  edges: AppEdge[],
  candidates: Set<string>,
): SubgraphMap {
  const merged = mergeActiveSubgraph({ ...state, nodes, edges });
  if (candidates.size === 0) {
    return merged;
  }
  const affected = new Set<string>();
  const visitAffected = (definitionId: string): void => {
    if (affected.has(definitionId)) {
      return;
    }
    const definition = merged[definitionId];
    if (!definition) {
      return;
    }
    affected.add(definitionId);
    for (const node of definition.nodes) {
      if (isSubgraphInstanceNode(node) && node.data.subgraphId) {
        visitAffected(node.data.subgraphId);
      }
    }
  };
  for (const candidate of candidates) {
    visitAffected(candidate);
  }

  const reachable = new Set<string>();
  const visitReachable = (definitionId: string): void => {
    if (reachable.has(definitionId)) {
      return;
    }
    const definition = merged[definitionId];
    if (!definition) {
      return;
    }
    reachable.add(definitionId);
    for (const node of definition.nodes) {
      if (isSubgraphInstanceNode(node) && node.data.subgraphId) {
        visitReachable(node.data.subgraphId);
      }
    }
  };
  const rootNodes =
    state.scopeStack.length === 0
      ? nodes
      : (state.rootGraph?.nodes ?? nodes);
  for (const node of rootNodes) {
    if (isSubgraphInstanceNode(node) && node.data.subgraphId) {
      visitReachable(node.data.subgraphId);
    }
  }

  return Object.fromEntries(
    Object.entries(merged).filter(
      ([id]) => !affected.has(id) || reachable.has(id),
    ),
  );
}

function rootGraphFor(
  state: Pick<GraphStore, "nodes" | "edges" | "scopeStack" | "rootGraph">,
): RootGraph {
  if (state.scopeStack.length === 0) {
    return { nodes: state.nodes, edges: state.edges };
  }
  return state.rootGraph ?? { nodes: [], edges: [] };
}

function graphForDefinition(
  definition: SubgraphDefinition,
): RootGraph {
  return {
    nodes: [
      ...definition.nodes.map((node) => ({ ...node, selected: false })),
      ...createSubgraphManagerNodes(definition),
    ],
    edges: definition.edges,
  };
}

function nextSubgraphName(subgraphs: SubgraphMap): string {
  const names = new Set(
    Object.values(subgraphs).map((definition) => definition.name),
  );
  let index = 1;
  while (names.has(`Subgraph ${index}`)) {
    index += 1;
  }
  return `Subgraph ${index}`;
}

function emptyWindowState(): WorkflowWindowState {
  return {
    nodes: [],
    edges: [],
    subgraphs: {},
    scopeStack: [],
    rootGraph: null,
    past: [],
    future: [],
    execution: idleExecution(),
    projectName: null,
    projectSlug: null,
  };
}

function nextUntitledTitle(windows: WorkflowWindow[]): string {
  const used = new Set(windows.map((window) => window.title));
  let index = 1;
  while (used.has(`Untitled ${index}`)) {
    index += 1;
  }
  return `Untitled ${index}`;
}

function createWorkflowWindow(
  windows: WorkflowWindow[],
  meta: WindowMeta = {},
): WorkflowWindow {
  return {
    id: crypto.randomUUID(),
    title: meta.title ?? nextUntitledTitle(windows),
    ...emptyWindowState(),
    projectName: meta.projectName ?? null,
    projectSlug: meta.projectSlug ?? null,
  };
}

function snapshotActiveWindow(state: GraphStore): WorkflowWindowState {
  return {
    nodes: state.nodes,
    edges: state.edges,
    subgraphs: mergeActiveSubgraph(state),
    scopeStack: state.scopeStack,
    rootGraph: state.rootGraph,
    past: state.past,
    future: state.future,
    execution: state.execution,
    projectName: state.projectName,
    projectSlug: state.projectSlug,
  };
}

function windowTopLevel(window: WorkflowWindow): Partial<GraphStore> {
  return {
    nodes: window.nodes,
    edges: window.edges,
    subgraphs: window.subgraphs,
    scopeStack: window.scopeStack,
    rootGraph: window.rootGraph,
    past: window.past,
    future: window.future,
    execution: window.execution,
    projectName: window.projectName,
    projectSlug: window.projectSlug,
    editingNodeId: null,
    pendingApproval: null,
    approvalBusy: false,
    approvalError: null,
    saveDialogOpen: false,
    saveError: null,
    historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },
  };
}

function windowsWithSnapshot(
  state: GraphStore,
  snapshot: WorkflowWindowState,
): WorkflowWindow[] {
  return state.windows.map((window) =>
    window.id === state.activeWindowId
      ? {
          ...window,
          ...snapshot,
          title: snapshot.projectName ?? window.title,
        }
      : window,
  );
}

function materialiseWorkflowTemplate(
  workflow: WorkflowTemplateInfo,
  templates: TemplateInfo[],
): { graph: RootGraph; missing: string[] } {
  const byName = new Map(
    templates.map((template) => [template.name, template]),
  );
  const missing = workflow.nodes
    .filter((node) => !byName.has(node.template))
    .map((node) => node.template);
  if (missing.length > 0) {
    return { graph: { nodes: [], edges: [] }, missing };
  }
  const nodes: AppNode[] = workflow.nodes.map((spec) => {
    const template = byName.get(spec.template) as TemplateInfo;
    return {
      id: spec.id,
      type: "complex",
      position: spec.position,
      data: {
        label: template.title,
        templateName: template.name,
        templateCode: template.code,
        code: template.code,
        schema: template.node_schema,
        widgetValues: {
          ...defaultWidgetValues(template.node_schema),
          ...spec.widget_values,
        },
        isTrusted: true,
        color: "default",
      },
    };
  });
  const edges: AppEdge[] = workflow.edges.map((edge) => ({
    id: `${edge.source_node_id}:${edge.source_handle}->${edge.target_node_id}:${edge.target_handle}`,
    source: edge.source_node_id,
    target: edge.target_node_id,
    sourceHandle: edge.source_handle,
    targetHandle: edge.target_handle,
  }));
  return {
    graph: { nodes: normalizeValueNodeValues(nodes, edges), edges },
    missing: [],
  };
}

interface BuiltWorkflow {
  nodes: AppNode[];
  edges: AppEdge[];
  subgraphs: SubgraphMap;
}

async function buildSavedWorkflow(
  saved: SavedWorkflow,
  templates: TemplateInfo[],
): Promise<BuiltWorkflow> {
  const templateByName = new Map(
    templates.map((template) => [template.name, template]),
  );
  const savedSubgraphs = saved.subgraphs ?? {};

  const buildNode = async (
    savedNode: SavedWorkflow["nodes"][number],
  ): Promise<AppNode> => {
    if (savedNode.isFrame) {
      return {
        id: savedNode.id,
        type: "complex",
        position: savedNode.position,
        width: savedNode.width ?? FRAME_MIN_WIDTH,
        height: savedNode.height ?? FRAME_MIN_HEIGHT,
        parentId: savedNode.parentId,
        extent: savedNode.parentId ? "parent" : undefined,
        zIndex: -1,
        data: {
          label: savedNode.label,
          templateName: null,
          templateCode: null,
          code: "",
          schema: emptySchema(savedNode.id),
          widgetValues: {},
          isTrusted: true,
          color: savedNode.color ?? "default",
          isFrame: true,
        },
      };
    }
    if (savedNode.isSubgraph) {
      const definition = savedSubgraphs[savedNode.subgraphId ?? ""];
      if (!definition) {
        throw new Error(
          `Node '${savedNode.id}' references a missing subgraph.`,
        );
      }
      return {
        id: savedNode.id,
        type: "complex",
        position: savedNode.position,
        width: savedNode.width,
        height: savedNode.height,
        parentId: savedNode.parentId,
        extent: savedNode.parentId ? "parent" : undefined,
        data: {
          label: savedNode.label,
          templateName: null,
          templateCode: null,
          code: "",
          schema: emptySchema(
            savedNode.id,
            definition.inputs,
            definition.outputs,
          ),
          widgetValues: {},
          isTrusted: true,
          color: savedNode.color ?? "default",
          isSubgraph: true,
          subgraphId: definition.id,
        },
      };
    }
    let schema: NodeSchema;
    try {
      schema = await parseCode(savedNode.code, savedNode.id);
    } catch (error) {
      schema = {
        ...emptySchema(savedNode.id),
        syntax_errors: [
          {
            line: 1,
            column: 0,
            message: `Failed to re-parse restored code: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
    const template = savedNode.templateName
      ? (templateByName.get(savedNode.templateName) ?? null)
      : null;
    const hash = await sha256Hex(savedNode.code);
    const isOfficial = templates.some(
      (candidate) => candidate.code_hash === hash,
    );
    return {
      id: savedNode.id,
      type: "complex",
      position: savedNode.position,
      width: savedNode.width,
      height: savedNode.height,
      parentId: savedNode.parentId,
      extent: savedNode.parentId ? "parent" : undefined,
      data: {
        label: savedNode.label,
        templateName: template?.name ?? savedNode.templateName,
        templateCode: template?.code ?? null,
        code: savedNode.code,
        schema,
        widgetValues: mergeWidgetValues(schema, savedNode.widgetValues),
        isTrusted: isOfficial,
        color: savedNode.color ?? "default",
      },
    };
  };

  const buildGraph = async (
    nodes: SavedWorkflow["nodes"],
    edges: SavedWorkflow["edges"],
    definition: boolean,
  ): Promise<RootGraph> => {
    const built = await Promise.all(nodes.map(buildNode));
    built.sort(
      (a, b) => Number(Boolean(a.parentId)) - Number(Boolean(b.parentId)),
    );
    const nodeIds = new Set(built.map((node) => node.id));
    const allowed = new Set(nodeIds);
    if (definition) {
      allowed.add(SUBGRAPH_INPUT_NODE_ID);
      allowed.add(SUBGRAPH_OUTPUT_NODE_ID);
    }
    let builtEdges: AppEdge[] = edges
      .filter(
        (edge) => allowed.has(edge.source) && allowed.has(edge.target),
      )
      .map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
      }));
    for (const node of built) {
      builtEdges = reconcileEdges(
        builtEdges,
        node.id,
        node.data.schema,
      );
    }
    return { nodes: built, edges: builtEdges };
  };

  const subgraphs: SubgraphMap = {};
  for (const [id, definition] of Object.entries(savedSubgraphs)) {
    const graph = await buildGraph(
      definition.nodes,
      definition.edges,
      true,
    );
    subgraphs[id] = {
      id: definition.id,
      name: definition.name,
      inputs: definition.inputs,
      outputs: definition.outputs,
      nodes: graph.nodes,
      edges: graph.edges,
      inputPosition: definition.inputPosition,
      outputPosition: definition.outputPosition,
    };
  }
  const root = await buildGraph(saved.nodes, saved.edges, false);
  return { nodes: root.nodes, edges: root.edges, subgraphs };
}

export const useGraphStore = create<GraphStore>((set, get) => {
  const initialWindow = createWorkflowWindow([]);

  const commitNodeCode = (
    nodeId: string,
    code: string,
    schema: NodeSchema,
    isTrusted: boolean,
  ) => {
    set((state) => {
      const node = state.nodes.find((candidate) => candidate.id === nodeId);
      if (!node) {
        return {};
      }
      const data: ComplexNodeData = {
        ...node.data,
        code,
        schema,
        widgetValues: mergeWidgetValues(schema, node.data.widgetValues),
        isTrusted,
      };
      return {
        nodes: state.nodes.map((candidate) =>
          candidate.id === nodeId ? { ...candidate, data } : candidate,
        ),
        edges: reconcileEdges(state.edges, nodeId, schema),
      };
    });
  };

  const snapshotGraph = (): GraphSnapshot => {
    const state = get();
    return {
      nodes: state.nodes,
      edges: state.edges,
      subgraphs: mergeActiveSubgraph(state),
      scopeStack: state.scopeStack,
      rootGraph: state.rootGraph,
    };
  };

  const pushHistory = (coalesceKey?: string): void => {
    const now = Date.now();
    const meta = get().historyMeta;
    if (
      coalesceKey !== undefined &&
      meta.coalesceKey === coalesceKey &&
      now - meta.coalesceAt < HISTORY_COALESCE_MS
    ) {
      set({ historyMeta: { ...meta, coalesceAt: now } });
      return;
    }
    const snapshot = snapshotGraph();
    set((state) => ({
      past: [...state.past.slice(-(HISTORY_LIMIT - 1)), snapshot],
      future: [],
      historyMeta: {
        coalesceKey: coalesceKey ?? null,
        coalesceAt: now,
        dragging: state.historyMeta.dragging,
      },
    }));
  };

  return {
    templates: [],
    templatesLoaded: false,
    templatesError: null,

    workflows: [],
    workflowsLoaded: false,
    workflowsError: null,

    projects: [],
    projectsLoaded: false,
    projectsError: null,
    projectName: initialWindow.projectName,
    projectSlug: initialWindow.projectSlug,
    saveDialogOpen: false,
    saveBusy: false,
    saveError: null,

    windows: [initialWindow],
    activeWindowId: initialWindow.id,
    executionWindowId: null,

    nodes: initialWindow.nodes,
    edges: initialWindow.edges,
    subgraphs: initialWindow.subgraphs,
    scopeStack: initialWindow.scopeStack,
    rootGraph: initialWindow.rootGraph,

    past: [],
    future: [],
    clipboard: null,
    historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },

    editingNodeId: null,
    pendingApproval: null,
    approvalBusy: false,
    approvalError: null,
    approvedHashes: new Set<string>(),

    wsConnected: false,
    execution: idleExecution(),

    newWorkflowWindow: () => {
      const state = get();
      const windows = windowsWithSnapshot(state, snapshotActiveWindow(state));
      const window = createWorkflowWindow(windows);
      set({
        windows: [...windows, window],
        activeWindowId: window.id,
        ...windowTopLevel(window),
      });
    },

    activateWorkflowWindow: (windowId) => {
      const state = get();
      if (state.activeWindowId === windowId) {
        return true;
      }
      const target = state.windows.find((window) => window.id === windowId);
      if (!target) {
        return false;
      }
      const windows = windowsWithSnapshot(state, snapshotActiveWindow(state));
      set({
        windows,
        activeWindowId: target.id,
        ...windowTopLevel(target),
      });
      return true;
    },

    closeWorkflowWindow: (windowId) => {
      const state = get();
      const index = state.windows.findIndex((window) => window.id === windowId);
      if (index < 0) {
        return false;
      }
      const target = state.windows[index];
      if (!target) {
        return false;
      }
      const closingActive = target.id === state.activeWindowId;
      const running = closingActive
        ? state.execution.status === "running"
        : target.execution.status === "running";
      if (running) {
        return false;
      }
      if (state.windows.length === 1) {
        const window = createWorkflowWindow([]);
        set({
          windows: [window],
          activeWindowId: window.id,
          executionWindowId: null,
          ...windowTopLevel(window),
        });
        return true;
      }
      if (!closingActive) {
        const windows = windowsWithSnapshot(
          state,
          snapshotActiveWindow(state),
        ).filter((window) => window.id !== windowId);
        set({
          windows,
          executionWindowId:
            state.executionWindowId === windowId
              ? null
              : state.executionWindowId,
        });
        return true;
      }
      const windows = state.windows.filter((window) => window.id !== windowId);
      const next = windows[Math.min(index, windows.length - 1)] ?? windows[0];
      if (!next) {
        return false;
      }
      set({
        windows,
        activeWindowId: next.id,
        executionWindowId:
          state.executionWindowId === windowId
            ? null
            : state.executionWindowId,
        ...windowTopLevel(next),
      });
      return true;
    },

    openWorkflowInWindow: async (saved, meta = {}) => {
      const built = await buildSavedWorkflow(saved, get().templates);
      const state = get();
      const windows = windowsWithSnapshot(state, snapshotActiveWindow(state));
      const projectName = meta.projectName ?? saved.name ?? null;
      const window = createWorkflowWindow(windows, {
        title: meta.title ?? projectName ?? undefined,
        projectName,
        projectSlug: meta.projectSlug ?? null,
      });
      window.nodes = built.nodes;
      window.edges = built.edges;
      window.subgraphs = built.subgraphs;
      set({
        windows: [...windows, window],
        activeWindowId: window.id,
        ...windowTopLevel(window),
      });
      return true;
    },

    openProjectInWindow: async (slug) => {
      const state = get();
      const existing = state.windows.find(
        (window) => window.projectSlug === slug,
      );
      if (existing) {
        return get().activateWorkflowWindow(existing.id);
      }
      try {
        const detail = await fetchProjectApi(slug);
        const opened = await get().openWorkflowInWindow(detail.workflow, {
          title: detail.name,
          projectName: detail.name,
          projectSlug: detail.slug,
        });
        if (opened) {
          set((current) => ({
            projects: [
              detail,
              ...current.projects.filter(
                (project) => project.slug !== detail.slug,
              ),
            ],
            projectsLoaded: true,
            projectsError: null,
          }));
        }
        return opened;
      } catch (error) {
        set({
          projectsError:
            error instanceof Error ? error.message : String(error),
        });
        return false;
      }
    },

    openWorkflowTemplateInWindow: (workflow) => {
      const state = get();
      const { graph, missing } = materialiseWorkflowTemplate(
        workflow,
        state.templates,
      );
      if (missing.length > 0) {
        set({
          workflowsError: `Missing templates: ${missing.join(", ")}`,
        });
        return false;
      }
      const windows = windowsWithSnapshot(state, snapshotActiveWindow(state));
      const window = createWorkflowWindow(windows, {
        title: workflow.title,
      });
      window.nodes = graph.nodes;
      window.edges = graph.edges;
      window.execution = idleExecution();
      set({
        windows: [...windows, window],
        activeWindowId: window.id,
        ...windowTopLevel(window),
        workflowsError: null,
      });
      return true;
    },

    getWindowDocument: (windowId) => {
      const state = get();
      if (windowId === state.activeWindowId) {
        return state.getWorkflowDocument(state.projectName);
      }
      const window = state.windows.find(
        (candidate) => candidate.id === windowId,
      );
      if (!window) {
        return null;
      }
      const graph = rootGraphFor(window);
      return serialiseWorkflow(
        graph.nodes,
        graph.edges,
        window.projectName,
        mergeActiveSubgraph(window),
      );
    },

    loadTemplates: async () => {
      try {
        const templates = await fetchTemplates();
        set({ templates, templatesLoaded: true, templatesError: null });
      } catch (error) {
        set({
          templatesError:
            error instanceof Error ? error.message : String(error),
        });
      }
    },

    saveNodeAsTemplate: async (nodeId, payload) => {
      const template = await createTemplateApi(payload);
      set((state) => ({
        templates: [
          ...state.templates.filter((item) => item.name !== template.name),
          template,
        ].sort((left, right) => left.title.localeCompare(right.title)),
        nodes: state.nodes.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  code: template.code,
                  schema: template.node_schema,
                  templateName: template.name,
                  templateCode: template.code,
                  isTrusted: true,
                },
              }
            : node,
        ),
      }));
      return template;
    },

    loadWorkflowTemplates: async () => {
      try {
        const workflows = await fetchWorkflowTemplates();
        set({ workflows, workflowsLoaded: true, workflowsError: null });
      } catch (error) {
        set({
          workflowsError:
            error instanceof Error ? error.message : String(error),
        });
      }
    },

    loadWorkflowTemplate: (workflow) => {
      const { graph, missing } = materialiseWorkflowTemplate(
        workflow,
        get().templates,
      );
      if (missing.length > 0) {
        set({
          workflowsError: `Missing templates: ${missing.join(", ")}`,
        });
        return false;
      }
      pushHistory();
      set((state) => ({
        windows: state.windows.map((window) =>
          window.id === state.activeWindowId
            ? {
                ...window,
                ...snapshotActiveWindow(state),
                nodes: graph.nodes,
                edges: graph.edges,
                subgraphs: {},
                scopeStack: [],
                rootGraph: null,
                execution: idleExecution(),
                projectName: null,
                projectSlug: null,
                title: workflow.title,
              }
            : window,
        ),
        nodes: graph.nodes,
        edges: graph.edges,
        subgraphs: {},
        scopeStack: [],
        rootGraph: null,
        projectName: null,
        projectSlug: null,
        saveDialogOpen: false,
        saveError: null,
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        workflowsError: null,
        execution: idleExecution(),
      }));
      return true;
    },

    loadProjects: async () => {
      try {
        const projects = await fetchProjectsApi();
        set({ projects, projectsLoaded: true, projectsError: null });
      } catch (error) {
        set({
          projectsLoaded: true,
          projectsError:
            error instanceof Error ? error.message : String(error),
        });
      }
    },

    openProject: async (slug) => {
      try {
        const detail = await fetchProjectApi(slug);
        await get().restoreWorkflow(detail.workflow);
        set((state) => ({
          projectName: detail.name,
          projectSlug: detail.slug,
          projectsError: null,
          saveDialogOpen: false,
          saveError: null,
          past: [],
          future: [],
          windows: state.windows.map((window) =>
            window.id === state.activeWindowId
              ? {
                  ...window,
                  ...snapshotActiveWindow(state),
                  past: [],
                  future: [],
                  projectName: detail.name,
                  projectSlug: detail.slug,
                  title: detail.name,
                }
              : window,
          ),
        }));
        const state = get();
        saveLocalWorkflow(state.getWorkflowDocument(detail.name));
        saveLocalProjectIdentity({ name: detail.name, slug: detail.slug });
        return true;
      } catch (error) {
        set({
          projectsError:
            error instanceof Error ? error.message : String(error),
        });
        return false;
      }
    },

    requestSave: () => {
      const state = get();
      if (
        state.nodes.length === 0 &&
        state.edges.length === 0 &&
        Object.keys(state.subgraphs).length === 0
      ) {
        return;
      }
      if (state.projectSlug && state.projectName) {
        void state.saveProject();
        return;
      }
      set({ saveDialogOpen: true, saveError: null });
    },

    openSaveDialog: () => set({ saveDialogOpen: true, saveError: null }),

    closeSaveDialog: () => {
      if (!get().saveBusy) {
        set({ saveDialogOpen: false, saveError: null });
      }
    },

    saveProject: async (name, overwrite = false) => {
      const state = get();
      if (state.saveBusy) {
        return false;
      }
      const projectName = (name ?? state.projectName ?? "").trim();
      if (!projectName) {
        set({
          saveDialogOpen: true,
          saveError: "Enter a project name between 1 and 80 characters.",
        });
        return false;
      }
      const windowId = state.activeWindowId;
      const workflow = state.getWorkflowDocument(projectName);
      set({ saveBusy: true, saveError: null });
      try {
        const result = state.projectSlug
          ? await updateProjectApi(state.projectSlug, projectName, workflow)
          : await createProjectApi(projectName, workflow, overwrite);
        const saved = result.project;
        set((current) => {
          const active = current.activeWindowId === windowId;
          return {
            ...(active
              ? { projectName: saved.name, projectSlug: saved.slug }
              : {}),
            projects: [
              saved,
              ...current.projects.filter(
                (project) => project.slug !== saved.slug,
              ),
            ],
            projectsLoaded: true,
            projectsError: null,
            saveBusy: false,
            saveDialogOpen: false,
            saveError: null,
            windows: current.windows.map((window) =>
              window.id === windowId
                ? {
                    ...window,
                    projectName: saved.name,
                    projectSlug: saved.slug,
                    title: saved.name,
                  }
                : window,
            ),
          };
        });
        if (get().activeWindowId === windowId) {
          saveLocalWorkflow(workflow);
          saveLocalProjectIdentity({ name: saved.name, slug: saved.slug });
        }
        return true;
      } catch (error) {
        const message =
          error instanceof Error ? error.message : String(error);
        set((current) =>
          current.activeWindowId === windowId
            ? {
                saveBusy: false,
                saveDialogOpen: true,
                saveError: message,
              }
            : { saveBusy: false },
        );
        return false;
      }
    },

    deleteProject: async (slug) => {
      try {
        await deleteProjectApi(slug);
        const wasCurrent = get().projectSlug === slug;
        set((state) => ({
          projects: state.projects.filter((project) => project.slug !== slug),
          projectsError: null,
          windows: state.windows.map((window) =>
            window.projectSlug === slug
              ? {
                  ...window,
                  projectName: null,
                  projectSlug: null,
                  title: "Untitled",
                }
              : window,
          ),
          ...(wasCurrent
            ? { projectName: null, projectSlug: null, saveError: null }
            : {}),
        }));
        if (wasCurrent) {
          saveLocalProjectIdentity({ name: null, slug: null });
        }
        return true;
      } catch (error) {
        set({
          projectsError:
            error instanceof Error ? error.message : String(error),
        });
        return false;
      }
    },

    restoreProjectVersion: async (slug, version) => {
      try {
        const snapshot = await fetchProjectVersionApi(slug, version);
        await get().restoreWorkflow(snapshot.workflow);
        const project = get().projects.find((item) => item.slug === slug);
        set((state) => {
          const projectName =
            snapshot.workflow.name ?? project?.name ?? state.projectName;
          return {
            projectName,
            projectSlug: slug,
            projectsError: null,
            saveDialogOpen: false,
            saveError: null,
            past: [],
            future: [],
            windows: state.windows.map((window) =>
              window.id === state.activeWindowId
                ? {
                    ...window,
                    ...snapshotActiveWindow(state),
                    past: [],
                    future: [],
                    projectName,
                    projectSlug: slug,
                    title: projectName ?? window.title,
                  }
                : window,
            ),
          };
        });
        const state = get();
        saveLocalWorkflow(state.getWorkflowDocument(state.projectName));
        saveLocalProjectIdentity({
          name: state.projectName,
          slug: state.projectSlug,
        });
        return true;
      } catch (error) {
        set({
          projectsError:
            error instanceof Error ? error.message : String(error),
        });
        return false;
      }
    },

    addNodeFromTemplate: (template, position) => {
      const id = crypto.randomUUID();
      const node: AppNode = {
        id,
        type: "complex",
        position,
        data: {
          label: template.title,
          templateName: template.name,
          templateCode: template.code,
          code: template.code,
          schema: template.node_schema,
          widgetValues: defaultWidgetValues(template.node_schema),
          isTrusted: true,
          color: "default",
        },
      };
      pushHistory();
      set((state) => ({ nodes: [...state.nodes, node] }));
      return id;
    },

    addConnectedNodeFromTemplate: (template, position, connection) => {
      const state = get();
      const existing = state.nodes.find(
        (node) => node.id === connection.nodeId,
      );
      if (!existing) {
        return null;
      }
      const newInput =
        existing.data.subgraphRole === "input" &&
        connection.handleType === "source" &&
        connection.handleId === NEW_SUBGRAPH_INPUT_HANDLE;
      const newOutput =
        existing.data.subgraphRole === "output" &&
        connection.handleType === "target" &&
        connection.handleId === NEW_SUBGRAPH_OUTPUT_HANDLE;
      let sourceHandle = connection.handleId;
      let targetHandle = connection.handleId;
      let match: TemplateConnectionMatch | null;
      let createdPort = false;
      if (newInput || newOutput) {
        if (newInput) {
          match = findCompatibleTarget(template.node_schema, "ANY");
        } else {
          const sourceSocket = findCompatibleOutput(
            template.node_schema,
            "ANY",
          );
          match = sourceSocket
            ? {
                socketName: sourceSocket.name,
                socketType: sourceSocket.data_type,
                targetKind: "output",
              }
            : null;
        }
        if (!match) {
          return null;
        }
        const port = get().addSubgraphPort(
          newInput ? "input" : "output",
          match.socketType,
          match.socketName,
        );
        if (!port) {
          return null;
        }
        if (newInput) {
          sourceHandle = port;
        } else {
          targetHandle = port;
        }
        createdPort = true;
      } else {
        const existingSocket =
          connection.handleType === "source"
            ? existing.data.schema.outputs.find(
                (socket) => socket.name === connection.handleId,
              )
            : resolveTargetSocket(
                existing.data.schema,
                connection.handleId,
              );
        if (!existingSocket) {
          return null;
        }
        if (
          connection.handleType === "target" &&
          isInputTaken(
            state.edges,
            connection.nodeId,
            connection.handleId,
          )
        ) {
          return null;
        }
        if (connection.handleType === "source") {
          if (connection.bundleEdgeIds?.length) {
            const sourceSocket = findCompatibleOutput(
              template.node_schema,
              existingSocket.data_type,
            );
            match = sourceSocket
              ? {
                  socketName: sourceSocket.name,
                  socketType: sourceSocket.data_type,
                  targetKind: "output",
                }
              : null;
          } else {
            match = findCompatibleTarget(
              template.node_schema,
              existingSocket.data_type,
            );
          }
        } else {
          const sourceSocket = findCompatibleOutput(
            template.node_schema,
            existingSocket.data_type,
          );
          match = sourceSocket
            ? {
                socketName: sourceSocket.name,
                socketType: sourceSocket.data_type,
                targetKind: "output",
              }
            : null;
        }
        if (!match) {
          return null;
        }
      }
      const id = crypto.randomUUID();
      const node: AppNode = {
        id,
        type: "complex",
        position,
        selected: true,
        data: {
          label: template.title,
          templateName: template.name,
          templateCode: template.code,
          code: template.code,
          schema: template.node_schema,
          widgetValues: defaultWidgetValues(template.node_schema),
          isTrusted: true,
          color: "default",
        },
      };
      if (
        connection.handleType === "target" &&
        isOfficialValueNode(node.data)
      ) {
        const targetWidget = existing.data.schema.widgets.find(
          (widget) => widget.name === connection.handleId,
        );
        if ((targetWidget?.options?.length ?? 0) > 0) {
          node.data.widgetValues.value = normalizeValueChoice(
            node.data.widgetValues.value,
            targetWidget?.options ?? [],
          );
        }
      }
      const bundleIds = connection.bundleEdgeIds;
      if (bundleIds && bundleIds.length > 0) {
        if (connection.handleType !== "source" || match.targetKind !== "output") {
          return null;
        }
        const ids = new Set(bundleIds);
        const bundleEdges = state.edges.filter((edge) => ids.has(edge.id));
        if (bundleEdges.length !== ids.size) {
          return null;
        }
        pushHistory();
        set((current) => ({
          nodes: [
            ...current.nodes.map((candidate) =>
              candidate.selected
                ? { ...candidate, selected: false }
                : candidate,
            ),
            node,
          ],
          edges: current.edges.map((edge) =>
            ids.has(edge.id)
              ? {
                  ...edge,
                  id: `${id}:${match.socketName}->${edge.target}:${edge.targetHandle}`,
                  source: id,
                  sourceHandle: match.socketName,
                }
              : edge,
          ),
        }));
        return { nodeId: id, edgeId: bundleIds[0] ?? "" };
      }
      let normalizeExistingValue = false;
      let existingValue = existing.data.widgetValues.value;
      if (
        connection.handleType === "source" &&
        match.targetKind === "widget" &&
        isOfficialValueNode(existing.data)
      ) {
        const targetWidget = template.node_schema.widgets.find(
          (widget) => widget.name === match.socketName,
        );
        if ((targetWidget?.options?.length ?? 0) > 0) {
          existingValue = normalizeValueChoice(
            existing.data.widgetValues.value,
            targetWidget?.options ?? [],
          );
          normalizeExistingValue = !Object.is(
            existingValue,
            existing.data.widgetValues.value,
          );
        }
      }
      const edge: AppEdge =
        connection.handleType === "source"
          ? {
              id: `${connection.nodeId}:${sourceHandle}->${id}:${match.socketName}`,
              source: connection.nodeId,
              sourceHandle,
              target: id,
              targetHandle: match.socketName,
            }
          : {
              id: `${id}:${match.socketName}->${connection.nodeId}:${targetHandle}`,
              source: id,
              sourceHandle: match.socketName,
              target: connection.nodeId,
              targetHandle,
            };
      if (!createdPort) {
        pushHistory();
      }
      set((current) => ({
        nodes: [
          ...current.nodes.map((candidate) => {
            const normalized =
              normalizeExistingValue && candidate.id === connection.nodeId
                ? {
                    ...candidate,
                    data: {
                      ...candidate.data,
                      widgetValues: {
                        ...candidate.data.widgetValues,
                        value: existingValue,
                      },
                    },
                  }
                : candidate;
            return normalized.selected
              ? { ...normalized, selected: false }
              : normalized;
          }),
          node,
        ],
        edges: [...current.edges, edge],
      }));
      return { nodeId: id, edgeId: edge.id };
    },

    addScratchNode: (position) => {
      const id = crypto.randomUUID();
      const schema = emptySchema(id);
      const node: AppNode = {
        id,
        type: "complex",
        position,
        data: {
          label: "Scratch node",
          templateName: null,
          templateCode: null,
          code: SCRATCH_NODE_CODE,
          schema,
          widgetValues: defaultWidgetValues(schema),
          isTrusted: false,
          color: "default",
        },
      };
      pushHistory();
      set((state) => ({ nodes: [...state.nodes, node] }));
      void (async () => {
        try {
          const parsed = await parseCode(SCRATCH_NODE_CODE, id);
          set((state) => ({
            nodes: state.nodes.map((candidate) =>
              candidate.id === id &&
              candidate.data.code === SCRATCH_NODE_CODE
                ? {
                    ...candidate,
                    data: {
                      ...candidate.data,
                      schema: parsed,
                      widgetValues: mergeWidgetValues(
                        parsed,
                        candidate.data.widgetValues,
                      ),
                    },
                  }
                : candidate,
            ),
          }));
        } catch {
          // Backend offline: node keeps an empty schema until it is edited.
        }
      })();
      return id;
    },

    duplicateNode: async (nodeId) => {
      const node = get().nodes.find((candidate) => candidate.id === nodeId);
      if (!node || isSubgraphManagerNode(node)) {
        return null;
      }
      pushHistory();
      const id = crypto.randomUUID();
      const hash = await sha256Hex(node.data.code);
      const isOfficial = get().templates.some(
        (template) => template.code_hash === hash,
      );
      const copy: AppNode = {
        ...node,
        id,
        selected: false,
        position: { x: node.position.x + 48, y: node.position.y + 48 },
        data: { ...node.data, isTrusted: isOfficial },
      };
      set((state) => ({ nodes: [...state.nodes, copy] }));
      return id;
    },

    deleteNode: (nodeId) => {
      const target = get().nodes.find((node) => node.id === nodeId);
      if (!target || isSubgraphManagerNode(target)) {
        return;
      }
      pushHistory();
      const removingFrame = isFrameNode(target);
      const candidates = new Set(
        isSubgraphInstanceNode(target) && target.data.subgraphId
          ? [target.data.subgraphId]
          : [],
      );
      set((state) => {
        const nodes = state.nodes
          .filter((node) => node.id !== nodeId)
          .map((node) =>
            removingFrame && node.parentId === nodeId
              ? {
                  ...node,
                  parentId: undefined,
                  extent: undefined,
                  position: {
                    x: node.position.x + target.position.x,
                    y: node.position.y + target.position.y,
                  },
                }
              : node,
          );
        const edges = state.edges.filter(
          (edge) => edge.source !== nodeId && edge.target !== nodeId,
        );
        return {
          nodes,
          edges,
          subgraphs:
            candidates.size > 0
              ? pruneUnusedSubgraphs(state, nodes, edges, candidates)
              : state.subgraphs,
          editingNodeId:
            state.editingNodeId === nodeId ? null : state.editingNodeId,
          execution: pruneExecution(state.execution, new Set([nodeId])),
        };
      });
    },

    deleteSelection: () => {
      const snapshot = get();
      const selected = snapshot.nodes.filter((node) => node.selected);
      if (selected.length === 0) {
        return false;
      }
      const frames = new Set(
        selected.filter(isFrameNode).map((node) => node.id),
      );
      const removed = new Set(
        selected.filter(isRunnableNode).map((node) => node.id),
      );
      if (frames.size === 0 && removed.size === 0) {
        return false;
      }
      const nodesById = new Map(
        snapshot.nodes.map((node) => [node.id, node]),
      );
      const candidates = new Set(
        selected
          .filter(isSubgraphInstanceNode)
          .map((node) => node.data.subgraphId)
          .filter((id): id is string => Boolean(id)),
      );
      pushHistory();
      set((state) => {
        const nodes = state.nodes
          .filter(
            (node) => !frames.has(node.id) && !removed.has(node.id),
          )
          .map((node) => {
            if (!node.parentId || !frames.has(node.parentId)) {
              return node;
            }
            return {
              ...node,
              parentId: undefined,
              extent: undefined,
              position: absolutePosition(node, nodesById),
            };
          });
        const edges = state.edges.filter(
          (edge) =>
            !removed.has(edge.source) && !removed.has(edge.target),
        );
        return {
          nodes,
          edges,
          subgraphs:
            candidates.size > 0
              ? pruneUnusedSubgraphs(state, nodes, edges, candidates)
              : state.subgraphs,
          editingNodeId:
            state.editingNodeId !== null &&
            removed.has(state.editingNodeId)
              ? null
              : state.editingNodeId,
          execution: pruneExecution(state.execution, removed),
        };
      });
      return true;
    },

    selectOnlyNode: (nodeId) => {
      if (!get().nodes.some((node) => node.id === nodeId)) {
        return;
      }
      set((state) => ({
        nodes: state.nodes.map((node) => ({
          ...node,
          selected: node.id === nodeId,
        })),
      }));
    },

    selectAllNodes: () => {
      if (get().nodes.length === 0) {
        return;
      }
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.selected ? node : { ...node, selected: true },
        ),
      }));
    },

    clearSelection: () => {
      if (!get().nodes.some((node) => node.selected)) {
        return;
      }
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.selected ? { ...node, selected: false } : node,
        ),
      }));
    },

    arrangeSelection: (mode) => {
      const allSelected = get().nodes.filter((node) => node.selected);
      if (
        allSelected.some(
          (node) => isFrameNode(node) || node.parentId !== undefined,
        )
      ) {
        return false;
      }
      const selected = allSelected;
      if (selected.length < 2) {
        return false;
      }
      if (
        (mode === "distribute-x" || mode === "distribute-y") &&
        selected.length < 3
      ) {
        return false;
      }
      const positions = arrangementPositions(selected, mode);
      if (positions.size === 0) {
        return false;
      }
      const changed = selected.some((node) => {
        const position = positions.get(node.id);
        return (
          position !== undefined &&
          (position.x !== node.position.x || position.y !== node.position.y)
        );
      });
      if (!changed) {
        return false;
      }
      pushHistory();
      set((state) => ({
        nodes: state.nodes.map((node) => {
          const position = positions.get(node.id);
          return position ? { ...node, position } : node;
        }),
      }));
      return true;
    },

    groupSelection: () => {
      const state = get();
      const selected = state.nodes.filter(
        (node) =>
          node.selected && isRunnableNode(node) && node.parentId === undefined,
      );
      if (selected.length < 2) {
        return null;
      }
      const sized = sizedNodes(selected);
      const left = Math.min(...sized.map((item) => item.node.position.x));
      const top = Math.min(...sized.map((item) => item.node.position.y));
      const right = Math.max(
        ...sized.map((item) => item.node.position.x + item.width),
      );
      const bottom = Math.max(
        ...sized.map((item) => item.node.position.y + item.height),
      );
      const framePosition = {
        x: left - FRAME_PADDING_X,
        y: top - FRAME_HEADER_HEIGHT,
      };
      const frameId = crypto.randomUUID();
      const selectedIds = new Set(selected.map((node) => node.id));
      const frame: AppNode = {
        id: frameId,
        type: "complex",
        position: framePosition,
        width: Math.max(
          FRAME_MIN_WIDTH,
          right - left + FRAME_PADDING_X * 2,
        ),
        height: Math.max(
          FRAME_MIN_HEIGHT,
          bottom - top + FRAME_HEADER_HEIGHT + FRAME_PADDING_BOTTOM,
        ),
        selected: true,
        zIndex: -1,
        data: {
          label: frameLabel(state.nodes),
          templateName: null,
          templateCode: null,
          code: "",
          schema: emptySchema(frameId),
          widgetValues: {},
          isTrusted: true,
          color: "default",
          isFrame: true,
        },
      };
      pushHistory();
      set((current) => ({
        nodes: [
          frame,
          ...current.nodes.map((node) =>
            selectedIds.has(node.id)
              ? {
                  ...node,
                  selected: false,
                  parentId: frameId,
                  extent: "parent" as const,
                  position: {
                    x: node.position.x - framePosition.x,
                    y: node.position.y - framePosition.y,
                  },
                }
              : node.selected
                ? { ...node, selected: false }
                : node,
          ),
        ],
      }));
      return frameId;
    },

    ungroupSelection: () => {
      const state = get();
      const frames = state.nodes.filter(
        (node) => node.selected && isFrameNode(node),
      );
      if (frames.length === 0) {
        return false;
      }
      const frameIds = new Set(frames.map((frame) => frame.id));
      const nodesById = new Map(
        state.nodes.map((node) => [node.id, node]),
      );
      pushHistory();
      set((current) => ({
        nodes: current.nodes
          .filter((node) => !frameIds.has(node.id))
          .map((node) =>
            node.parentId && frameIds.has(node.parentId)
              ? {
                  ...node,
                  selected: true,
                  parentId: undefined,
                  extent: undefined,
                  position: absolutePosition(node, nodesById),
                }
              : node,
          ),
      }));
      return true;
    },

    createSubgraphFromSelection: () => {
      const state = get();
      const selectedAll = state.nodes.filter((node) => node.selected);
      if (
        selectedAll.length < 2 ||
        selectedAll.some(
          (node) =>
            isFrameNode(node) ||
            isSubgraphManagerNode(node) ||
            node.parentId !== undefined,
        )
      ) {
        return null;
      }
      const selected = selectedAll.filter(isRunnableNode);
      if (selected.length < 2) {
        return null;
      }
      const definitionId = crypto.randomUUID();
      const instanceId = crypto.randomUUID();
      const name = nextSubgraphName(state.subgraphs);
      let built: ReturnType<typeof buildSubgraphFromSelection>;
      try {
        built = buildSubgraphFromSelection({
          definitionId,
          instanceId,
          name,
          nodes: state.nodes,
          edges: state.edges,
          selectedIds: new Set(selected.map((node) => node.id)),
        });
      } catch {
        return null;
      }
      pushHistory();
      set((current) => ({
        nodes: built.nodes,
        edges: built.edges,
        subgraphs: {
          ...current.subgraphs,
          [definitionId]: built.definition,
        },
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        execution: idleExecution(),
      }));
      return instanceId;
    },

    addSubgraphInstance: (definitionId, position) => {
      const state = get();
      const definition = state.subgraphs[definitionId];
      if (!definition) {
        return null;
      }
      const active = state.scopeStack[state.scopeStack.length - 1];
      if (
        active &&
        subgraphDependsOn(state.subgraphs, definitionId, active.definitionId)
      ) {
        return null;
      }
      const id = crypto.randomUUID();
      const node = createSubgraphInstanceNode(id, definition, position);
      pushHistory();
      set((current) => ({
        nodes: [
          ...current.nodes.map((candidate) =>
            candidate.selected
              ? { ...candidate, selected: false }
              : candidate,
          ),
          node,
        ],
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        execution: idleExecution(),
      }));
      return id;
    },

    addSubgraphPort: (kind, dataType, suggestedName) => {
      const state = get();
      const active = state.scopeStack[state.scopeStack.length - 1];
      if (!active) {
        return null;
      }
      const managerId =
        kind === "input"
          ? SUBGRAPH_INPUT_NODE_ID
          : SUBGRAPH_OUTPUT_NODE_ID;
      const manager = state.nodes.find((node) => node.id === managerId);
      const definition = state.subgraphs[active.definitionId];
      if (!manager || !definition) {
        return null;
      }
      const existing =
        kind === "input"
          ? manager.data.schema.outputs
          : manager.data.schema.inputs;
      const name = uniqueSubgraphPortName(
        suggestedName ?? kind,
        new Set(existing.map((socket) => socket.name)),
      );
      const port = { name, data_type: dataType };
      const schema =
        kind === "input"
          ? {
              ...manager.data.schema,
              outputs: [...manager.data.schema.outputs, port],
            }
          : {
              ...manager.data.schema,
              inputs: [...manager.data.schema.inputs, port],
            };
      const inputs =
        kind === "input" ? [...definition.inputs, port] : definition.inputs;
      const outputs =
        kind === "output"
          ? [...definition.outputs, port]
          : definition.outputs;
      pushHistory();
      set((current) => ({
        nodes: current.nodes.map((node) =>
          node.id === managerId
            ? { ...node, data: { ...node.data, schema } }
            : node,
        ),
        subgraphs: {
          ...current.subgraphs,
          [active.definitionId]: {
            ...(current.subgraphs[active.definitionId] ?? definition),
            inputs,
            outputs,
          },
        },
      }));
      return name;
    },

    enterSubgraph: (nodeId) => {
      const state = get();
      const node = state.nodes.find((candidate) => candidate.id === nodeId);
      if (!node || !isSubgraphInstanceNode(node)) {
        return false;
      }
      const definitionId = node.data.subgraphId ?? "";
      const definition = state.subgraphs[definitionId];
      if (!definition) {
        return false;
      }
      if (
        state.scopeStack.some(
          (scope) => scope.definitionId === definitionId,
        )
      ) {
        return false;
      }
      const subgraphs = mergeActiveSubgraph(state);
      const rootGraph =
        state.scopeStack.length === 0
          ? { nodes: state.nodes, edges: state.edges }
          : state.rootGraph;
      if (!rootGraph) {
        return false;
      }
      const target = subgraphs[definitionId] ?? definition;
      const graph = graphForDefinition(target);
      set({
        nodes: graph.nodes,
        edges: graph.edges,
        subgraphs,
        rootGraph,
        scopeStack: [
          ...state.scopeStack,
          {
            definitionId,
            instanceId: node.id,
            name: node.data.label,
          },
        ],
        past: [],
        future: [],
        clipboard: null,
        historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        execution: idleExecution(),
      });
      return true;
    },

    exitSubgraph: () => {
      const { scopeStack } = get();
      if (scopeStack.length === 0) {
        return false;
      }
      return get().navigateToScope(scopeStack.length - 2);
    },

    navigateToScope: (depth) => {
      const state = get();
      if (depth >= state.scopeStack.length - 1) {
        return false;
      }
      if (depth < -1) {
        return false;
      }
      const subgraphs = syncSubgraphDefinitions(
        mergeActiveSubgraph(state),
      );
      let graph: RootGraph;
      let scopeStack: SubgraphScope[];
      if (depth < 0) {
        if (!state.rootGraph) {
          return false;
        }
        graph = {
          nodes: syncSubgraphInstanceSchemas(
            state.rootGraph.nodes,
            subgraphs,
          ),
          edges: state.rootGraph.edges,
        };
        scopeStack = [];
      } else {
        const scope = state.scopeStack[depth];
        const definition = scope
          ? subgraphs[scope.definitionId]
          : undefined;
        if (!scope || !definition) {
          return false;
        }
        graph = graphForDefinition({
          ...definition,
          nodes: syncSubgraphInstanceSchemas(
            definition.nodes,
            subgraphs,
          ),
        });
        scopeStack = state.scopeStack.slice(0, depth + 1);
      }
      set({
        nodes: graph.nodes.map((node) => ({ ...node, selected: false })),
        edges: graph.edges,
        subgraphs,
        scopeStack,
        rootGraph: depth < 0 ? null : state.rootGraph,
        past: [],
        future: [],
        clipboard: null,
        historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        execution: idleExecution(),
      });
      return true;
    },

    getWorkflowDocument: (name) => {
      const state = get();
      const graph = rootGraphFor(state);
      return serialiseWorkflow(
        graph.nodes,
        graph.edges,
        name === undefined ? state.projectName : name,
        mergeActiveSubgraph(state),
      );
    },

    getExecutionGraph: () => {
      const state = get();
      const graph = rootGraphFor(state);
      return flattenWorkflow(
        graph.nodes,
        graph.edges,
        mergeActiveSubgraph(state),
      );
    },

    renameNode: (nodeId, label) => {
      const normalized = label.trim();
      if (
        normalized.length === 0 ||
        normalized.length > MAX_NODE_LABEL_CHARS ||
        !get().nodes.some((node) => node.id === nodeId)
      ) {
        return false;
      }
      const current = get().nodes.find((node) => node.id === nodeId);
      if (current?.data.label === normalized) {
        return true;
      }
      pushHistory();
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, label: normalized } }
            : node,
        ),
      }));
      return true;
    },

    setNodeColor: (nodeIds, color) => {
      const targets = new Set(nodeIds);
      const affected = get().nodes.filter(
        (node) => targets.has(node.id) && node.data.color !== color,
      );
      if (affected.length === 0) {
        return false;
      }
      pushHistory();
      set((state) => ({
        nodes: state.nodes.map((node) =>
          targets.has(node.id)
            ? { ...node, data: { ...node.data, color } }
            : node,
        ),
      }));
      return true;
    },

    resetNode: (nodeId) => {
      const node = get().nodes.find((candidate) => candidate.id === nodeId);
      if (!node || isSubgraphManagerNode(node) || isSubgraphInstanceNode(node)) {
        return false;
      }
      let code = SCRATCH_NODE_CODE;
      let schema = emptySchema(nodeId);
      let label = "Scratch node";
      let templateCode: string | null = null;
      let isTrusted = false;
      let shouldParse = true;
      if (node.data.templateName !== null) {
        const template = get().templates.find(
          (candidate) => candidate.name === node.data.templateName,
        );
        if (!template) {
          return false;
        }
        code = template.code;
        schema = template.node_schema;
        label = template.title;
        templateCode = template.code;
        isTrusted = true;
        shouldParse = false;
      }
      pushHistory();
      set((state) => ({
        nodes: state.nodes.map((candidate) =>
          candidate.id === nodeId
            ? {
                ...candidate,
                data: {
                  ...candidate.data,
                  label,
                  templateCode,
                  code,
                  schema,
                  widgetValues: defaultWidgetValues(schema),
                  isTrusted,
                  color: "default",
                },
              }
            : candidate,
        ),
        edges: reconcileEdges(state.edges, nodeId, schema),
        execution: pruneExecution(state.execution, new Set([nodeId])),
      }));
      if (shouldParse) {
        void (async () => {
          try {
            const parsed = await parseCode(code, nodeId);
            set((state) => ({
              nodes: state.nodes.map((candidate) =>
                candidate.id === nodeId && candidate.data.code === code
                  ? {
                      ...candidate,
                      data: {
                        ...candidate.data,
                        schema: parsed,
                        widgetValues: defaultWidgetValues(parsed),
                      },
                    }
                  : candidate,
              ),
            }));
          } catch {
            return;
          }
        })();
      }
      return true;
    },

    clearCanvas: () => {
      if (get().nodes.length === 0 && get().edges.length === 0) {
        return;
      }
      pushHistory();
      set((state) => {
        if (state.scopeStack.length === 0) {
          return {
            nodes: [],
            edges: [],
            subgraphs: {},
            rootGraph: null,
            editingNodeId: null,
            pendingApproval: null,
            approvalError: null,
            execution: idleExecution(),
          };
        }
        const candidates = new Set(
          state.nodes
            .filter(isSubgraphInstanceNode)
            .map((node) => node.data.subgraphId)
            .filter((id): id is string => Boolean(id)),
        );
        return {
          nodes: [],
          edges: [],
          subgraphs:
            candidates.size > 0
              ? pruneUnusedSubgraphs(state, [], [], candidates)
              : state.subgraphs,
          editingNodeId: null,
          pendingApproval: null,
          approvalError: null,
          execution: idleExecution(),
        };
      });
    },

    checkpoint: () => {
      pushHistory();
    },

    undo: () => {
      if (get().execution.status === "running") {
        return;
      }
      const { past, future } = get();
      if (past.length === 0) {
        return;
      }
      const previous = past[past.length - 1];
      const snapshot = snapshotGraph();
      set((state) => ({
        past: past.slice(0, -1),
        future: [...future, snapshot],
        nodes: previous.nodes,
        edges: previous.edges,
        subgraphs: previous.subgraphs,
        scopeStack: previous.scopeStack,
        rootGraph: previous.rootGraph,
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        historyMeta: { ...state.historyMeta, coalesceKey: null, dragging: false },
      }));
    },

    redo: () => {
      if (get().execution.status === "running") {
        return;
      }
      const { past, future } = get();
      if (future.length === 0) {
        return;
      }
      const next = future[future.length - 1];
      const snapshot = snapshotGraph();
      set((state) => ({
        past: [...past, snapshot],
        future: future.slice(0, -1),
        nodes: next.nodes,
        edges: next.edges,
        subgraphs: next.subgraphs,
        scopeStack: next.scopeStack,
        rootGraph: next.rootGraph,
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        historyMeta: { ...state.historyMeta, coalesceKey: null, dragging: false },
      }));
    },

    copySelection: () => {
      const { nodes, edges } = get();
      const selected = nodes.filter(
        (node) => node.selected && !isSubgraphManagerNode(node),
      );
      if (selected.length === 0) {
        return false;
      }
      const nodesById = new Map(nodes.map((node) => [node.id, node]));
      const ids = new Set(selected.map((node) => node.id));
      let expanded = true;
      while (expanded) {
        expanded = false;
        for (const node of nodes) {
          if (node.parentId && ids.has(node.parentId) && !ids.has(node.id)) {
            ids.add(node.id);
            expanded = true;
          }
        }
      }
      const copied = nodes
        .filter((node) => ids.has(node.id))
        .map((node) => {
          if (node.parentId && !ids.has(node.parentId)) {
            return {
              ...node,
              parentId: undefined,
              extent: undefined,
              position: absolutePosition(node, nodesById),
              data: { ...node.data },
            };
          }
          return { ...node, data: { ...node.data } };
        });
      set({
        clipboard: {
          nodes: copied,
          edges: edges.filter(
            (edge) => ids.has(edge.source) && ids.has(edge.target),
          ),
          subgraphs: mergeActiveSubgraph(get()),
          scopeStack: get().scopeStack,
          rootGraph: get().rootGraph,
        },
      });
      return true;
    },

    cutSelection: () => {
      if (!get().copySelection()) {
        return false;
      }
      const copied = get().clipboard?.nodes ?? [];
      pushHistory();
      const removed = new Set(copied.map((node) => node.id));
      const candidates = new Set(
        copied
          .filter(isSubgraphInstanceNode)
          .map((node) => node.data.subgraphId)
          .filter((id): id is string => Boolean(id)),
      );
      set((state) => {
        const nodes = state.nodes.filter(
          (node) => !removed.has(node.id),
        );
        const edges = state.edges.filter(
          (edge) => !removed.has(edge.source) && !removed.has(edge.target),
        );
        return {
          nodes,
          edges,
          subgraphs:
            candidates.size > 0
              ? pruneUnusedSubgraphs(state, nodes, edges, candidates)
              : state.subgraphs,
          editingNodeId:
            state.editingNodeId !== null &&
            removed.has(state.editingNodeId)
              ? null
              : state.editingNodeId,
          execution: pruneExecution(state.execution, removed),
        };
      });
      return true;
    },

    pasteClipboard: async (anchor) => {
      const clip = get().clipboard;
      if (!clip || clip.nodes.length === 0) {
        return [];
      }
      pushHistory();
      const templates = get().templates;
      const clipById = new Map(clip.nodes.map((node) => [node.id, node]));
      const absolute = new Map(
        clip.nodes.map((node) => [
          node.id,
          absolutePosition(node, clipById),
        ]),
      );
      const minX = Math.min(
        ...clip.nodes.map((node) => absolute.get(node.id)?.x ?? 0),
      );
      const minY = Math.min(
        ...clip.nodes.map((node) => absolute.get(node.id)?.y ?? 0),
      );
      const offset = anchor
        ? { x: anchor.x - minX, y: anchor.y - minY }
        : { x: PASTE_OFFSET, y: PASTE_OFFSET };
      const idMap = new Map<string, string>();
      for (const node of clip.nodes) {
        idMap.set(node.id, crypto.randomUUID());
      }
      const hashCache = new Map<string, Promise<string>>();
      const codeHash = (code: string): Promise<string> => {
        let promise = hashCache.get(code);
        if (!promise) {
          promise = sha256Hex(code);
          hashCache.set(code, promise);
        }
        return promise;
      };
      const newNodes: AppNode[] = [];
      for (const node of clip.nodes) {
        const hash = await codeHash(node.data.code);
        const isOfficial = templates.some(
          (template) => template.code_hash === hash,
        );
        const newParentId = node.parentId
          ? idMap.get(node.parentId)
          : undefined;
        const position = absolute.get(node.id) ?? node.position;
        const parentPosition = newParentId
          ? absolute.get(node.parentId ?? "")
          : undefined;
        newNodes.push({
          ...node,
          id: idMap.get(node.id) as string,
          parentId: newParentId,
          extent: newParentId ? "parent" : undefined,
          selected: true,
          position: parentPosition
            ? {
                x: position.x - parentPosition.x,
                y: position.y - parentPosition.y,
              }
            : { x: position.x + offset.x, y: position.y + offset.y },
          data: {
            ...node.data,
            isTrusted: isFrameNode(node) ? true : isOfficial,
          },
        });
      }
      newNodes.sort(
        (a, b) => Number(Boolean(a.parentId)) - Number(Boolean(b.parentId)),
      );
      const newEdges: AppEdge[] = clip.edges.map((edge) => {
        const source = idMap.get(edge.source) as string;
        const target = idMap.get(edge.target) as string;
        return {
          ...edge,
          id: `${source}:${edge.sourceHandle}->${target}:${edge.targetHandle}`,
          source,
          target,
        };
      });
      set((state) => ({
        nodes: [
          ...state.nodes.map((node) =>
            node.selected ? { ...node, selected: false } : node,
          ),
          ...newNodes,
        ],
        edges: [...state.edges, ...newEdges],
      }));
      return newNodes.map((node) => node.id);
    },

    duplicateSelection: async () => {
      if (!get().copySelection()) {
        return [];
      }
      return get().pasteClipboard();
    },

    onNodesChange: (changes) => {
      const managerIds = new Set(
        get().nodes.filter(isSubgraphManagerNode).map((node) => node.id),
      );
      const safeChanges = changes.filter(
        (change) =>
          !(change.type === "remove" && managerIds.has(change.id)),
      );
      const removed = safeChanges
        .filter((change) => change.type === "remove")
        .map((change) => change.id);
      const dragStart = safeChanges.some(
        (change) => change.type === "position" && change.dragging === true,
      );
      const dragEnd = safeChanges.some(
        (change) => change.type === "position" && change.dragging === false,
      );
      if (dragStart && !get().historyMeta.dragging) {
        pushHistory();
        set((state) => ({
          historyMeta: { ...state.historyMeta, dragging: true },
        }));
      }
      set((state) => {
        const removedSet = new Set(removed);
        const frames = new Map(
          state.nodes
            .filter(
              (node) => removedSet.has(node.id) && isFrameNode(node),
            )
            .map((node) => [node.id, node]),
        );
        const nodesById = new Map(
          state.nodes.map((node) => [node.id, node]),
        );
        const nodes = applyNodeChanges<AppNode>(safeChanges, state.nodes).map(
          (node) => {
            if (!node.parentId || !frames.has(node.parentId)) {
              return node;
            }
            return {
              ...node,
              parentId: undefined,
              extent: undefined,
              position: absolutePosition(node, nodesById),
            };
          },
        );
        const candidates = new Set(
          state.nodes
            .filter(
              (node) =>
                removedSet.has(node.id) &&
                isSubgraphInstanceNode(node) &&
                node.data.subgraphId,
            )
            .map((node) => node.data.subgraphId as string),
        );
        const edges =
          removed.length > 0
            ? state.edges.filter(
                (edge) =>
                  !removedSet.has(edge.source) &&
                  !removedSet.has(edge.target),
              )
            : state.edges;
        const next: Partial<GraphStore> = { nodes };
        if (removed.length > 0) {
          next.edges = edges;
          next.execution = pruneExecution(
            state.execution,
            removedSet,
          );
        }
        if (candidates.size > 0) {
          next.subgraphs = pruneUnusedSubgraphs(
            state,
            nodes,
            edges,
            candidates,
          );
        }
        if (dragEnd) {
          next.historyMeta = { ...state.historyMeta, dragging: false };
        }
        return next;
      });
    },

    onEdgesChange: (changes) => {
      set((state) => ({
        edges: applyEdgeChanges(changes, state.edges),
      }));
    },

    reassignOutputBundle: (edgeIds, sourceNodeId, sourceHandleId) => {
      const state = get();
      const sourceNode = state.nodes.find((node) => node.id === sourceNodeId);
      if (!sourceNode) {
        return false;
      }
      const sourceSocket = sourceNode.data.schema.outputs.find(
        (socket) => socket.name === sourceHandleId,
      );
      if (!sourceSocket) {
        return false;
      }
      const ids = new Set(edgeIds);
      const selected = state.edges.filter((edge) => ids.has(edge.id));
      if (selected.length !== ids.size) {
        return false;
      }
      pushHistory();
      set((current) => ({
        edges: current.edges.map((edge) => {
          if (!ids.has(edge.id)) {
            return edge;
          }
          return {
            ...edge,
            id: `${sourceNodeId}:${sourceHandleId}->${edge.target}:${edge.targetHandle}`,
            source: sourceNodeId,
            sourceHandle: sourceHandleId,
          };
        }),
      }));
      return true;
    },

    reconnectEdgeTarget: (edgeId, targetNodeId, targetHandleId) => {
      const state = get();
      const edge = state.edges.find((candidate) => candidate.id === edgeId);
      const targetNode = state.nodes.find((node) => node.id === targetNodeId);
      if (!edge || !targetNode || targetNodeId === edge.source) {
        return false;
      }
      const sourceNode = state.nodes.find((node) => node.id === edge.source);
      const sourceSocket = sourceNode?.data.schema.outputs.find(
        (socket) => socket.name === edge.sourceHandle,
      );
      const targetSocket = resolveTargetSocket(
        targetNode.data.schema,
        targetHandleId,
      );
      if (
        !sourceSocket ||
        !targetSocket ||
        !socketTypesCompatible(sourceSocket.data_type, targetSocket.data_type)
      ) {
        return false;
      }
      if (
        edge.target === targetNodeId &&
        edge.targetHandle === targetHandleId
      ) {
        return true;
      }
      const occupied = state.edges.some(
        (candidate) =>
          candidate.id !== edge.id &&
          candidate.target === targetNodeId &&
          candidate.targetHandle === targetHandleId,
      );
      if (occupied) {
        return false;
      }
      pushHistory();
      set((current) => ({
        edges: current.edges.map((candidate) =>
          candidate.id === edge.id
            ? {
                ...candidate,
                id: `${candidate.source}:${candidate.sourceHandle}->${targetNodeId}:${targetHandleId}`,
                target: targetNodeId,
                targetHandle: targetHandleId,
              }
            : candidate,
        ),
      }));
      return true;
    },

    deleteEdge: (edgeId) => {
      if (!get().edges.some((edge) => edge.id === edgeId)) {
        return false;
      }
      pushHistory();
      set((state) => ({
        edges: state.edges.filter((edge) => edge.id !== edgeId),
      }));
      return true;
    },

    onConnect: (connection) => {
      const { source, target } = connection;
      let { sourceHandle, targetHandle } = connection;
      if (!source || !target || !sourceHandle || !targetHandle) {
        return;
      }
      if (source === target) {
        return;
      }
      let state = get();
      let sourceNode = state.nodes.find((node) => node.id === source);
      let targetNode = state.nodes.find((node) => node.id === target);
      if (!sourceNode || !targetNode) {
        return;
      }
      let createdPort = false;
      if (
        sourceNode.data.subgraphRole === "input" &&
        sourceHandle === NEW_SUBGRAPH_INPUT_HANDLE
      ) {
        if (targetNode.data.subgraphRole) {
          return;
        }
        const targetSocket = resolveTargetSocket(
          targetNode.data.schema,
          targetHandle,
        );
        if (
          !targetSocket ||
          isInputTaken(state.edges, target, targetHandle)
        ) {
          return;
        }
        const port = get().addSubgraphPort(
          "input",
          targetSocket.data_type,
          targetHandle,
        );
        if (!port) {
          return;
        }
        sourceHandle = port;
        createdPort = true;
        state = get();
        sourceNode = state.nodes.find((node) => node.id === source);
        targetNode = state.nodes.find((node) => node.id === target);
      } else if (
        targetNode.data.subgraphRole === "output" &&
        targetHandle === NEW_SUBGRAPH_OUTPUT_HANDLE
      ) {
        if (sourceNode.data.subgraphRole) {
          return;
        }
        const sourceSocket = sourceNode.data.schema.outputs.find(
          (socket) => socket.name === sourceHandle,
        );
        if (!sourceSocket) {
          return;
        }
        const port = get().addSubgraphPort(
          "output",
          sourceSocket.data_type,
          sourceHandle,
        );
        if (!port) {
          return;
        }
        targetHandle = port;
        createdPort = true;
        state = get();
        sourceNode = state.nodes.find((node) => node.id === source);
        targetNode = state.nodes.find((node) => node.id === target);
      }
      if (!sourceNode || !targetNode) {
        return;
      }
      if (
        sourceNode.data.subgraphRole === "input" &&
        targetNode.data.subgraphRole === "output"
      ) {
        return;
      }
      const sourceSocket = sourceNode.data.schema.outputs.find(
        (socket) => socket.name === sourceHandle,
      );
      const targetSocket = resolveTargetSocket(
        targetNode.data.schema,
        targetHandle,
      );
      if (!sourceSocket || !targetSocket) {
        return;
      }
      const compatible = socketTypesCompatible(
        sourceSocket.data_type,
        targetSocket.data_type,
      );
      const inputTaken = isInputTaken(state.edges, target, targetHandle);
      if (!compatible || inputTaken) {
        return;
      }
      const targetWidget = targetNode.data.schema.widgets.find(
        (widget) => widget.name === targetHandle,
      );
      const inheritedValue =
        isOfficialValueNode(sourceNode.data) &&
        targetWidget &&
        (targetWidget.options?.length ?? 0) > 0
          ? normalizeValueChoice(
              sourceNode.data.widgetValues.value,
              targetWidget.options ?? [],
            )
          : sourceNode.data.widgetValues.value;
      const edge: AppEdge = {
        id: `${source}:${sourceHandle}->${target}:${targetHandle}`,
        source,
        target,
        sourceHandle,
        targetHandle,
      };
      if (!createdPort) {
        pushHistory();
      }
      set((current) => ({
        nodes: Object.is(inheritedValue, sourceNode.data.widgetValues.value)
          ? current.nodes
          : current.nodes.map((node) =>
              node.id === source
                ? {
                    ...node,
                    data: {
                      ...node.data,
                      widgetValues: {
                        ...node.data.widgetValues,
                        value: inheritedValue,
                      },
                    },
                  }
                : node,
            ),
        edges: addEdge(edge, current.edges),
      }));
    },

    openEditor: (nodeId) => {
      const node = get().nodes.find((candidate) => candidate.id === nodeId);
      if (
        !node ||
        isSubgraphManagerNode(node) ||
        isSubgraphInstanceNode(node)
      ) {
        return;
      }
      set({ editingNodeId: nodeId });
    },

    closeEditor: () => set({ editingNodeId: null }),

    updateWidgetValue: (nodeId, name, value) => {
      const node = get().nodes.find((candidate) => candidate.id === nodeId);
      if (!node || node.data.widgetValues[name] === value) {
        return;
      }
      pushHistory(`widget:${nodeId}:${name}`);
      set((state) => ({
        nodes: state.nodes.map((item) =>
          item.id === nodeId
            ? {
                ...item,
                data: {
                  ...item.data,
                  widgetValues: { ...item.data.widgetValues, [name]: value },
                },
              }
            : item,
        ),
      }));
    },

    saveNodeCode: async (nodeId, code, schema) => {
      const hash = await sha256Hex(code);
      const state = get();
      const isOfficial = state.templates.some(
        (template) => template.code_hash === hash,
      );
      const isSessionApproved = state.approvedHashes.has(`${nodeId}:${hash}`);
      if (isOfficial || isSessionApproved) {
        pushHistory();
        commitNodeCode(nodeId, code, schema, true);
        set({ editingNodeId: null });
        return;
      }
      set({
        pendingApproval: { nodeId, code, codeHash: hash, schema },
        approvalError: null,
        editingNodeId: null,
      });
    },

    approvePending: async () => {
      const pending = get().pendingApproval;
      if (!pending) {
        return;
      }
      set({ approvalBusy: true, approvalError: null });
      try {
        await approveNode(pending.nodeId, pending.codeHash);
        set((state) => ({
          approvedHashes: new Set(state.approvedHashes).add(
            `${pending.nodeId}:${pending.codeHash}`,
          ),
        }));
        pushHistory();
        commitNodeCode(pending.nodeId, pending.code, pending.schema, true);
        set({ pendingApproval: null, approvalBusy: false, editingNodeId: null });
      } catch (error) {
        set({
          approvalBusy: false,
          approvalError:
            error instanceof Error ? error.message : String(error),
        });
      }
    },

    rejectPending: () => set({ pendingApproval: null, approvalError: null }),

    restoreWorkflow: async (saved) => {
      pushHistory();
      const templates = get().templates;
      const templateByName = new Map(
        templates.map((template) => [template.name, template]),
      );
      const savedSubgraphs = saved.subgraphs ?? {};

      const buildNode = async (
        savedNode: SavedWorkflow["nodes"][number],
      ): Promise<AppNode> => {
        if (savedNode.isFrame) {
          return {
            id: savedNode.id,
            type: "complex",
            position: savedNode.position,
            width: savedNode.width ?? FRAME_MIN_WIDTH,
            height: savedNode.height ?? FRAME_MIN_HEIGHT,
            parentId: savedNode.parentId,
            extent: savedNode.parentId ? "parent" : undefined,
            zIndex: -1,
            data: {
              label: savedNode.label,
              templateName: null,
              templateCode: null,
              code: "",
              schema: emptySchema(savedNode.id),
              widgetValues: {},
              isTrusted: true,
              color: savedNode.color ?? "default",
              isFrame: true,
            },
          };
        }
        if (savedNode.isSubgraph) {
          const definition = savedSubgraphs[savedNode.subgraphId ?? ""];
          if (!definition) {
            throw new Error(
              `Node '${savedNode.id}' references a missing subgraph.`,
            );
          }
          return {
            id: savedNode.id,
            type: "complex",
            position: savedNode.position,
            width: savedNode.width,
            height: savedNode.height,
            parentId: savedNode.parentId,
            extent: savedNode.parentId ? "parent" : undefined,
            data: {
              label: savedNode.label,
              templateName: null,
              templateCode: null,
              code: "",
              schema: emptySchema(
                savedNode.id,
                definition.inputs,
                definition.outputs,
              ),
              widgetValues: {},
              isTrusted: true,
              color: savedNode.color ?? "default",
              isSubgraph: true,
              subgraphId: definition.id,
            },
          };
        }
        let schema: NodeSchema;
        try {
          schema = await parseCode(savedNode.code, savedNode.id);
        } catch (error) {
          schema = {
            ...emptySchema(savedNode.id),
            syntax_errors: [
              {
                line: 1,
                column: 0,
                message: `Failed to re-parse restored code: ${
                  error instanceof Error ? error.message : String(error)
                }`,
              },
            ],
          };
        }
        const template = savedNode.templateName
          ? (templateByName.get(savedNode.templateName) ?? null)
          : null;
        const hash = await sha256Hex(savedNode.code);
        const isOfficial = templates.some(
          (candidate) => candidate.code_hash === hash,
        );
        return {
          id: savedNode.id,
          type: "complex",
          position: savedNode.position,
          width: savedNode.width,
          height: savedNode.height,
          parentId: savedNode.parentId,
          extent: savedNode.parentId ? "parent" : undefined,
          data: {
            label: savedNode.label,
            templateName: template?.name ?? savedNode.templateName,
            templateCode: template?.code ?? null,
            code: savedNode.code,
            schema,
            widgetValues: mergeWidgetValues(schema, savedNode.widgetValues),
            isTrusted: isOfficial,
            color: savedNode.color ?? "default",
          },
        };
      };

      const buildGraph = async (
        nodes: SavedWorkflow["nodes"],
        edges: SavedWorkflow["edges"],
        definition: boolean,
      ): Promise<RootGraph> => {
        const built = await Promise.all(nodes.map(buildNode));
        built.sort(
          (a, b) => Number(Boolean(a.parentId)) - Number(Boolean(b.parentId)),
        );
        const nodeIds = new Set(built.map((node) => node.id));
        const allowed = new Set(nodeIds);
        if (definition) {
          allowed.add(SUBGRAPH_INPUT_NODE_ID);
          allowed.add(SUBGRAPH_OUTPUT_NODE_ID);
        }
        let builtEdges: AppEdge[] = edges
          .filter(
            (edge) => allowed.has(edge.source) && allowed.has(edge.target),
          )
          .map((edge) => ({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            sourceHandle: edge.sourceHandle,
            targetHandle: edge.targetHandle,
          }));
        for (const node of built) {
          builtEdges = reconcileEdges(
            builtEdges,
            node.id,
            node.data.schema,
          );
        }
        return {
          nodes: normalizeValueNodeValues(built, builtEdges),
          edges: builtEdges,
        };
      };

      const subgraphs: SubgraphMap = {};
      for (const [id, definition] of Object.entries(savedSubgraphs)) {
        const graph = await buildGraph(
          definition.nodes,
          definition.edges,
          true,
        );
        subgraphs[id] = {
          id: definition.id,
          name: definition.name,
          inputs: definition.inputs,
          outputs: definition.outputs,
          nodes: graph.nodes,
          edges: graph.edges,
          inputPosition: definition.inputPosition,
          outputPosition: definition.outputPosition,
        };
      }
      const root = await buildGraph(saved.nodes, saved.edges, false);

      set((state) => ({
        nodes: root.nodes,
        edges: root.edges,
        subgraphs,
        scopeStack: [],
        rootGraph: null,
        projectName: saved.name ?? null,
        projectSlug: null,
        saveDialogOpen: false,
        saveError: null,
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        execution: idleExecution(),
        windows: state.windows.map((window) =>
          window.id === state.activeWindowId
            ? {
                ...window,
                nodes: root.nodes,
                edges: root.edges,
                subgraphs,
                scopeStack: [],
                rootGraph: null,
                execution: idleExecution(),
                projectName: saved.name ?? null,
                projectSlug: null,
                title: saved.name ?? window.title,
              }
            : window,
        ),
      }));
    },

    restoreFromStorage: async () => {
      if (get().nodes.length > 0) {
        return;
      }

      const workspace = loadLocalWorkspace();
      if (workspace) {
        const templates = get().templates;
        const hydrated: WorkflowWindow[] = [];
        for (const savedWindow of workspace.windows) {
          try {
            const built = await buildSavedWorkflow(
              savedWindow.workflow,
              templates,
            );
            hydrated.push({
              id: savedWindow.id,
              title: savedWindow.projectName ?? savedWindow.title,
              ...emptyWindowState(),
              ...built,
              projectName: savedWindow.projectName,
              projectSlug: savedWindow.projectSlug,
              execution: idleExecution(),
            });
          } catch {
            continue;
          }
        }
        if (hydrated.length > 0) {
          const active =
            hydrated.find(
              (window) => window.id === workspace.activeWindowId,
            ) ?? hydrated[0];
          if (active) {
            set({
              windows: hydrated,
              activeWindowId: active.id,
              executionWindowId: null,
              clipboard: null,
              ...windowTopLevel(active),
            });
            saveLocalWorkflow(
              get().getWorkflowDocument(active.projectName),
            );
            saveLocalProjectIdentity({
              name: active.projectName,
              slug: active.projectSlug,
            });
            return;
          }
        }
      }

      const saved = loadLocalWorkflow();
      if (
        saved &&
        (saved.nodes.length > 0 ||
          Object.keys(saved.subgraphs ?? {}).length > 0)
      ) {
        await get().restoreWorkflow(saved);
        const identity = loadLocalProjectIdentity();
        set((state) => {
          const projectName = identity?.name ?? saved.name ?? null;
          const projectSlug = identity?.slug ?? null;
          return {
            projectName,
            projectSlug,
            past: [],
            future: [],
            windows: state.windows.map((window) =>
              window.id === state.activeWindowId
                ? {
                    ...window,
                    projectName,
                    projectSlug,
                    title: projectName ?? window.title,
                    past: [],
                    future: [],
                  }
                : window,
            ),
          };
        });
        return;
      }

      let session: WorkspaceSession | null;
      try {
        session = await fetchWorkspaceSession();
      } catch {
        return;
      }
      if (
        !session ||
        (session.workflow.nodes.length === 0 &&
          Object.keys(session.workflow.subgraphs ?? {}).length === 0)
      ) {
        return;
      }
      await get().restoreWorkflow(session.workflow);
      const projectName =
        session.project_name ?? session.workflow.name ?? null;
      set((state) => ({
        projectName,
        projectSlug: session.project_slug,
        past: [],
        future: [],
        windows: state.windows.map((window) =>
          window.id === state.activeWindowId
            ? {
                ...window,
                projectName,
                projectSlug: session.project_slug,
                title: projectName ?? window.title,
                past: [],
                future: [],
              }
            : window,
        ),
      }));
      const state = get();
      saveLocalWorkflow(state.getWorkflowDocument(projectName));
      saveLocalProjectIdentity({
        name: projectName,
        slug: session.project_slug,
      });
    },

    runGraph: async (targets?: string[]) => {
      const state = get();
      if (state.execution.status === "running") {
        return;
      }
      const otherRunning = state.windows.some(
        (window) =>
          window.id !== state.activeWindowId &&
          window.execution.status === "running",
      );
      if (otherRunning) {
        set((current) => ({
          execution: {
            ...current.execution,
            status: "idle",
            error: {
              message:
                "Another workflow window is running. Stop it or switch to that window first.",
              errorType: "LocalValidation",
              nodeId: null,
              line: null,
              traceback: "",
            },
          },
        }));
        return;
      }
      if (state.scopeStack.length > 0) {
        set((current) => ({
          execution: {
            ...current.execution,
            status: "idle",
            error: {
              message:
                "Exit the subgraph to run the workflow from the root canvas.",
              errorType: "LocalValidation",
              nodeId: null,
              line: null,
              traceback: "",
            },
          },
        }));
        return;
      }
      let flattened: FlattenedWorkflow;
      try {
        flattened = state.getExecutionGraph();
      } catch (error) {
        set((current) => ({
          execution: {
            ...current.execution,
            status: "idle",
            error: {
              message: error instanceof Error ? error.message : String(error),
              errorType: "Subgraph",
              nodeId: null,
              line: null,
              traceback: "",
            },
          },
        }));
        return;
      }
      if (flattened.graph.nodes.length === 0) {
        set((current) => ({
          execution: {
            ...current.execution,
            status: "idle",
            error: {
              message:
                state.nodes.length === 0
                  ? "Canvas is empty — drag a template node in first."
                  : "No executable nodes selected.",
              errorType: "LocalValidation",
              nodeId: null,
              line: null,
              traceback: "",
            },
          },
        }));
        return;
      }
      const candidates = state.nodes.filter(
        (node) =>
          isRunnableNode(node) &&
          !isSubgraphInstanceNode(node) &&
          (!targets || targets.includes(node.id)),
      );
      const broken = candidates.find(
        (node) => node.data.schema.syntax_errors.length > 0,
      );
      if (broken) {
        const firstError = broken.data.schema.syntax_errors[0];
        set((state) => ({
          execution: {
            ...state.execution,
            status: "idle",
            error: {
              message: `Node "${broken.data.label}" contains syntax errors`,
              errorType: "LocalValidation",
              nodeId: broken.id,
              line: firstError ? firstError.line : null,
              traceback: "",
            },
          },
        }));
        get().openEditor(broken.id);
        return;
      }

      set({
        execution: { ...idleExecution(), status: "running" },
        executionWindowId: state.activeWindowId,
      });
      const scopedTargets = targets?.filter((target) =>
        flattened.graph.nodes.some((node) => node.id === target),
      );
      try {
        const response: ExecuteResponse = await executeGraph(
          flattened.graph,
          scopedTargets,
          flattened.labels,
        );
        set((state) =>
          state.execution.status === "running"
            ? {
                execution: {
                  ...state.execution,
                  runId: state.execution.runId ?? response.run_id,
                },
              }
            : {},
        );
      } catch (error) {
        if (error instanceof ApiError) {
          const untrusted = untrustedNodesFrom(error);
          if (untrusted.length > 0) {
            set((state) => ({
              nodes: state.nodes.map((node) =>
                untrusted.some((ref) => ref.node_id === node.id)
                  ? { ...node, data: { ...node.data, isTrusted: false } }
                  : node,
              ),
              execution: {
                ...state.execution,
                status: "idle",
                error: {
                  message: `${untrusted.length} node(s) contain untrusted code — open and approve them before running.`,
                  errorType: "UntrustedCode",
                  nodeId: untrusted[0]?.node_id ?? null,
                  line: null,
                  traceback: "",
                },
              },
            }));
            return;
          }
          const modelIssues = modelSelectionIssuesFrom(error);
          if (modelIssues.length > 0) {
            const first = modelIssues[0];
            set((state) => ({
              execution: {
                ...state.execution,
                status: "idle",
                error: {
                  message:
                    modelIssues.length === 1
                      ? `Local model unavailable for ${first.node_id}.${first.widget}: ${first.message}`
                      : `${modelIssues.length} local model selections are unavailable. Fix them before running.`,
                  errorType: "ModelSelection",
                  nodeId: first.node_id,
                  line: null,
                  traceback: "",
                },
              },
            }));
            return;
          }
          const syntax = syntaxErrorFrom(error);
          set((state) => ({
            execution: {
              ...state.execution,
              status: "idle",
              error: {
                message: syntax
                  ? `Syntax error (line ${syntax.line}): ${syntax.message}`
                  : error.message,
                errorType: "Preflight",
                nodeId: syntax ? syntax.nodeId : null,
                line: syntax ? syntax.line : null,
                traceback: "",
              },
            },
          }));
          if (syntax && syntax.nodeId) {
            get().openEditor(syntax.nodeId);
          }
          return;
        }
        set((state) => ({
          execution: {
            ...state.execution,
            status: "idle",
            error: {
              message: error instanceof Error ? error.message : String(error),
              errorType: "Unknown",
              nodeId: null,
              line: null,
              traceback: "",
            },
          },
        }));
      }
    },

    runSweep: async (request, onProgress) => {
      const initial = get();
      if (initial.execution.status === "running") {
        return { completed: 0, total: request.values.length, status: "failed" };
      }
      const target = initial.nodes.find((node) => node.id === request.nodeId);
      if (!target) {
        return { completed: 0, total: request.values.length, status: "failed" };
      }
      const originalValue = target.data.widgetValues[request.widget];
      const saveNodes = request.suffixOutputs
        ? initial.nodes.filter(
            (node) =>
              node.data.templateName === "save_media" &&
              typeof node.data.widgetValues.path === "string",
          )
        : [];
      const originalPaths = new Map(
        saveNodes.map((node) => [node.id, node.data.widgetValues.path]),
      );
      let completed = 0;
      let status: ExecutionStatus = "idle";
      const waitForTerminal = (runId: string | null) =>
        new Promise<ExecutionStatus>((resolve) => {
          let unsubscribe: () => void = () => undefined;
          const finish = (next: ExecutionStatus) => {
            unsubscribe();
            resolve(next);
          };
          unsubscribe = useGraphStore.subscribe((state) => {
            const execution = state.execution;
            if (
              execution.runId === runId &&
              (execution.status === "completed" ||
                execution.status === "failed" ||
                execution.status === "cancelled")
            ) {
              finish(execution.status);
            }
          });
          const execution = useGraphStore.getState().execution;
          if (
            execution.runId === runId &&
            (execution.status === "completed" ||
              execution.status === "failed" ||
              execution.status === "cancelled")
          ) {
            finish(execution.status);
          }
        });

      try {
        for (const [index, value] of request.values.entries()) {
          get().setWidgetValueForSweep(request.nodeId, request.widget, value);
          if (request.suffixOutputs) {
            const safe = String(value)
              .replace(/[^a-zA-Z0-9_-]+/g, "_")
              .replace(/^_+|_+$/g, "")
              .slice(0, 32);
            for (const node of saveNodes) {
              const original = String(originalPaths.get(node.id) ?? "media.png");
              const dot = original.lastIndexOf(".");
              const stem = dot > 0 ? original.slice(0, dot) : original;
              const extension = dot > 0 ? original.slice(dot) : "";
              get().setWidgetValueForSweep(
                node.id,
                "path",
                `${stem}_sweep_${index + 1}_${safe || "value"}${extension}`,
              );
            }
          }
          await get().runGraph();
          const current = get().execution;
          if (current.status === "idle") {
            status = "failed";
            break;
          }
          if (current.runId === null) {
            status = current.status;
            break;
          }
          status = await waitForTerminal(current.runId);
          if (status !== "completed") {
            break;
          }
          completed += 1;
          onProgress?.(completed, request.values.length);
        }
      } finally {
        get().setWidgetValueForSweep(
          request.nodeId,
          request.widget,
          originalValue,
        );
        for (const [nodeId, path] of originalPaths) {
          get().setWidgetValueForSweep(nodeId, "path", path);
        }
      }
      return { completed, total: request.values.length, status };
    },

    setWidgetValueForSweep: (nodeId, name, value) => {
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  widgetValues: { ...node.data.widgetValues, [name]: value },
                },
              }
            : node,
        ),
      }));
    },

    cancelRun: async () => {
      if (get().execution.status !== "running") {
        return;
      }
      try {
        await cancelExecution();
      } catch (error) {
        if (error instanceof ApiError && error.status === 409) {
          return;
        }
        set((state) => ({
          execution: {
            ...state.execution,
            error: {
              message: error instanceof Error ? error.message : String(error),
              errorType: "Cancel",
              nodeId: null,
              line: null,
              traceback: "",
            },
          },
        }));
      }
    },

    applyExecutionEvent: (event) => {
      const current = get();
      if (
        current.executionWindowId !== null &&
        current.executionWindowId !== current.activeWindowId
      ) {
        const target = current.windows.find(
          (window) => window.id === current.executionWindowId,
        );
        if (target) {
          const execution = reduceExecutionEvent(target.execution, event);
          const nodes = advanceSeedNodes(target.nodes, event);
          set((state) => ({
            windows: state.windows.map((window) =>
              window.id === target.id
                ? {
                    ...window,
                    execution,
                    ...(nodes !== target.nodes ? { nodes } : {}),
                  }
                : window,
            ),
          }));
          return;
        }
      }
      set((state) => {
        const execution = reduceExecutionEvent(state.execution, event);
        const nodes = advanceSeedNodes(state.nodes, event);
        if (execution === state.execution && nodes === state.nodes) {
          return {};
        }
        return {
          ...(execution !== state.execution ? { execution } : {}),
          ...(nodes !== state.nodes ? { nodes } : {}),
        };
      });
    },

    syncRunStatus: (status) => {
      const current = get();
      if (
        current.executionWindowId !== null &&
        current.executionWindowId !== current.activeWindowId
      ) {
        const target = current.windows.find(
          (window) => window.id === current.executionWindowId,
        );
        if (target) {
          const execution = reduceRunStatus(target.execution, status);
          set((state) => ({
            windows: state.windows.map((window) =>
              window.id === target.id ? { ...window, execution } : window,
            ),
          }));
          return;
        }
      }
      set((state) => {
        const execution = reduceRunStatus(state.execution, status);
        return execution === state.execution ? {} : { execution };
      });
    },

    setWsConnected: (connected) => set({ wsConnected: connected }),
  };
});
