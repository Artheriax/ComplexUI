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
  executeGraph,
  fetchTemplates,
  fetchWorkflowTemplates,
  parseCode,
  syntaxErrorFrom,
  untrustedNodesFrom,
} from "../lib/api";
import { sha256Hex } from "../lib/hash";
import { loadLocalWorkflow } from "../lib/persistence";
import type { SavedWorkflow } from "../lib/persistence";
import type {
  AppEdge,
  AppNode,
  BackendRunStatus,
  ComplexNodeData,
  ExecuteResponse,
  ExecutionError,
  ExecutionEvent,
  ExecutionStatus,
  NodeExecutionState,
  NodeSchema,
  PendingApproval,
  PreviewValue,
  RichOutput,
  TemplateInfo,
  WorkflowGraphPayload,
  WorkflowTemplateInfo,
} from "../types/graph";
import { defaultWidgetValues, parseRichOutputs } from "../types/graph";

export interface GraphExecutionState {
  status: ExecutionStatus;
  runId: string | null;
  order: string[];
  nodeStates: Record<string, NodeExecutionState>;
  previews: Record<string, Record<string, PreviewValue>>;
  richOutputs: Record<string, RichOutput[]>;
  cached: Record<string, boolean>;
  error: ExecutionError | null;
}

function pruneExecution(
  execution: GraphExecutionState,
  removed: Set<string>,
): GraphExecutionState {
  if (removed.size === 0) {
    return execution;
  }
  const nodeStates = { ...execution.nodeStates };
  const previews = { ...execution.previews };
  const richOutputs = { ...execution.richOutputs };
  const cached = { ...execution.cached };
  for (const id of removed) {
    delete nodeStates[id];
    delete previews[id];
    delete richOutputs[id];
    delete cached[id];
  }
  return { ...execution, nodeStates, previews, richOutputs, cached };
}

export interface GraphSnapshot {
  nodes: AppNode[];
  edges: AppEdge[];
}

interface HistoryMeta {
  coalesceKey: string | null;
  coalesceAt: number;
  dragging: boolean;
}

const HISTORY_LIMIT = 100;
const HISTORY_COALESCE_MS = 800;
const PASTE_OFFSET = 40;

interface GraphStore {
  templates: TemplateInfo[];
  templatesLoaded: boolean;
  templatesError: string | null;

  workflows: WorkflowTemplateInfo[];
  workflowsLoaded: boolean;
  workflowsError: string | null;

  nodes: AppNode[];
  edges: AppEdge[];

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

  loadTemplates: () => Promise<void>;
  loadWorkflowTemplates: () => Promise<void>;
  loadWorkflowTemplate: (workflow: WorkflowTemplateInfo) => boolean;
  addNodeFromTemplate: (template: TemplateInfo, position: XYPosition) => string;
  addScratchNode: (position: XYPosition) => string;
  duplicateNode: (nodeId: string) => Promise<string | null>;
  deleteNode: (nodeId: string) => void;
  clearCanvas: () => void;
  checkpoint: () => void;
  undo: () => void;
  redo: () => void;
  copySelection: () => boolean;
  cutSelection: () => boolean;
  pasteClipboard: () => Promise<string[]>;
  duplicateSelection: () => Promise<string[]>;
  onNodesChange: (changes: NodeChange<AppNode>[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
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
  cancelRun: () => Promise<void>;
  applyExecutionEvent: (event: ExecutionEvent) => void;
  syncRunStatus: (status: BackendRunStatus) => void;
  setWsConnected: (connected: boolean) => void;
}

export const SCRATCH_NODE_CODE = `# Scratch node — edit freely.
# - bare annotations create input sockets  (e.g. \\\`train: DATASET\\\`)
# - annotated defaults create parameter widgets  (e.g. \\\`k: int = 5\\\`)
# - named returns create output sockets  (e.g. \\\`return result: ANY\\\`)

factor: float = 2.0
result = factor * 21
return result: ANY
`;

function emptySchema(nodeId: string): NodeSchema {
  return {
    node_id: nodeId,
    inputs: [],
    widgets: [],
    outputs: [],
    dependencies: [],
    syntax_errors: [],
  };
}

const idleExecution = (): GraphExecutionState => ({
  status: "idle",
  runId: null,
  order: [],
  nodeStates: {},
  previews: {},
  richOutputs: {},
  cached: {},
  error: null,
});

function widgetValueAllowed(dataType: string, value: unknown): boolean {
  switch (dataType) {
    case "int":
      return typeof value === "number" && Number.isInteger(value);
    case "float":
      return typeof value === "number" && Number.isFinite(value);
    case "str":
      return typeof value === "string";
    case "bool":
      return typeof value === "boolean";
    default:
      return true;
  }
}

function mergeWidgetValues(
  schema: NodeSchema,
  previous: Record<string, unknown>,
): Record<string, unknown> {
  const next = defaultWidgetValues(schema);
  for (const widget of schema.widgets) {
    if (
      previous[widget.name] !== undefined &&
      widgetValueAllowed(widget.data_type, previous[widget.name])
    ) {
      next[widget.name] = previous[widget.name];
    }
  }
  return next;
}

function reconcileEdges(
  edges: AppEdge[],
  nodeId: string,
  schema: NodeSchema,
): AppEdge[] {
  const inputs = new Set(schema.inputs.map((socket) => socket.name));
  const outputs = new Set(schema.outputs.map((socket) => socket.name));
  return edges.filter((edge) => {
    if (edge.source === nodeId && !outputs.has(edge.sourceHandle ?? "")) {
      return false;
    }
    if (edge.target === nodeId && !inputs.has(edge.targetHandle ?? "")) {
      return false;
    }
    return true;
  });
}

export function serialiseGraph(
  nodes: AppNode[],
  edges: AppEdge[],
): WorkflowGraphPayload {
  return {
    nodes: nodes.map((node) => ({
      id: node.id,
      code: node.data.code,
      widget_values: node.data.widgetValues,
    })),
    edges: edges.map((edge) => ({
      source_node_id: edge.source,
      source_handle: edge.sourceHandle ?? "",
      target_node_id: edge.target,
      target_handle: edge.targetHandle ?? "",
    })),
  };
}

export const useGraphStore = create<GraphStore>((set, get) => {
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
    const { nodes, edges } = get();
    return { nodes, edges };
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

    nodes: [],
    edges: [],

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
      const templates = new Map(
        get().templates.map((template) => [template.name, template]),
      );
      const missing = workflow.nodes.filter(
        (node) => !templates.has(node.template),
      );
      if (missing.length > 0) {
        set({
          workflowsError: `Missing templates: ${missing
            .map((node) => node.template)
            .join(", ")}`,
        });
        return false;
      }
      const nodes: AppNode[] = workflow.nodes.map((spec) => {
        const template = templates.get(spec.template) as TemplateInfo;
        const node: AppNode = {
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
          },
        };
        return node;
      });
      const edges: AppEdge[] = workflow.edges.map((edge) => ({
        id: `${edge.source_node_id}:${edge.source_handle}->${edge.target_node_id}:${edge.target_handle}`,
        source: edge.source_node_id,
        target: edge.target_node_id,
        sourceHandle: edge.source_handle,
        targetHandle: edge.target_handle,
      }));
      pushHistory();
      set({
        nodes,
        edges,
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        workflowsError: null,
        execution: idleExecution(),
      });
      return true;
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
        },
      };
      pushHistory();
      set((state) => ({ nodes: [...state.nodes, node] }));
      return id;
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
      if (!node) {
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
      if (!get().nodes.some((node) => node.id === nodeId)) {
        return;
      }
      pushHistory();
      set((state) => ({
        nodes: state.nodes.filter((node) => node.id !== nodeId),
        edges: state.edges.filter(
          (edge) => edge.source !== nodeId && edge.target !== nodeId,
        ),
        editingNodeId:
          state.editingNodeId === nodeId ? null : state.editingNodeId,
        execution: pruneExecution(state.execution, new Set([nodeId])),
      }));
    },

    clearCanvas: () => {
      if (get().nodes.length === 0 && get().edges.length === 0) {
        return;
      }
      pushHistory();
      set({
        nodes: [],
        edges: [],
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        execution: idleExecution(),
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
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        historyMeta: { ...state.historyMeta, coalesceKey: null, dragging: false },
      }));
    },

    copySelection: () => {
      const { nodes, edges } = get();
      const selected = nodes.filter((node) => node.selected);
      if (selected.length === 0) {
        return false;
      }
      const ids = new Set(selected.map((node) => node.id));
      set({
        clipboard: {
          nodes: selected.map((node) => ({
            ...node,
            data: { ...node.data },
          })),
          edges: edges.filter(
            (edge) => ids.has(edge.source) && ids.has(edge.target),
          ),
        },
      });
      return true;
    },

    cutSelection: () => {
      const { nodes } = get();
      const selected = nodes.filter((node) => node.selected);
      if (selected.length === 0) {
        return false;
      }
      get().copySelection();
      pushHistory();
      const removed = new Set(selected.map((node) => node.id));
      set((state) => ({
        nodes: state.nodes.filter((node) => !removed.has(node.id)),
        edges: state.edges.filter(
          (edge) => !removed.has(edge.source) && !removed.has(edge.target),
        ),
        execution: pruneExecution(state.execution, removed),
      }));
      return true;
    },

    pasteClipboard: async () => {
      const clip = get().clipboard;
      if (!clip || clip.nodes.length === 0) {
        return [];
      }
      pushHistory();
      const templates = get().templates;
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
        newNodes.push({
          ...node,
          id: idMap.get(node.id) as string,
          selected: true,
          position: {
            x: node.position.x + PASTE_OFFSET,
            y: node.position.y + PASTE_OFFSET,
          },
          data: { ...node.data, isTrusted: isOfficial },
        });
      }
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
      const removed = changes
        .filter((change) => change.type === "remove")
        .map((change) => change.id);
      const dragStart = changes.some(
        (change) => change.type === "position" && change.dragging === true,
      );
      const dragEnd = changes.some(
        (change) => change.type === "position" && change.dragging === false,
      );
      if (dragStart && !get().historyMeta.dragging) {
        pushHistory();
        set((state) => ({
          historyMeta: { ...state.historyMeta, dragging: true },
        }));
      }
      set((state) => {
        const next: Partial<GraphStore> = {
          nodes: applyNodeChanges<AppNode>(changes, state.nodes),
        };
        if (removed.length > 0) {
          next.edges = state.edges.filter(
            (edge) =>
              !removed.includes(edge.source) && !removed.includes(edge.target),
          );
          next.execution = pruneExecution(
            state.execution,
            new Set(removed),
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

    onConnect: (connection) => {
      const { source, target, sourceHandle, targetHandle } = connection;
      if (!source || !target || !sourceHandle || !targetHandle) {
        return;
      }
      if (source === target) {
        return;
      }
      const state = get();
      const sourceNode = state.nodes.find((node) => node.id === source);
      const targetNode = state.nodes.find((node) => node.id === target);
      if (!sourceNode || !targetNode) {
        return;
      }
      const sourceSocket = sourceNode.data.schema.outputs.find(
        (socket) => socket.name === sourceHandle,
      );
      const targetSocket = targetNode.data.schema.inputs.find(
        (socket) => socket.name === targetHandle,
      );
      if (!sourceSocket || !targetSocket) {
        return;
      }
      const compatible =
        sourceSocket.data_type === targetSocket.data_type ||
        sourceSocket.data_type === "ANY" ||
        targetSocket.data_type === "ANY";
      const inputTaken = state.edges.some(
        (edge) => edge.target === target && edge.targetHandle === targetHandle,
      );
      if (!compatible || inputTaken) {
        return;
      }
      const edge: AppEdge = {
        id: `${source}:${sourceHandle}->${target}:${targetHandle}`,
        source,
        target,
        sourceHandle,
        targetHandle,
      };
      pushHistory();
      set((current) => ({ edges: addEdge(edge, current.edges) }));
    },

    openEditor: (nodeId) => set({ editingNodeId: nodeId }),

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

      const built = await Promise.all(
        saved.nodes.map(async (savedNode): Promise<AppNode> => {
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
            data: {
              label: savedNode.label,
              templateName: template?.name ?? savedNode.templateName,
              templateCode: template?.code ?? null,
              code: savedNode.code,
              schema,
              widgetValues: mergeWidgetValues(schema, savedNode.widgetValues),
              isTrusted: isOfficial,
            },
          };
        }),
      );

      const nodeIds = new Set(built.map((node) => node.id));
      let edges: AppEdge[] = saved.edges
        .filter(
          (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target),
        )
        .map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle,
        }));
      for (const node of built) {
        edges = reconcileEdges(edges, node.id, node.data.schema);
      }

      set({
        nodes: built,
        edges,
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        execution: idleExecution(),
      });
    },

    restoreFromStorage: async () => {
      if (get().nodes.length > 0) {
        return;
      }
      const saved = loadLocalWorkflow();
      if (saved && saved.nodes.length > 0) {
        await get().restoreWorkflow(saved);
        set({ past: [], future: [] });
      }
    },

    runGraph: async (targets?: string[]) => {
      const { nodes, edges } = get();
      const candidates = targets
        ? nodes.filter((node) => targets.includes(node.id))
        : nodes;
      if (candidates.length === 0) {
        set((state) => ({
          execution: {
            ...state.execution,
            status: "idle",
            error: {
              message: "Canvas is empty — drag a template node in first.",
              errorType: "LocalValidation",
              nodeId: null,
              line: null,
              traceback: "",
            },
          },
        }));
        return;
      }
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

      set({ execution: { ...idleExecution(), status: "running" } });
      try {
        const response: ExecuteResponse = await executeGraph(
          serialiseGraph(nodes, edges),
          targets,
        );
        set((state) => ({
          execution: { ...state.execution, status: "running", runId: response.run_id },
        }));
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
      set((state) => {
        const execution = state.execution;
        switch (event.type) {
          case "EXECUTION_STARTED": {
            const order = Array.isArray(event.data.order)
              ? (event.data.order as string[])
              : [];
            return {
              execution: {
                ...idleExecution(),
                status: "running",
                runId:
                  typeof event.data.run_id === "string"
                    ? event.data.run_id
                    : execution.runId,
                order,
              },
            };
          }
          case "NODE_STARTED":
            return {
              execution: {
                ...execution,
                nodeStates: {
                  ...execution.nodeStates,
                  [event.node_id]: { status: "running" },
                },
              },
            };
          case "INSTALLING_DEPS": {
            const installing = Array.isArray(event.data.dependencies)
              ? (event.data.dependencies as string[])
              : [];
            return {
              execution: {
                ...execution,
                nodeStates: {
                  ...execution.nodeStates,
                  [event.node_id]: { status: "running", installing },
                },
              },
            };
          }
          case "NODE_COMPLETED": {
            const outputs = (event.data.outputs ?? {}) as Record<
              string,
              PreviewValue
            >;
            const richOutputs = parseRichOutputs(event.data.rich_outputs);
            return {
              execution: {
                ...execution,
                previews: {
                  ...execution.previews,
                  [event.node_id]: outputs,
                },
                richOutputs: {
                  ...execution.richOutputs,
                  [event.node_id]: richOutputs,
                },
                cached: {
                  ...execution.cached,
                  [event.node_id]: event.data.cached === true,
                },
                nodeStates: {
                  ...execution.nodeStates,
                  [event.node_id]: {
                    status: "completed",
                    cached: event.data.cached === true,
                    outputs,
                    richOutputs,
                  },
                },
              },
            };
          }
          case "NODE_ERROR": {
            const richOutputs = parseRichOutputs(event.data.rich_outputs);
            return {
              execution: {
                ...execution,
                richOutputs: {
                  ...execution.richOutputs,
                  [event.node_id]: richOutputs,
                },
                nodeStates: {
                  ...execution.nodeStates,
                  [event.node_id]: {
                    status: "error",
                    richOutputs,
                    error: {
                      line:
                        typeof event.data.line === "number"
                          ? event.data.line
                          : null,
                      message: String(event.data.message ?? "Node failed"),
                      errorType: String(event.data.error_type ?? "Error"),
                      traceback: String(event.data.traceback ?? ""),
                    },
                  },
                },
              },
            };
          }
          case "EXECUTION_COMPLETED":
            return {
              execution: {
                ...execution,
                status: "completed",
                previews: (event.data.previews ?? {}) as Record<
                  string,
                  Record<string, PreviewValue>
                >,
                cached: (event.data.cached ?? {}) as Record<string, boolean>,
              },
            };
          case "EXECUTION_FAILED":
            return {
              execution: {
                ...execution,
                status: "failed",
                error: {
                  message: String(event.data.message ?? "Execution failed"),
                  errorType: String(event.data.error_type ?? "Error"),
                  nodeId:
                    typeof event.data.node_id === "string"
                      ? event.data.node_id
                      : null,
                  line:
                    typeof event.data.line === "number"
                      ? event.data.line
                      : null,
                  traceback: String(event.data.traceback ?? ""),
                },
              },
            };
          case "EXECUTION_CANCELLED": {
            const nodeStates: Record<string, NodeExecutionState> = {};
            for (const [nodeId, nodeState] of Object.entries(
              execution.nodeStates,
            )) {
              nodeStates[nodeId] =
                nodeState.status === "running" ? { status: "idle" } : nodeState;
            }
            return {
              execution: {
                ...execution,
                status: "cancelled",
                error: null,
                nodeStates,
              },
            };
          }
        }
      });
    },

    syncRunStatus: (status) => {
      set((state) => {
        const execution = state.execution;
        if (status.status === "running") {
          if (execution.status === "running") {
            return {};
          }
          return {
            execution: {
              ...idleExecution(),
              status: "running",
              runId: status.run_id,
            },
          };
        }
        if (execution.status !== "running" || execution.runId === null) {
          return {};
        }
        const nodeStates: Record<string, NodeExecutionState> = {};
        for (const [nodeId, nodeState] of Object.entries(
          execution.nodeStates,
        )) {
          nodeStates[nodeId] =
            nodeState.status === "running" ? { status: "idle" } : nodeState;
        }
        return {
          execution: {
            ...execution,
            status: "failed",
            nodeStates,
            error: {
              message:
                "Lost contact with the backend before the run finished. Re-run the workflow to refresh outputs.",
              errorType: "Disconnected",
              nodeId: null,
              line: null,
              traceback: "",
            },
          },
        };
      });
    },

    setWsConnected: (connected) => set({ wsConnected: connected }),
  };
});
