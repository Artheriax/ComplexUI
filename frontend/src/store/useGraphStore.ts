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
  ComplexNodeData,
  ExecuteResponse,
  ExecutionError,
  ExecutionEvent,
  ExecutionStatus,
  NodeExecutionState,
  NodeSchema,
  PendingApproval,
  PreviewValue,
  TemplateInfo,
  WorkflowGraphPayload,
} from "../types/graph";
import { defaultWidgetValues } from "../types/graph";

export interface GraphExecutionState {
  status: ExecutionStatus;
  runId: string | null;
  order: string[];
  nodeStates: Record<string, NodeExecutionState>;
  previews: Record<string, Record<string, PreviewValue>>;
  cached: Record<string, boolean>;
  error: ExecutionError | null;
}

interface GraphStore {
  templates: TemplateInfo[];
  templatesLoaded: boolean;
  templatesError: string | null;

  nodes: AppNode[];
  edges: AppEdge[];

  editingNodeId: string | null;
  pendingApproval: PendingApproval | null;
  approvalBusy: boolean;
  approvalError: string | null;
  approvedHashes: Set<string>;

  wsConnected: boolean;
  execution: GraphExecutionState;

  loadTemplates: () => Promise<void>;
  addNodeFromTemplate: (template: TemplateInfo, position: XYPosition) => string;
  addScratchNode: (position: XYPosition) => string;
  duplicateNode: (nodeId: string) => Promise<string | null>;
  deleteNode: (nodeId: string) => void;
  clearCanvas: () => void;
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
  cached: {},
  error: null,
});

function mergeWidgetValues(
  schema: NodeSchema,
  previous: Record<string, unknown>,
): Record<string, unknown> {
  const next = defaultWidgetValues(schema);
  for (const widget of schema.widgets) {
    if (previous[widget.name] !== undefined) {
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

  return {
    templates: [],
    templatesLoaded: false,
    templatesError: null,

    nodes: [],
    edges: [],

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
      set((state) => ({
        nodes: state.nodes.filter((node) => node.id !== nodeId),
        edges: state.edges.filter(
          (edge) => edge.source !== nodeId && edge.target !== nodeId,
        ),
        editingNodeId:
          state.editingNodeId === nodeId ? null : state.editingNodeId,
      }));
    },

    clearCanvas: () => {
      set({
        nodes: [],
        edges: [],
        editingNodeId: null,
        pendingApproval: null,
        approvalError: null,
        execution: idleExecution(),
      });
    },

    onNodesChange: (changes) => {
      const removed = changes
        .filter((change) => change.type === "remove")
        .map((change) => change.id);
      set((state) => {
        const next: Partial<GraphStore> = {
          nodes: applyNodeChanges<AppNode>(changes, state.nodes),
        };
        if (removed.length > 0) {
          next.edges = state.edges.filter(
            (edge) =>
              !removed.includes(edge.source) && !removed.includes(edge.target),
          );
          const nodeStates = { ...state.execution.nodeStates };
          const previews = { ...state.execution.previews };
          const cached = { ...state.execution.cached };
          for (const id of removed) {
            delete nodeStates[id];
            delete previews[id];
            delete cached[id];
          }
          next.execution = { ...state.execution, nodeStates, previews, cached };
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
      set((state) => {
        const sourceNode = state.nodes.find((node) => node.id === source);
        const targetNode = state.nodes.find((node) => node.id === target);
        if (!sourceNode || !targetNode) {
          return {};
        }
        const sourceSocket = sourceNode.data.schema.outputs.find(
          (socket) => socket.name === sourceHandle,
        );
        const targetSocket = targetNode.data.schema.inputs.find(
          (socket) => socket.name === targetHandle,
        );
        if (!sourceSocket || !targetSocket) {
          return {};
        }
        const compatible =
          sourceSocket.data_type === targetSocket.data_type ||
          sourceSocket.data_type === "ANY" ||
          targetSocket.data_type === "ANY";
        const inputTaken = state.edges.some(
          (edge) => edge.target === target && edge.targetHandle === targetHandle,
        );
        if (!compatible || inputTaken) {
          return {};
        }
        const edge: AppEdge = {
          id: `${source}:${sourceHandle}->${target}:${targetHandle}`,
          source,
          target,
          sourceHandle,
          targetHandle,
        };
        return { edges: addEdge(edge, state.edges) };
      });
    },

    openEditor: (nodeId) => set({ editingNodeId: nodeId }),

    closeEditor: () => set({ editingNodeId: null }),

    updateWidgetValue: (nodeId, name, value) => {
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

    saveNodeCode: async (nodeId, code, schema) => {
      const hash = await sha256Hex(code);
      const state = get();
      const isOfficial = state.templates.some(
        (template) => template.code_hash === hash,
      );
      const isSessionApproved = state.approvedHashes.has(`${nodeId}:${hash}`);
      if (isOfficial || isSessionApproved) {
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
          case "NODE_COMPLETED":
            return {
              execution: {
                ...execution,
                nodeStates: {
                  ...execution.nodeStates,
                  [event.node_id]: {
                    status: "completed",
                    cached: event.data.cached === true,
                    outputs: (event.data.outputs ?? {}) as Record<
                      string,
                      PreviewValue
                    >,
                  },
                },
              },
            };
          case "NODE_ERROR":
            return {
              execution: {
                ...execution,
                nodeStates: {
                  ...execution.nodeStates,
                  [event.node_id]: {
                    status: "error",
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
          case "EXECUTION_CANCELLED":
            return {
              execution: {
                ...execution,
                status: "cancelled",
                error: null,
              },
            };
        }
      });
    },

    setWsConnected: (connected) => set({ wsConnected: connected }),
  };
});
