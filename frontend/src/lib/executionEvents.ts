import type {
  BackendRunStatus,
  ExecutionError,
  ExecutionEvent,
  ExecutionStatus,
  NodeExecutionState,
  PreviewValue,
  RichOutput,
} from "../types/graph";
import { parseRichOutputs } from "../types/graph";

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

export function idleExecution(): GraphExecutionState {
  return {
    status: "idle",
    runId: null,
    order: [],
    nodeStates: {},
    previews: {},
    richOutputs: {},
    cached: {},
    error: null,
  };
}

export function pruneExecution(
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

export function reduceExecutionEvent(
  execution: GraphExecutionState,
  event: ExecutionEvent,
): GraphExecutionState {
  switch (event.type) {
    case "EXECUTION_STARTED": {
      const order = Array.isArray(event.data.order)
        ? (event.data.order as string[])
        : [];
      return {
        ...idleExecution(),
        status: "running",
        runId:
          typeof event.data.run_id === "string"
            ? event.data.run_id
            : execution.runId,
        order,
      };
    }
    case "NODE_STARTED":
      return {
        ...execution,
        nodeStates: {
          ...execution.nodeStates,
          [event.node_id]: { status: "running" },
        },
      };
    case "NODE_PROGRESS": {
      const rawProgress =
        typeof event.data.progress === "number" &&
        Number.isFinite(event.data.progress)
          ? event.data.progress
          : 0;
      const existing = execution.nodeStates[event.node_id];
      return {
        ...execution,
        nodeStates: {
          ...execution.nodeStates,
          [event.node_id]: {
            ...(existing ?? { status: "running" as const }),
            status: "running",
            progress: Math.min(1, Math.max(0, rawProgress)),
            progressMessage:
              typeof event.data.message === "string"
                ? event.data.message
                : undefined,
          },
        },
      };
    }
    case "INSTALLING_DEPS": {
      const installing = Array.isArray(event.data.dependencies)
        ? (event.data.dependencies as string[])
        : [];
      return {
        ...execution,
        nodeStates: {
          ...execution.nodeStates,
          [event.node_id]: { status: "running", installing },
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
      };
    }
    case "NODE_ERROR": {
      const richOutputs = parseRichOutputs(event.data.rich_outputs);
      return {
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
      };
    }
    case "EXECUTION_COMPLETED":
      return {
        ...execution,
        status: "completed",
        previews: (event.data.previews ?? {}) as Record<
          string,
          Record<string, PreviewValue>
        >,
        cached: (event.data.cached ?? {}) as Record<string, boolean>,
      };
    case "EXECUTION_FAILED":
      return {
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
        ...execution,
        status: "cancelled",
        error: null,
        nodeStates,
      };
    }
    default:
      return execution;
  }
}

export function reduceRunStatus(
  execution: GraphExecutionState,
  status: BackendRunStatus,
): GraphExecutionState {
  if (status.status === "running") {
    if (execution.status === "running") {
      return execution;
    }
    return {
      ...idleExecution(),
      status: "running",
      runId: status.run_id,
    };
  }
  if (execution.status !== "running" || execution.runId === null) {
    return execution;
  }
  const nodeStates: Record<string, NodeExecutionState> = {};
  for (const [nodeId, nodeState] of Object.entries(execution.nodeStates)) {
    nodeStates[nodeId] =
      nodeState.status === "running" ? { status: "idle" } : nodeState;
  }
  return {
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
  };
}
