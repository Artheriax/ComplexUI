import { beforeEach, describe, expect, it, vi } from "vitest";

import { sha256Hex } from "../lib/hash";
import {
  clearLocalProjectIdentity,
  clearLocalWorkflow,
  saveLocalWorkflow,
} from "../lib/persistence";
import type { SavedWorkflow } from "../lib/persistence";
import type {
  AppEdge,
  AppNode,
  InputSocket,
  NodeSchema,
  OutputSocket,
  TemplateInfo,
  WorkflowTemplateInfo,
} from "../types/graph";
import {
  NEW_SUBGRAPH_INPUT_HANDLE,
  NEW_SUBGRAPH_OUTPUT_HANDLE,
  SUBGRAPH_INPUT_NODE_ID,
  SUBGRAPH_OUTPUT_NODE_ID,
} from "../types/graph";
import {
  SCRATCH_NODE_CODE,
  serialiseGraph,
  useGraphStore,
} from "./useGraphStore";

function makeSchema(overrides: Partial<NodeSchema> = {}): NodeSchema {
  return {
    node_id: "",
    inputs: [],
    widgets: [],
    outputs: [],
    dependencies: [],
    syntax_errors: [],
    ...overrides,
  };
}

async function makeTemplate(
  overrides: Partial<TemplateInfo> = {},
): Promise<TemplateInfo> {
  const code = "# demo node\nk: int = 5\nreturn k: ANY\n";
  const nodeSchema = makeSchema({
    inputs: [{ name: "df", data_type: "DATAFRAME" }],
    widgets: [{ name: "k", data_type: "int", default: 5 }],
    outputs: [{ name: "k", data_type: "ANY" }],
  });
  return {
    name: "demo",
    title: "Demo Node",
    category: "Preprocessing",
    description: "Demo template for tests",
    hidden: false,
    code,
    code_hash: await sha256Hex(code),
    node_schema: nodeSchema,
    ...overrides,
  };
}

function resetStore(): void {
  useGraphStore.setState({
    templates: [],
    templatesLoaded: false,
    templatesError: null,
    workflows: [],
    workflowsLoaded: false,
    workflowsError: null,
    projects: [],
    projectsLoaded: false,
    projectsError: null,
    projectName: null,
    projectSlug: null,
    saveDialogOpen: false,
    saveBusy: false,
    saveError: null,
    nodes: [],
    edges: [],
    subgraphs: {},
    scopeStack: [],
    rootGraph: null,
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
    execution: {
      status: "idle",
      runId: null,
      order: [],
      nodeStates: {},
      previews: {},
      richOutputs: {},
      cached: {},
      error: null,
    },
  });
}

beforeEach(() => {
  resetStore();
  clearLocalWorkflow();
  clearLocalProjectIdentity();
  vi.unstubAllGlobals();
});

describe("addNodeFromTemplate", () => {
  it("creates a trusted node with widget defaults", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 10, y: 20 });

    const node = useGraphStore.getState().nodes.find((n) => n.id === id);
    expect(node).toBeDefined();
    expect(node?.type).toBe("complex");
    expect(node?.position).toEqual({ x: 10, y: 20 });
    expect(node?.data.code).toBe(template.code);
    expect(node?.data.templateName).toBe(template.name);
    expect(node?.data.isTrusted).toBe(true);
    expect(node?.data.widgetValues).toEqual({ k: 5 });
  });
});

describe("updateWidgetValue", () => {
  it("updates a single widget value immutably", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });

    useGraphStore.getState().updateWidgetValue(id, "k", 9);

    const node = useGraphStore.getState().nodes.find((n) => n.id === id);
    expect(node?.data.widgetValues).toEqual({ k: 9 });
  });
});

describe("onConnect", () => {
  it("creates edges for compatible sockets and rejects invalid ones", async () => {
    const producer = await makeTemplate();
    const consumer = await makeTemplate({
      name: "sink",
      title: "Sink",
      node_schema: makeSchema({ inputs: [{ name: "k", data_type: "ANY" }] }),
    });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(producer, { x: 0, y: 0 });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 200, y: 0 });

    useGraphStore.getState().onConnect({
      source: sourceId,
      target: targetId,
      sourceHandle: "k",
      targetHandle: "k",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);
    expect(useGraphStore.getState().edges[0]?.id).toBe(
      `${sourceId}:k->${targetId}:k`,
    );

    const otherSource = useGraphStore
      .getState()
      .addNodeFromTemplate(producer, { x: 0, y: 150 });
    useGraphStore.getState().onConnect({
      source: otherSource,
      target: targetId,
      sourceHandle: "k",
      targetHandle: "k",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);

    useGraphStore.getState().onConnect({
      source: sourceId,
      target: targetId,
      sourceHandle: "missing",
      targetHandle: "k",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);

    useGraphStore.getState().onConnect({
      source: sourceId,
      target: sourceId,
      sourceHandle: "k",
      targetHandle: "k",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);
  });
});

describe("saveNodeCode", () => {
  it("commits official code immediately and reconciles orphaned edges", async () => {
    const template = await makeTemplate();
    const consumer = await makeTemplate({
      name: "sink",
      node_schema: makeSchema({ inputs: [{ name: "k", data_type: "ANY" }] }),
    });
    useGraphStore.setState({ templates: [template] });

    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 200, y: 0 });
    useGraphStore.getState().onConnect({
      source: sourceId,
      target: targetId,
      sourceHandle: "k",
      targetHandle: "k",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);

    const renamedSchema = makeSchema({
      outputs: [{ name: "result", data_type: "ANY" }],
    });
    await useGraphStore
      .getState()
      .saveNodeCode(sourceId, template.code, renamedSchema);

    const state = useGraphStore.getState();
    expect(state.pendingApproval).toBeNull();
    expect(state.editingNodeId).toBeNull();
    expect(state.edges).toHaveLength(0);
    expect(
      state.nodes.find((n) => n.id === sourceId)?.data.schema.outputs[0]?.name,
    ).toBe("result");
  });

  it("requires approval for modified code and commits after approve", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const modifiedCode = template.code.replace("k: int = 5", "k: int = 9");

    await useGraphStore
      .getState()
      .saveNodeCode(id, modifiedCode, template.node_schema);

    const pending = useGraphStore.getState().pendingApproval;
    expect(pending?.nodeId).toBe(id);
    expect(pending?.codeHash).toBe(await sha256Hex(modifiedCode));
    expect(
      useGraphStore.getState().nodes.find((n) => n.id === id)?.data.code,
    ).toBe(template.code);

    useGraphStore.getState().rejectPending();
    expect(useGraphStore.getState().pendingApproval).toBeNull();
    expect(
      useGraphStore.getState().nodes.find((n) => n.id === id)?.data.code,
    ).toBe(template.code);

    await useGraphStore
      .getState()
      .saveNodeCode(id, modifiedCode, template.node_schema);
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({ status: "approved", node_id: id, code_hash: "x" }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);
    await useGraphStore.getState().approvePending();
    vi.unstubAllGlobals();

    const state = useGraphStore.getState();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(state.pendingApproval).toBeNull();
    expect(state.approvalError).toBeNull();
    const node = state.nodes.find((n) => n.id === id);
    expect(node?.data.code).toBe(modifiedCode);
    expect(node?.data.isTrusted).toBe(true);
  });
});

describe("applyExecutionEvent", () => {
  it("tracks node states and completion previews", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const store = useGraphStore.getState();

    store.applyExecutionEvent({
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "r1", order: [id] },
    });
    expect(useGraphStore.getState().execution.status).toBe("running");
    expect(useGraphStore.getState().execution.runId).toBe("r1");

    store.applyExecutionEvent({ type: "NODE_STARTED", node_id: id, data: {} });
    expect(
      useGraphStore.getState().execution.nodeStates[id]?.status,
    ).toBe("running");

    store.applyExecutionEvent({
      type: "NODE_COMPLETED",
      node_id: id,
      data: { cached: false, outputs: { k: 5 } },
    });
    expect(
      useGraphStore.getState().execution.nodeStates[id]?.status,
    ).toBe("completed");
    expect(useGraphStore.getState().execution.nodeStates[id]?.cached).toBe(
      false,
    );

    store.applyExecutionEvent({
      type: "NODE_ERROR",
      node_id: id,
      data: {
        line: 12,
        message: "boom",
        error_type: "ValueError",
        traceback: "traceback text",
      },
    });
    const errored = useGraphStore.getState().execution.nodeStates[id];
    expect(errored?.status).toBe("error");
    expect(errored?.error?.line).toBe(12);
    expect(errored?.error?.errorType).toBe("ValueError");

    store.applyExecutionEvent({
      type: "EXECUTION_COMPLETED",
      node_id: "",
      data: { run_id: "r1", previews: { [id]: { k: 5 } }, cached: { [id]: false } },
    });
    expect(useGraphStore.getState().execution.status).toBe("completed");
    expect(useGraphStore.getState().execution.previews[id]).toEqual({ k: 5 });

    store.applyExecutionEvent({
      type: "EXECUTION_FAILED",
      node_id: "",
      data: { message: "boom", error_type: "ValueError" },
    });
    expect(useGraphStore.getState().execution.error?.errorType).toBe(
      "ValueError",
    );

    store.applyExecutionEvent({
      type: "EXECUTION_CANCELLED",
      node_id: "",
      data: { run_id: "r1" },
    });
    expect(useGraphStore.getState().execution.status).toBe("cancelled");
    expect(useGraphStore.getState().execution.error).toBeNull();
  });

  it("publishes previews as soon as each node completes", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const store = useGraphStore.getState();

    store.applyExecutionEvent({
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "r2", order: [id] },
    });
    store.applyExecutionEvent({
      type: "NODE_COMPLETED",
      node_id: id,
      data: { cached: true, outputs: { k: 9 } },
    });

    const execution = useGraphStore.getState().execution;
    expect(execution.status).toBe("running");
    expect(execution.previews[id]).toEqual({ k: 9 });
    expect(execution.cached[id]).toBe(true);
    expect(execution.nodeStates[id]?.status).toBe("completed");
  });

  it("tracks node progress and clears it on completion", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const store = useGraphStore.getState();

    store.applyExecutionEvent({
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "progress-run", order: [id] },
    });
    store.applyExecutionEvent({
      type: "NODE_PROGRESS",
      node_id: id,
      data: { progress: 0.4, message: "step 4/10" },
    });

    let state = useGraphStore.getState().execution.nodeStates[id];
    expect(state?.status).toBe("running");
    expect(state?.progress).toBeCloseTo(0.4);
    expect(state?.progressMessage).toBe("step 4/10");

    store.applyExecutionEvent({
      type: "NODE_PROGRESS",
      node_id: id,
      data: { progress: 3, message: "clamped" },
    });
    state = useGraphStore.getState().execution.nodeStates[id];
    expect(state?.progress).toBe(1);

    store.applyExecutionEvent({
      type: "NODE_COMPLETED",
      node_id: id,
      data: { cached: false, outputs: { k: 1 } },
    });
    state = useGraphStore.getState().execution.nodeStates[id];
    expect(state?.status).toBe("completed");
    expect(state?.progress).toBeUndefined();
  });

  it("stores rich outputs from completion and error events", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const store = useGraphStore.getState();
    const textOutput = {
      kind: "text" as const,
      source: "print",
      text: "hello\n",
      truncated: false,
    };
    const tableOutput = {
      kind: "table" as const,
      source: "display",
      columns: ["a"],
      rows: [[1]],
      total_rows: 1,
      total_cols: 1,
      truncated: false,
    };

    store.applyExecutionEvent({
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "r3", order: [id] },
    });
    store.applyExecutionEvent({
      type: "NODE_COMPLETED",
      node_id: id,
      data: { cached: false, outputs: { k: 1 }, rich_outputs: [textOutput, tableOutput] },
    });

    let execution = useGraphStore.getState().execution;
    expect(execution.richOutputs[id]).toEqual([textOutput, tableOutput]);
    expect(execution.nodeStates[id]?.richOutputs).toEqual([textOutput, tableOutput]);

    store.applyExecutionEvent({
      type: "NODE_ERROR",
      node_id: id,
      data: {
        line: 3,
        message: "boom",
        error_type: "ValueError",
        traceback: "",
        rich_outputs: [textOutput],
      },
    });

    execution = useGraphStore.getState().execution;
    expect(execution.richOutputs[id]).toEqual([textOutput]);
    expect(execution.nodeStates[id]?.richOutputs).toEqual([textOutput]);
  });

  it("resets running nodes when the run is cancelled", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const store = useGraphStore.getState();

    store.applyExecutionEvent({
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "r1", order: [id] },
    });
    store.applyExecutionEvent({
      type: "INSTALLING_DEPS",
      node_id: id,
      data: { dependencies: ["torch>=2.2"] },
    });
    store.applyExecutionEvent({
      type: "EXECUTION_CANCELLED",
      node_id: "",
      data: { run_id: "r1" },
    });

    const state = useGraphStore.getState();
    expect(state.execution.status).toBe("cancelled");
    expect(state.execution.nodeStates[id]?.status).toBe("idle");
    expect(state.execution.nodeStates[id]?.installing).toBeUndefined();
  });
});

describe("syncRunStatus", () => {
  it("adopts a run that started while the socket was down", () => {
    useGraphStore
      .getState()
      .syncRunStatus({ status: "running", run_id: "srv-1" });

    const execution = useGraphStore.getState().execution;
    expect(execution.status).toBe("running");
    expect(execution.runId).toBe("srv-1");
  });

  it("fails a tracked run when the backend reports idle", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const store = useGraphStore.getState();

    store.applyExecutionEvent({
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "r1", order: [id] },
    });
    store.applyExecutionEvent({ type: "NODE_STARTED", node_id: id, data: {} });

    useGraphStore
      .getState()
      .syncRunStatus({ status: "idle", run_id: null });

    const execution = useGraphStore.getState().execution;
    expect(execution.status).toBe("failed");
    expect(execution.error?.errorType).toBe("Disconnected");
    expect(execution.nodeStates[id]?.status).toBe("idle");
  });

  it("ignores idle reports while a run request is still in flight", () => {
    useGraphStore.setState((state) => ({
      execution: { ...state.execution, status: "running", runId: null },
    }));

    useGraphStore
      .getState()
      .syncRunStatus({ status: "idle", run_id: null });

    expect(useGraphStore.getState().execution.status).toBe("running");
  });
});

describe("runGraph", () => {
  it("reports an empty canvas locally", async () => {
    await useGraphStore.getState().runGraph();
    expect(useGraphStore.getState().execution.error?.errorType).toBe(
      "LocalValidation",
    );
  });

  it("marks nodes as untrusted when the backend answers 403", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });

    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            detail: {
              message: "untrusted",
              untrusted_nodes: [{ node_id: id, code_hash: "abc" }],
            },
          }),
          { status: 403, headers: { "Content-Type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);
    await useGraphStore.getState().runGraph();
    vi.unstubAllGlobals();

    const state = useGraphStore.getState();
    expect(state.execution.error?.errorType).toBe("UntrustedCode");
    expect(state.nodes.find((n) => n.id === id)?.data.isTrusted).toBe(false);
  });

  it("starts execution via the API and stores the run id", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    useGraphStore.getState().addNodeFromTemplate(template, { x: 0, y: 0 });

    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ status: "started", run_id: "run-42" }), {
          status: 202,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await useGraphStore.getState().runGraph();
    vi.unstubAllGlobals();

    expect(fetchMock).toHaveBeenCalledOnce();
    const execution = useGraphStore.getState().execution;
    expect(execution.status).toBe("running");
    expect(execution.runId).toBe("run-42");
  });

  it("does not overwrite a fast completion before the execute response", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    useGraphStore.getState().addNodeFromTemplate(template, { x: 0, y: 0 });

    let resolveFetch: ((response: Response) => void) | undefined;
    const fetchMock = vi.fn<typeof fetch>(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const run = useGraphStore.getState().runGraph();
    useGraphStore.getState().applyExecutionEvent({
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "run-fast", order: [] },
    });
    useGraphStore.getState().applyExecutionEvent({
      type: "EXECUTION_COMPLETED",
      node_id: "",
      data: { previews: {}, cached: {} },
    });

    resolveFetch?.(
      new Response(JSON.stringify({ status: "started", run_id: "run-fast" }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await run;
    vi.unstubAllGlobals();

    const execution = useGraphStore.getState().execution;
    expect(execution.status).toBe("completed");
    expect(execution.runId).toBe("run-fast");
  });

  it("sends targets in the execute payload for run-to-node", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });

    const fetchMock = vi.fn<typeof fetch>(
      async () =>
        new Response(JSON.stringify({ status: "started", run_id: "run-7" }), {
          status: 202,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await useGraphStore.getState().runGraph([sourceId]);
    vi.unstubAllGlobals();

    const init = fetchMock.mock.calls[0]?.[1];
    const body = JSON.parse((init?.body ?? "") as string) as {
      graph: { nodes: { id: string }[] };
      targets: string[] | null;
      node_labels: Record<string, string>;
    };
    expect(body.targets).toEqual([sourceId]);
    expect(body.node_labels).toEqual({ [sourceId]: "Demo Node" });
    expect(body.graph.nodes.map((node) => node.id)).toEqual([sourceId]);
  });

  it("saves a node as a trusted user template", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const saved = {
      ...template,
      name: "user_demo",
      title: "User Demo",
      category: "Custom",
      description: "Saved locally",
      source: "user" as const,
      code_hash: "user-hash",
    };
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify(saved), {
          status: 201,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const result = await useGraphStore.getState().saveNodeAsTemplate(id, {
      name: "user_demo",
      title: "User Demo",
      category: "Custom",
      description: "Saved locally",
      code: template.code,
    });
    vi.unstubAllGlobals();

    expect(result.name).toBe("user_demo");
    const node = useGraphStore.getState().nodes.find((item) => item.id === id);
    expect(node?.data.templateName).toBe("user_demo");
    expect(node?.data.isTrusted).toBe(true);
    expect(useGraphStore.getState().templates.some((item) => item.name === "user_demo")).toBe(true);
  });

  it("runs a sequential widget sweep and restores the original value", async () => {
    const template = await makeTemplate({
      node_schema: makeSchema({
        widgets: [{ name: "seed", data_type: "int", default: 1 }],
      }),
    });
    useGraphStore.setState({ templates: [template] });
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    useGraphStore.getState().selectOnlyNode(id);
    let runCounter = 0;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      runCounter += 1;
      const runId = `sweep-${runCounter}`;
      const body = JSON.parse(String(init?.body ?? "{}")) as {
        graph: { nodes: { widget_values: { seed?: number } }[] };
      };
      expect(body.graph.nodes[0]?.widget_values.seed).toBe(9 + runCounter);
      queueMicrotask(() => {
        useGraphStore.getState().applyExecutionEvent({
          type: "EXECUTION_STARTED",
          node_id: "",
          data: { run_id: runId, order: [id] },
        });
        useGraphStore.getState().applyExecutionEvent({
          type: "EXECUTION_COMPLETED",
          node_id: "",
          data: { previews: {}, cached: {} },
        });
      });
      return new Response(JSON.stringify({ status: "started", run_id: runId }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const result = await useGraphStore.getState().runSweep({
      nodeId: id,
      widget: "seed",
      values: [10, 11],
      suffixOutputs: false,
    });
    vi.unstubAllGlobals();

    expect(result).toMatchObject({ completed: 2, total: 2, status: "completed" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(
      useGraphStore.getState().nodes.find((node) => node.id === id)?.data
        .widgetValues.seed,
    ).toBe(1);
  });

  it("sends null targets for whole-graph runs", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    useGraphStore.getState().addNodeFromTemplate(template, { x: 0, y: 0 });

    const fetchMock = vi.fn<typeof fetch>(
      async () =>
        new Response(JSON.stringify({ status: "started", run_id: "run-8" }), {
          status: 202,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await useGraphStore.getState().runGraph();
    vi.unstubAllGlobals();

    const init = fetchMock.mock.calls[0]?.[1];
    const body = JSON.parse((init?.body ?? "") as string) as {
      targets: string[] | null;
      node_labels: Record<string, string>;
    };
    expect(body.targets).toBeNull();
    expect(Object.values(body.node_labels)).toEqual(["Demo Node"]);
  });
});

describe("cancelRun", () => {
  it("posts to the cancel endpoint and tolerates a 409", async () => {
    useGraphStore.setState((state) => ({
      execution: { ...state.execution, status: "running" },
    }));
    const fetchMock = vi.fn<typeof fetch>(
      async () =>
        new Response(
          JSON.stringify({ detail: "No execution is running" }),
          { status: 409, headers: { "Content-Type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await useGraphStore.getState().cancelRun();
    vi.unstubAllGlobals();

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0] ?? [];
    expect(String(url)).toContain("/api/cancel");
    expect(init?.method).toBe("POST");
    expect(useGraphStore.getState().execution.error).toBeNull();
  });

  it("does nothing when no execution is running", async () => {
    const fetchMock = vi.fn<typeof fetch>(
      async () =>
        new Response(JSON.stringify({ status: "cancelling" }), {
          status: 202,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await useGraphStore.getState().cancelRun();
    vi.unstubAllGlobals();

    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("serialiseGraph", () => {
  it("maps canvas nodes and edges to backend payloads", async () => {
    const template = await makeTemplate();
    const consumer = await makeTemplate({
      name: "sink",
      node_schema: makeSchema({ inputs: [{ name: "k", data_type: "ANY" }] }),
    });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 200, y: 0 });
    useGraphStore.getState().onConnect({
      source: sourceId,
      target: targetId,
      sourceHandle: "k",
      targetHandle: "k",
    });
    useGraphStore.getState().updateWidgetValue(sourceId, "k", 9);

    const payload = serialiseGraph(
      useGraphStore.getState().nodes,
      useGraphStore.getState().edges,
    );

    expect(payload.nodes).toEqual([
      { id: sourceId, code: template.code, widget_values: { k: 9 } },
      { id: targetId, code: consumer.code, widget_values: {} },
    ]);
    expect(payload.edges).toEqual([
      {
        source_node_id: sourceId,
        source_handle: "k",
        target_node_id: targetId,
        target_handle: "k",
      },
    ]);
  });
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("duplicateNode", () => {
  it("copies official-template code as trusted with an offset position", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 10, y: 20 });

    const copyId = await useGraphStore.getState().duplicateNode(sourceId);

    expect(copyId).not.toBeNull();
    expect(copyId).not.toBe(sourceId);
    const copy = useGraphStore
      .getState()
      .nodes.find((node) => node.id === copyId);
    expect(copy?.data.code).toBe(template.code);
    expect(copy?.data.isTrusted).toBe(true);
    expect(copy?.position).toEqual({ x: 58, y: 68 });
  });

  it("marks custom code duplicates as untrusted", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    useGraphStore.setState((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === sourceId
          ? {
              ...node,
              data: {
                ...node.data,
                code: "# custom\nreturn value: ANY\n",
                isTrusted: true,
              },
            }
          : node,
      ),
    }));

    const copyId = await useGraphStore.getState().duplicateNode(sourceId);

    const copy = useGraphStore
      .getState()
      .nodes.find((node) => node.id === copyId);
    expect(copy?.data.isTrusted).toBe(false);
  });

  it("returns null for unknown nodes", async () => {
    expect(await useGraphStore.getState().duplicateNode("missing")).toBeNull();
  });
});

describe("addScratchNode", () => {
  it("creates an untrusted node and fills its schema via the parser", async () => {
    const parsedSchema = makeSchema({
      widgets: [{ name: "factor", data_type: "float", default: 2.0 }],
      outputs: [{ name: "result", data_type: "ANY" }],
    });
    const fetchMock = vi.fn<typeof fetch>(async () => jsonResponse(parsedSchema));
    vi.stubGlobal("fetch", fetchMock);

    const id = useGraphStore.getState().addScratchNode({ x: 0, y: 0 });

    const created = useGraphStore
      .getState()
      .nodes.find((node) => node.id === id);
    expect(created?.data.code).toBe(SCRATCH_NODE_CODE);
    expect(created?.data.isTrusted).toBe(false);
    expect(created?.data.templateName).toBeNull();

    await vi.waitFor(() => {
      const current = useGraphStore
        .getState()
        .nodes.find((node) => node.id === id);
      expect(current?.data.schema.outputs).toHaveLength(1);
    });
    vi.unstubAllGlobals();

    const node = useGraphStore.getState().nodes.find((n) => n.id === id);
    expect(node?.data.widgetValues.factor).toBe(2.0);
    expect(node?.data.schema.outputs[0]?.name).toBe("result");
    const init = fetchMock.mock.calls[0]?.[1];
    const body = JSON.parse((init?.body ?? "") as string) as {
      code: string;
      node_id: string;
    };
    expect(body.node_id).toBe(id);
    expect(body.code).toBe(SCRATCH_NODE_CODE);
  });
});

describe("onConnect socket types", () => {
  it("rejects incompatible socket types and accepts ANY", async () => {
    const producer = await makeTemplate({
      node_schema: makeSchema({
        outputs: [{ name: "metrics", data_type: "METRICS" }],
      }),
    });
    const consumer = await makeTemplate({
      name: "knn",
      node_schema: makeSchema({
        inputs: [{ name: "train", data_type: "DATASET" }],
      }),
    });
    const anyConsumer = await makeTemplate({
      name: "flexible",
      node_schema: makeSchema({
        inputs: [{ name: "metrics", data_type: "ANY" }],
      }),
    });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(producer, { x: 0, y: 0 });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 200, y: 0 });
    const anyId = useGraphStore
      .getState()
      .addNodeFromTemplate(anyConsumer, { x: 200, y: 200 });

    useGraphStore.getState().onConnect({
      source: sourceId,
      target: targetId,
      sourceHandle: "metrics",
      targetHandle: "train",
    });
    expect(useGraphStore.getState().edges).toHaveLength(0);

    useGraphStore.getState().onConnect({
      source: sourceId,
      target: anyId,
      sourceHandle: "metrics",
      targetHandle: "metrics",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);
  });

  it("creates a source node when dropped on a widget target", async () => {
    const producer = await makeTemplate({
      name: "float_value",
      node_schema: makeSchema({
        outputs: [{ name: "result", data_type: "FLOAT" }],
      }),
    });
    const consumer = await makeTemplate({
      name: "split",
      node_schema: makeSchema({
        widgets: [{ name: "test_size", data_type: "float", default: 0.2 }],
      }),
    });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 200, y: 0 });

    const createdId = useGraphStore
      .getState()
      .addConnectedNodeFromTemplate(producer, { x: 0, y: 0 }, {
        nodeId: targetId,
        handleId: "test_size",
        handleType: "target",
        dataType: "FLOAT",
      });

    expect(createdId).not.toBeNull();
    const edge = useGraphStore.getState().edges[0];
    expect(edge?.source).toBe(createdId?.nodeId);
    expect(edge?.sourceHandle).toBe("result");
    expect(edge?.targetHandle).toBe("test_size");
  });

  it("wires primitive outputs into widget parameters", async () => {
    const producer = await makeTemplate({
      name: "float_value",
      node_schema: makeSchema({
        outputs: [{ name: "result", data_type: "FLOAT" }],
      }),
    });
    const consumer = await makeTemplate({
      name: "split",
      node_schema: makeSchema({
        widgets: [{ name: "test_size", data_type: "float", default: 0.2 }],
      }),
    });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(producer, { x: 0, y: 0 });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 200, y: 0 });
    const conflictingId = useGraphStore
      .getState()
      .addNodeFromTemplate(producer, { x: 0, y: 200 });

    useGraphStore.getState().onConnect({
      source: sourceId,
      target: targetId,
      sourceHandle: "result",
      targetHandle: "test_size",
    });

    const edge = useGraphStore.getState().edges[0];
    expect(edge?.target).toBe(targetId);
    expect(edge?.targetHandle).toBe("test_size");

    useGraphStore.getState().onConnect({
      source: conflictingId,
      target: targetId,
      sourceHandle: "result",
      targetHandle: "test_size",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);
  });
});

describe("restoreWorkflow", () => {
  it("re-parses schemas, re-checks trust and drops orphaned edges", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const parsedSchema = makeSchema({
      inputs: [{ name: "k", data_type: "ANY" }],
      widgets: [
        { name: "k", data_type: "int", default: 5 },
        { name: "w", data_type: "float", default: 0.2 },
      ],
      outputs: [{ name: "k", data_type: "ANY" }],
    });
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(parsedSchema)));

    const saved: SavedWorkflow = {
      version: 1,
      savedAt: "2026-01-01T00:00:00.000Z",
      nodes: [
        {
          id: "n1",
          label: "Demo Node",
          position: { x: 1, y: 2 },
          code: template.code,
          widgetValues: { k: 7 },
          templateName: "demo",
        },
        {
          id: "n2",
          label: "Custom",
          position: { x: 5, y: 6 },
          code: "# custom code\nreturn value: ANY\n",
          widgetValues: {},
          templateName: null,
        },
      ],
      edges: [
        { id: "e1", source: "n1", sourceHandle: "k", target: "n2", targetHandle: "k" },
        { id: "e2", source: "n1", sourceHandle: "missing", target: "n2", targetHandle: "k" },
        { id: "e3", source: "n1", sourceHandle: "k", target: "ghost", targetHandle: "k" },
        { id: "e4", source: "n1", sourceHandle: "k", target: "n2", targetHandle: "w" },
      ],
    };

    await useGraphStore.getState().restoreWorkflow(saved);
    vi.unstubAllGlobals();

    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(2);

    const restored = state.nodes.find((node) => node.id === "n1");
    expect(restored?.data.isTrusted).toBe(true);
    expect(restored?.data.templateCode).toBe(template.code);
    expect(restored?.data.widgetValues).toEqual({ k: 7, w: 0.2 });

    const custom = state.nodes.find((node) => node.id === "n2");
    expect(custom?.data.isTrusted).toBe(false);
    expect(custom?.data.templateCode).toBeNull();

    expect(state.edges.map((edge) => edge.id)).toEqual(["e1", "e4"]);
  });

  it("drops restored widget values that no longer match the schema", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const parsedSchema = makeSchema({
      widgets: [
        { name: "k", data_type: "int", default: 5 },
        {
          name: "level",
          data_type: "str",
          default: "low",
          options: ["low", "medium", "high"],
        },
      ],
    });
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(parsedSchema)));

    await useGraphStore.getState().restoreWorkflow({
      version: 1,
      savedAt: "",
      nodes: [
        {
          id: "n1",
          label: "Demo Node",
          position: { x: 0, y: 0 },
          code: template.code,
          widgetValues: { k: "not-a-number", level: "extreme" },
          templateName: "demo",
        },
      ],
      edges: [],
    });
    vi.unstubAllGlobals();

    expect(useGraphStore.getState().nodes[0]?.data.widgetValues).toEqual({
      k: 5,
      level: "low",
    });
  });
});

describe("restoreFromStorage", () => {
  it("restores autosaved workflows only when the canvas is empty", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const parsedSchema = makeSchema({
      outputs: [{ name: "k", data_type: "ANY" }],
    });
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(parsedSchema)));

    saveLocalWorkflow({
      version: 1,
      savedAt: "",
      nodes: [
        {
          id: "s1",
          label: "Saved",
          position: { x: 0, y: 0 },
          code: template.code,
          widgetValues: {},
          templateName: "demo",
        },
      ],
      edges: [],
    });

    await useGraphStore.getState().restoreFromStorage();
    vi.unstubAllGlobals();
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    expect(useGraphStore.getState().past).toHaveLength(0);

    const before = useGraphStore.getState().nodes;
    await useGraphStore.getState().restoreFromStorage();
    expect(useGraphStore.getState().nodes).toBe(before);
  });
});

function makeWorkflow(
  overrides: Partial<WorkflowTemplateInfo> = {},
): WorkflowTemplateInfo {
  return {
    name: "demo_flow",
    title: "Demo Flow",
    category: "Getting Started",
    description: "Demo workflow for tests",
    nodes: [
      {
        id: "load",
        template: "demo",
        position: { x: 0, y: 0 },
        widget_values: {},
      },
      {
        id: "tuned",
        template: "demo",
        position: { x: 240, y: 0 },
        widget_values: { k: 9 },
      },
    ],
    edges: [
      {
        source_node_id: "load",
        source_handle: "k",
        target_node_id: "tuned",
        target_handle: "df",
      },
    ],
    ...overrides,
  };
}

describe("loadWorkflowTemplates", () => {
  it("stores fetched workflows and clears prior errors", async () => {
    useGraphStore.setState({ workflowsError: "stale" });
    const workflow = makeWorkflow();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse([workflow])),
    );

    await useGraphStore.getState().loadWorkflowTemplates();
    vi.unstubAllGlobals();

    const state = useGraphStore.getState();
    expect(state.workflowsLoaded).toBe(true);
    expect(state.workflows).toHaveLength(1);
    expect(state.workflows[0]?.name).toBe("demo_flow");
    expect(state.workflowsError).toBeNull();
  });

  it("records fetch failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("offline");
      }),
    );

    await useGraphStore.getState().loadWorkflowTemplates();
    vi.unstubAllGlobals();

    const state = useGraphStore.getState();
    expect(state.workflowsLoaded).toBe(false);
    expect(state.workflowsError).toContain("Backend unreachable");
  });
});

describe("loadWorkflowTemplate", () => {
  it("materialises nodes with widget overrides and resets execution", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({
      templates: [template],
      nodes: [await builtNode(template)],
      execution: { ...useGraphStore.getState().execution, status: "running" },
    });

    const ok = useGraphStore.getState().loadWorkflowTemplate(makeWorkflow());

    expect(ok).toBe(true);
    const state = useGraphStore.getState();
    expect(state.nodes.map((node) => node.id)).toEqual(["load", "tuned"]);
    expect(state.nodes[0]?.data.widgetValues).toEqual({ k: 5 });
    expect(state.nodes[1]?.data.widgetValues).toEqual({ k: 9 });
    expect(state.nodes[0]?.data.isTrusted).toBe(true);
    expect(state.nodes[0]?.position).toEqual({ x: 0, y: 0 });
    expect(state.edges).toHaveLength(1);
    expect(state.edges[0]?.sourceHandle).toBe("k");
    expect(state.execution.status).toBe("idle");
    expect(state.workflowsError).toBeNull();
  });

  it("still loads workflows that reference hidden templates", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({
      templates: [{ ...template, hidden: true }],
      nodes: [],
      edges: [],
    });

    const ok = useGraphStore.getState().loadWorkflowTemplate(makeWorkflow());

    expect(ok).toBe(true);
    const state = useGraphStore.getState();
    expect(state.nodes.map((node) => node.id)).toEqual(["load", "tuned"]);
    expect(state.nodes[0]?.data.isTrusted).toBe(true);
    expect(state.workflowsError).toBeNull();
  });

  it("reports missing templates without clobbering the canvas", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const existing = await builtNode(template);
    useGraphStore.setState({ nodes: [existing] });

    const workflow = makeWorkflow({
      nodes: [
        {
          id: "ghost",
          template: "missing_template",
          position: { x: 0, y: 0 },
          widget_values: {},
        },
      ],
      edges: [],
    });
    const ok = useGraphStore.getState().loadWorkflowTemplate(workflow);

    expect(ok).toBe(false);
    const state = useGraphStore.getState();
    expect(state.workflowsError).toContain("missing_template");
    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0]?.id).toBe(existing.id);
  });
});

async function builtNode(template: TemplateInfo): Promise<AppNode> {
  return {
    id: "existing",
    type: "complex",
    position: { x: 0, y: 0 },
    data: {
      label: template.title,
      templateName: template.name,
      templateCode: template.code,
      code: template.code,
      schema: template.node_schema,
      widgetValues: { k: 5 },
      isTrusted: true,
      color: "default",
    },
  };
}

function selectNodes(ids: string[]): void {
  useGraphStore.setState((state) => ({
    nodes: state.nodes.map((node) => ({
      ...node,
      selected: ids.includes(node.id),
    })),
  }));
}

describe("undo/redo", () => {
  it("undoes and redoes node additions", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    expect(useGraphStore.getState().past).toHaveLength(1);

    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes).toHaveLength(0);
    expect(useGraphStore.getState().future).toHaveLength(1);

    useGraphStore.getState().redo();
    expect(useGraphStore.getState().nodes.map((node) => node.id)).toEqual([id]);
    expect(useGraphStore.getState().past).toHaveLength(1);
    expect(useGraphStore.getState().future).toHaveLength(0);
  });

  it("restores edges when undoing a node deletion", async () => {
    const template = await makeTemplate();
    const source = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const target = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 200, y: 0 });
    useGraphStore.getState().onConnect({
      source,
      target,
      sourceHandle: "k",
      targetHandle: "df",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);
    const edgeId = useGraphStore.getState().edges[0]?.id as string;
    const historyBefore = useGraphStore.getState().past.length;

    useGraphStore.getState().checkpoint();
    useGraphStore.getState().onEdgesChange([{ type: "remove", id: edgeId }]);
    useGraphStore.getState().onNodesChange([{ type: "remove", id: target }]);
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    expect(useGraphStore.getState().edges).toHaveLength(0);
    expect(useGraphStore.getState().past).toHaveLength(historyBefore + 1);

    useGraphStore.getState().undo();
    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(2);
    expect(state.edges).toHaveLength(1);
  });

  it("undoes connections", async () => {
    const template = await makeTemplate();
    const source = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const target = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 200, y: 0 });
    useGraphStore.getState().onConnect({
      source,
      target,
      sourceHandle: "k",
      targetHandle: "df",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);

    useGraphStore.getState().undo();
    expect(useGraphStore.getState().edges).toHaveLength(0);

    useGraphStore.getState().redo();
    expect(useGraphStore.getState().edges).toHaveLength(1);
  });

  it("coalesces rapid widget edits into a single undo step", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    vi.useFakeTimers();
    try {
      useGraphStore.getState().updateWidgetValue(id, "k", 6);
      useGraphStore.getState().updateWidgetValue(id, "k", 7);
      expect(useGraphStore.getState().past).toHaveLength(2);

      useGraphStore.getState().undo();
      const node = useGraphStore.getState().nodes.find((n) => n.id === id);
      expect(node?.data.widgetValues).toEqual({ k: 5 });
    } finally {
      vi.useRealTimers();
    }
  });

  it("starts a new history entry after the coalesce window", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    vi.useFakeTimers();
    try {
      useGraphStore.getState().updateWidgetValue(id, "k", 6);
      vi.advanceTimersByTime(1000);
      useGraphStore.getState().updateWidgetValue(id, "k", 7);
      expect(useGraphStore.getState().past).toHaveLength(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("coalesces a node drag into a single undo step", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });

    useGraphStore.getState().onNodesChange([
      { type: "position", id, position: { x: 10, y: 0 }, dragging: true },
    ]);
    useGraphStore.getState().onNodesChange([
      { type: "position", id, position: { x: 20, y: 0 }, dragging: true },
    ]);
    expect(useGraphStore.getState().past).toHaveLength(2);

    useGraphStore.getState().onNodesChange([
      { type: "position", id, position: { x: 30, y: 0 }, dragging: false },
    ]);
    expect(useGraphStore.getState().nodes[0]?.position).toEqual({ x: 30, y: 0 });

    useGraphStore.getState().onNodesChange([
      { type: "position", id, position: { x: 40, y: 0 }, dragging: true },
    ]);
    expect(useGraphStore.getState().past).toHaveLength(3);

    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes[0]?.position).toEqual({ x: 30, y: 0 });
  });

  it("ignores selection and measurement changes", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    useGraphStore.getState().onNodesChange([
      { type: "select", id, selected: true },
      { type: "dimensions", id, dimensions: { width: 100, height: 40 } },
    ]);
    expect(useGraphStore.getState().past).toHaveLength(1);
  });

  it("undoes clearCanvas and workflow loads", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const existing = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });

    useGraphStore.getState().clearCanvas();
    expect(useGraphStore.getState().nodes).toHaveLength(0);
    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes.map((node) => node.id)).toEqual([
      existing,
    ]);

    const ok = useGraphStore.getState().loadWorkflowTemplate(makeWorkflow());
    expect(ok).toBe(true);
    expect(useGraphStore.getState().nodes.map((node) => node.id)).toEqual([
      "load",
      "tuned",
    ]);
    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes.map((node) => node.id)).toEqual([
      existing,
    ]);
  });

  it("undoes approved code commits", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const code = "# demo node\nk: int = 9\nreturn k: ANY\n";
    const hash = await sha256Hex(code);
    useGraphStore.setState((state) => ({
      approvedHashes: new Set(state.approvedHashes).add(`${id}:${hash}`),
    }));

    await useGraphStore.getState().saveNodeCode(id, code, template.node_schema);
    expect(useGraphStore.getState().nodes[0]?.data.code).toBe(code);

    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes[0]?.data.code).toBe(template.code);
  });

  it("clears the redo stack when a new action lands", async () => {
    const template = await makeTemplate();
    const first = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    useGraphStore.getState().undo();
    expect(useGraphStore.getState().future).toHaveLength(1);

    const second = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 200, y: 0 });
    expect(useGraphStore.getState().future).toHaveLength(0);

    useGraphStore.getState().redo();
    expect(useGraphStore.getState().nodes.map((node) => node.id)).toEqual([
      second,
    ]);
    expect(useGraphStore.getState().nodes.map((node) => node.id)).not.toContain(
      first,
    );
  });

  it("caps the history stack and blocks undo while running", async () => {
    const template = await makeTemplate();
    for (let index = 0; index < 105; index += 1) {
      useGraphStore
        .getState()
        .addNodeFromTemplate(template, { x: index, y: 0 });
    }
    expect(useGraphStore.getState().past).toHaveLength(100);

    useGraphStore.setState((state) => ({
      execution: { ...state.execution, status: "running" },
    }));
    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes).toHaveLength(105);

    useGraphStore.setState((state) => ({
      execution: { ...state.execution, status: "idle" },
    }));
    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes).toHaveLength(104);
  });
});

describe("clipboard", () => {
  it("copies and pastes selected nodes with new ids and remapped edges", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const source = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const target = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 200, y: 0 });
    useGraphStore.getState().onConnect({
      source,
      target,
      sourceHandle: "k",
      targetHandle: "df",
    });
    selectNodes([source, target]);

    expect(useGraphStore.getState().copySelection()).toBe(true);
    const pasted = await useGraphStore.getState().pasteClipboard();

    expect(pasted).toHaveLength(2);
    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(4);
    const pastedNodes = state.nodes.filter((node) => pasted.includes(node.id));
    expect(pastedNodes.every((node) => node.selected)).toBe(true);
    expect(
      state.nodes
        .filter((node) => !pasted.includes(node.id))
        .every((node) => !node.selected),
    ).toBe(true);
    expect(
      pastedNodes.every((node) => node.data.isTrusted),
    ).toBe(true);
    const original = state.nodes.find((node) => node.id === source);
    const copySource = pastedNodes.find(
      (node) => node.position.x === (original?.position.x ?? 0) + 40,
    );
    expect(copySource).toBeDefined();

    const pastedEdges = state.edges.filter(
      (edge) => pasted.includes(edge.source) && pasted.includes(edge.target),
    );
    expect(pastedEdges).toHaveLength(1);
    expect(pastedEdges[0]?.sourceHandle).toBe("k");
    expect(pastedEdges[0]?.targetHandle).toBe("df");

    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes).toHaveLength(2);
  });

  it("only copies internal edges for the selection", async () => {
    const template = await makeTemplate();
    const source = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const target = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 200, y: 0 });
    useGraphStore.getState().onConnect({
      source,
      target,
      sourceHandle: "k",
      targetHandle: "df",
    });
    selectNodes([source]);

    expect(useGraphStore.getState().copySelection()).toBe(true);
    expect(useGraphStore.getState().clipboard?.edges).toHaveLength(0);
  });

  it("cuts selected nodes and their attached edges", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const source = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const target = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 200, y: 0 });
    useGraphStore.getState().onConnect({
      source,
      target,
      sourceHandle: "k",
      targetHandle: "df",
    });
    useGraphStore.setState((state) => ({
      execution: {
        ...state.execution,
        richOutputs: {
          [source]: [
            {
              kind: "text",
              source: "print",
              text: "cached output",
              truncated: false,
            },
          ],
        },
      },
    }));
    selectNodes([source, target]);

    expect(useGraphStore.getState().cutSelection()).toBe(true);

    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(0);
    expect(state.edges).toHaveLength(0);
    expect(state.clipboard?.nodes).toHaveLength(2);
    expect(state.clipboard?.edges).toHaveLength(1);
    expect(state.execution.richOutputs).toEqual({});

    state.undo();
    expect(useGraphStore.getState().nodes).toHaveLength(2);
    expect(useGraphStore.getState().edges).toHaveLength(1);
  });

  it("marks pasted modified code as untrusted", async () => {
    const id = useGraphStore.getState().addScratchNode({ x: 0, y: 0 });
    selectNodes([id]);

    expect(useGraphStore.getState().copySelection()).toBe(true);
    const pasted = await useGraphStore.getState().pasteClipboard();

    const copied = useGraphStore
      .getState()
      .nodes.find((node) => node.id === pasted[0]);
    expect(copied?.data.isTrusted).toBe(false);
    expect(copied?.data.code).toBe(SCRATCH_NODE_CODE);
  });

  it("keeps the clipboard snapshot independent of later edits", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    selectNodes([id]);
    useGraphStore.getState().copySelection();

    useGraphStore.getState().deleteNode(id);
    expect(useGraphStore.getState().nodes).toHaveLength(0);

    const pasted = await useGraphStore.getState().pasteClipboard();
    expect(pasted).toHaveLength(1);
    expect(useGraphStore.getState().nodes[0]?.data.code).toBe(template.code);
  });

  it("no-ops on empty selections and empty clipboards", async () => {
    expect(useGraphStore.getState().copySelection()).toBe(false);
    expect(useGraphStore.getState().cutSelection()).toBe(false);
    expect(useGraphStore.getState().clipboard).toBeNull();
    expect(await useGraphStore.getState().pasteClipboard()).toEqual([]);
    expect(useGraphStore.getState().past).toHaveLength(0);
  });

  it("duplicates the selection in one step", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    selectNodes([id]);

    const created = await useGraphStore.getState().duplicateSelection();
    expect(created).toHaveLength(1);
    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(2);
    const copy = state.nodes.find((node) => node.id === created[0]);
    expect(copy?.position).toEqual({ x: 40, y: 40 });
  });
});

describe("addConnectedNodeFromTemplate", () => {
  it("creates a connected input node and selects it", async () => {
    const producer = await makeTemplate();
    const consumer = await makeTemplate({
      name: "sink",
      title: "Sink",
      node_schema: makeSchema({
        inputs: [{ name: "value", data_type: "ANY" }],
      }),
    });
    useGraphStore.setState({ templates: [producer, consumer] });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(producer, { x: 0, y: 0 });
    useGraphStore.getState().selectOnlyNode(sourceId);

    const created = useGraphStore.getState().addConnectedNodeFromTemplate(
      consumer,
      { x: 260, y: 20 },
      {
        nodeId: sourceId,
        handleId: "k",
        handleType: "source",
        dataType: "ANY",
      },
    );

    expect(created).not.toBeNull();
    const state = useGraphStore.getState();
    const added = state.nodes.find((node) => node.id === created?.nodeId);
    expect(added?.position).toEqual({ x: 260, y: 20 });
    expect(added?.selected).toBe(true);
    expect(state.nodes.find((node) => node.id === sourceId)?.selected).toBe(
      false,
    );
    expect(state.edges).toEqual([
      {
        id: `${sourceId}:k->${created?.nodeId}:value`,
        source: sourceId,
        sourceHandle: "k",
        target: created?.nodeId,
        targetHandle: "value",
      },
    ]);
  });

  it("creates a source node when dragging from an input", async () => {
    const producer = await makeTemplate();
    const consumer = await makeTemplate({
      name: "sink",
      title: "Sink",
      node_schema: makeSchema({
        inputs: [{ name: "value", data_type: "ANY" }],
      }),
    });
    useGraphStore.setState({ templates: [producer, consumer] });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 300, y: 0 });

    const created = useGraphStore.getState().addConnectedNodeFromTemplate(
      producer,
      { x: 60, y: 0 },
      {
        nodeId: targetId,
        handleId: "value",
        handleType: "target",
        dataType: "ANY",
      },
    );

    expect(created).not.toBeNull();
    expect(useGraphStore.getState().edges[0]).toEqual({
      id: `${created?.nodeId}:k->${targetId}:value`,
      source: created?.nodeId,
      sourceHandle: "k",
      target: targetId,
      targetHandle: "value",
    });
  });

  it("creates a source node when dragging from a widget parameter", async () => {
    const producer = await makeTemplate();
    const consumer = await makeTemplate({ name: "sink", title: "Sink" });
    useGraphStore.setState({ templates: [producer, consumer] });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 300, y: 0 });

    const created = useGraphStore.getState().addConnectedNodeFromTemplate(
      producer,
      { x: 60, y: 0 },
      {
        nodeId: targetId,
        handleId: "k",
        handleType: "target",
        dataType: "INT",
      },
    );

    expect(created).not.toBeNull();
    expect(useGraphStore.getState().edges[0]).toEqual({
      id: `${created?.nodeId}:k->${targetId}:k`,
      source: created?.nodeId,
      sourceHandle: "k",
      target: targetId,
      targetHandle: "k",
    });
  });

  it("rejects incompatible templates and occupied inputs", async () => {
    const metricsTemplate = await makeTemplate({
      node_schema: makeSchema({
        inputs: [{ name: "df", data_type: "DATAFRAME" }],
        outputs: [{ name: "metrics", data_type: "METRICS" }],
      }),
    });
    const tensorTemplate = await makeTemplate({
      name: "tensor",
      title: "Tensor",
      node_schema: makeSchema({
        inputs: [{ name: "tensor", data_type: "TENSOR" }],
      }),
    });
    const flexibleTemplate = await makeTemplate({
      name: "flexible",
      title: "Flexible",
      node_schema: makeSchema({
        outputs: [{ name: "result", data_type: "ANY" }],
      }),
    });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(metricsTemplate, { x: 0, y: 0 });

    expect(
      useGraphStore.getState().addConnectedNodeFromTemplate(
        tensorTemplate,
        { x: 200, y: 0 },
        {
          nodeId: sourceId,
          handleId: "metrics",
          handleType: "source",
          dataType: "METRICS",
        },
      ),
    ).toBeNull();

    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(tensorTemplate, { x: 200, y: 0 });
    useGraphStore.setState((state) => ({
      edges: [
        ...state.edges,
        {
          id: `${sourceId}:metrics->${targetId}:tensor`,
          source: sourceId,
          sourceHandle: "metrics",
          target: targetId,
          targetHandle: "tensor",
        },
      ],
    }));
    expect(
      useGraphStore.getState().addConnectedNodeFromTemplate(
        flexibleTemplate,
        { x: 0, y: 200 },
        {
          nodeId: targetId,
          handleId: "tensor",
          handleType: "target",
          dataType: "TENSOR",
        },
      ),
    ).toBeNull();
  });
});

describe("node selection and appearance", () => {
  it("renames, colors, selects and deletes nodes", async () => {
    const template = await makeTemplate();
    const first = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const second = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 200, y: 0 });

    useGraphStore.getState().selectOnlyNode(first);
    expect(
      useGraphStore.getState().nodes.filter((node) => node.selected),
    ).toHaveLength(1);

    expect(useGraphStore.getState().renameNode(first, "  Tuned  ")).toBe(true);
    expect(
      useGraphStore.getState().nodes.find((node) => node.id === first)?.data
        .label,
    ).toBe("Tuned");
    expect(useGraphStore.getState().renameNode(first, "   ")).toBe(false);

    expect(
      useGraphStore.getState().setNodeColor([first, second], "rose"),
    ).toBe(true);
    expect(
      useGraphStore
        .getState()
        .nodes.filter((node) => node.data.color === "rose"),
    ).toHaveLength(2);
    useGraphStore.getState().undo();
    expect(
      useGraphStore
        .getState()
        .nodes.filter((node) => node.data.color === "default"),
    ).toHaveLength(2);

    useGraphStore.getState().selectOnlyNode(second);
    expect(useGraphStore.getState().deleteSelection()).toBe(true);
    expect(useGraphStore.getState().nodes.map((node) => node.id)).toEqual([
      first,
    ]);
    expect(useGraphStore.getState().deleteSelection()).toBe(false);
  });

  it("pastes the clipboard at an explicit anchor", async () => {
    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 100, y: 100 });
    useGraphStore.getState().selectOnlyNode(id);
    useGraphStore.getState().copySelection();

    const pasted = await useGraphStore
      .getState()
      .pasteClipboard({ x: 10, y: 20 });

    const copy = useGraphStore
      .getState()
      .nodes.find((node) => node.id === pasted[0]);
    expect(copy?.position).toEqual({ x: 10, y: 20 });
  });
});

describe("canvas arrangement", () => {
  function makeNode(
    id: string,
    x: number,
    y: number,
    width: number,
    height: number,
    selected = true,
  ): AppNode {
    return {
      id,
      type: "complex",
      position: { x, y },
      selected,
      measured: { width, height },
      data: {
        label: id,
        templateName: null,
        templateCode: null,
        code: "",
        schema: makeSchema(),
        widgetValues: {},
        isTrusted: false,
        color: "default",
      },
    };
  }

  function setNodes(nodes: AppNode[]): void {
    useGraphStore.setState({ nodes, past: [], future: [] });
  }

  function positions(): Record<string, { x: number; y: number }> {
    return Object.fromEntries(
      useGraphStore
        .getState()
        .nodes.map((node) => [node.id, node.position]),
    );
  }

  it.each([
    ["align-left", { a: { x: 10, y: 20 }, b: { x: 10, y: 80 }, c: { x: 10, y: 40 } }],
    [
      "align-center-x",
      { a: { x: 55, y: 20 }, b: { x: 75, y: 80 }, c: { x: 65, y: 40 } },
    ],
    [
      "align-right",
      { a: { x: 100, y: 20 }, b: { x: 140, y: 80 }, c: { x: 120, y: 40 } },
    ],
    ["align-top", { a: { x: 10, y: 20 }, b: { x: 50, y: 20 }, c: { x: 120, y: 20 } }],
    [
      "align-center-y",
      { a: { x: 10, y: 60 }, b: { x: 50, y: 50 }, c: { x: 120, y: 40 } },
    ],
    [
      "align-bottom",
      { a: { x: 10, y: 100 }, b: { x: 50, y: 80 }, c: { x: 120, y: 60 } },
    ],
  ] as const)("applies %s", (mode, expected) => {
    setNodes([
      makeNode("a", 10, 20, 100, 40),
      makeNode("b", 50, 80, 60, 60),
      makeNode("c", 120, 40, 80, 80),
    ]);

    expect(useGraphStore.getState().arrangeSelection(mode)).toBe(true);
    expect(positions()).toEqual(expected);

    useGraphStore.getState().undo();
    expect(positions()).toEqual({
      a: { x: 10, y: 20 },
      b: { x: 50, y: 80 },
      c: { x: 120, y: 40 },
    });
  });

  it("distributes three or more nodes with equal gaps", () => {
    setNodes([
      makeNode("a", 0, 0, 100, 40),
      makeNode("b", 300, 50, 100, 40),
      makeNode("c", 700, 100, 100, 40),
    ]);

    expect(useGraphStore.getState().arrangeSelection("distribute-x")).toBe(
      true,
    );
    expect(positions()).toEqual({
      a: { x: 0, y: 0 },
      b: { x: 350, y: 50 },
      c: { x: 700, y: 100 },
    });

    useGraphStore.setState((state) => ({
      nodes: state.nodes.map((node) => ({
        ...node,
        position: {
          x: node.position.x,
          y: node.id === "a" ? 0 : node.id === "b" ? 30 : 200,
        },
      })),
    }));
    expect(useGraphStore.getState().arrangeSelection("distribute-y")).toBe(
      true,
    );
    expect(positions()).toEqual({
      a: { x: 0, y: 0 },
      b: { x: 350, y: 100 },
      c: { x: 700, y: 200 },
    });
  });

  it("requires three selected nodes to distribute", () => {
    setNodes([
      makeNode("a", 0, 0, 100, 40),
      makeNode("b", 300, 50, 100, 40),
    ]);

    expect(useGraphStore.getState().arrangeSelection("distribute-x")).toBe(
      false,
    );
    expect(useGraphStore.getState().past).toHaveLength(0);
  });

  it("arranges selected nodes into a grid and leaves others alone", () => {
    setNodes([
      makeNode("a", 0, 0, 100, 40),
      makeNode("b", 200, 20, 120, 60),
      makeNode("c", 400, 40, 80, 80),
      makeNode("d", 600, 60, 90, 50),
      makeNode("outside", 900, 900, 100, 40, false),
    ]);

    expect(useGraphStore.getState().arrangeSelection("grid")).toBe(true);
    expect(positions()).toEqual({
      a: { x: 0, y: 0 },
      b: { x: 148, y: 0 },
      c: { x: 0, y: 92 },
      d: { x: 148, y: 92 },
      outside: { x: 900, y: 900 },
    });
  });

  it("selects all nodes and clears the selection without history", () => {
    setNodes([
      makeNode("a", 0, 0, 100, 40, false),
      makeNode("b", 200, 0, 100, 40, false),
    ]);

    useGraphStore.getState().selectAllNodes();
    expect(
      useGraphStore.getState().nodes.every((node) => node.selected),
    ).toBe(true);
    expect(useGraphStore.getState().past).toHaveLength(0);

    useGraphStore.getState().clearSelection();
    expect(
      useGraphStore.getState().nodes.some((node) => node.selected),
    ).toBe(false);
    expect(useGraphStore.getState().past).toHaveLength(0);
  });
});

describe("frames", () => {
  function makeNode(
    id: string,
    x: number,
    y: number,
    selected = true,
  ): AppNode {
    return {
      id,
      type: "complex",
      position: { x, y },
      selected,
      measured: { width: 100, height: 40 },
      data: {
        label: id,
        templateName: null,
        templateCode: null,
        code: "",
        schema: makeSchema(),
        widgetValues: {},
        isTrusted: false,
        color: "default",
      },
    };
  }

  function setNodes(nodes: AppNode[]): void {
    useGraphStore.setState({ nodes, edges: [], past: [], future: [] });
  }

  function node(id: string): AppNode | undefined {
    return useGraphStore.getState().nodes.find((item) => item.id === id);
  }

  it("groups selected nodes into a frame and undoes in one step", () => {
    setNodes([
      makeNode("a", 10, 20),
      makeNode("b", 50, 80),
      makeNode("c", 400, 400, false),
    ]);

    const frameId = useGraphStore.getState().groupSelection();
    expect(frameId).not.toBeNull();

    const frame = node(frameId as string);
    expect(frame?.data.isFrame).toBe(true);
    expect(frame?.position).toEqual({ x: -14, y: -20 });
    expect(frame?.width).toBe(280);
    expect(frame?.height).toBe(180);
    expect(frame?.selected).toBe(true);

    const grouped = node("a");
    expect(grouped?.parentId).toBe(frameId);
    expect(grouped?.extent).toBe("parent");
    expect(grouped?.selected).toBe(false);
    expect(grouped?.position).toEqual({ x: 24, y: 40 });
    expect(node("b")?.position).toEqual({ x: 64, y: 100 });
    expect(node("c")?.parentId).toBeUndefined();

    const payload = serialiseGraph(
      useGraphStore.getState().nodes,
      useGraphStore.getState().edges,
    );
    expect(payload.nodes.map((item) => item.id).sort()).toEqual([
      "a",
      "b",
      "c",
    ]);

    useGraphStore.getState().undo();
    expect(useGraphStore.getState().nodes).toHaveLength(3);
    expect(node("a")?.parentId).toBeUndefined();
    expect(node("a")?.position).toEqual({ x: 10, y: 20 });
    expect(node("a")?.selected).toBe(true);
  });

  it("ungroups a selected frame while preserving absolute positions", () => {
    setNodes([makeNode("a", 10, 20), makeNode("b", 50, 80)]);
    const frameId = useGraphStore.getState().groupSelection();
    expect(frameId).not.toBeNull();

    expect(useGraphStore.getState().ungroupSelection()).toBe(true);

    expect(node(frameId as string)).toBeUndefined();
    expect(node("a")?.parentId).toBeUndefined();
    expect(node("a")?.selected).toBe(true);
    expect(node("a")?.position).toEqual({ x: 10, y: 20 });
    expect(node("b")?.position).toEqual({ x: 50, y: 80 });

    useGraphStore.getState().undo();
    expect(node(frameId as string)?.data.isFrame).toBe(true);
    expect(node("a")?.parentId).toBe(frameId);
  });

  it("deleting a frame keeps unselected children and their positions", () => {
    setNodes([makeNode("a", 10, 20), makeNode("b", 50, 80)]);
    const frameId = useGraphStore.getState().groupSelection();
    expect(frameId).not.toBeNull();

    expect(useGraphStore.getState().deleteSelection()).toBe(true);

    expect(node(frameId as string)).toBeUndefined();
    expect(node("a")?.parentId).toBeUndefined();
    expect(node("a")?.extent).toBeUndefined();
    expect(node("a")?.position).toEqual({ x: 10, y: 20 });
    expect(node("b")?.position).toEqual({ x: 50, y: 80 });
  });

  it("copies and pastes a frame together with its children", async () => {
    setNodes([makeNode("a", 10, 20), makeNode("b", 50, 80)]);
    const frameId = useGraphStore.getState().groupSelection();
    expect(frameId).not.toBeNull();

    expect(useGraphStore.getState().copySelection()).toBe(true);
    expect(useGraphStore.getState().clipboard?.nodes).toHaveLength(3);

    const pasted = await useGraphStore.getState().pasteClipboard();
    expect(pasted).toHaveLength(3);

    const state = useGraphStore.getState();
    const frames = state.nodes.filter((item) => item.data.isFrame);
    expect(frames).toHaveLength(2);
    const pastedFrame = frames.find((item) => item.id !== frameId);
    expect(pastedFrame).toBeDefined();
    const children = state.nodes.filter(
      (item) => item.parentId === pastedFrame?.id,
    );
    expect(children).toHaveLength(2);
    expect(children.map((item) => item.position)).toEqual(
      expect.arrayContaining([
        { x: 24, y: 40 },
        { x: 64, y: 100 },
      ]),
    );
    expect(children.every((item) => item.selected)).toBe(true);
    expect(
      state.nodes.filter((item) => item.parentId === frameId),
    ).toHaveLength(2);
  });

  it("unparents children when React Flow removes a frame", () => {
    setNodes([makeNode("a", 10, 20), makeNode("b", 50, 80)]);
    const frameId = useGraphStore.getState().groupSelection();
    expect(frameId).not.toBeNull();

    useGraphStore.getState().onNodesChange([
      { type: "remove", id: frameId as string },
    ]);

    expect(node(frameId as string)).toBeUndefined();
    expect(node("a")?.parentId).toBeUndefined();
    expect(node("a")?.position).toEqual({ x: 10, y: 20 });
  });

  it("restores frames before their children with dimensions intact", async () => {
    const saved: SavedWorkflow = {
      version: 1,
      savedAt: "",
      nodes: [
        {
          id: "child",
          label: "Child",
          position: { x: 24, y: 40 },
          code: "return 1: ANY\n",
          widgetValues: {},
          templateName: null,
          parentId: "frame",
        },
        {
          id: "frame",
          label: "Frame 1",
          position: { x: 100, y: 100 },
          code: "",
          widgetValues: {},
          templateName: null,
          isFrame: true,
          width: 640,
          height: 420,
        },
      ],
      edges: [],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse(makeSchema({ node_id: "child" }))),
    );

    await useGraphStore.getState().restoreWorkflow(saved);
    vi.unstubAllGlobals();

    const state = useGraphStore.getState();
    expect(state.nodes.map((item) => item.id)).toEqual(["frame", "child"]);
    const frame = state.nodes[0];
    expect(frame?.width).toBe(640);
    expect(frame?.height).toBe(420);
    expect(frame?.data.isFrame).toBe(true);
    expect(state.nodes[1]?.parentId).toBe("frame");
    expect(state.nodes[1]?.extent).toBe("parent");
  });
});

describe("subgraphs", () => {
  function makeNode(
    id: string,
    inputs: InputSocket[],
    outputs: OutputSocket[],
    x: number,
    y: number,
    selected = false,
  ): AppNode {
    return {
      id,
      type: "complex",
      position: { x, y },
      selected,
      measured: { width: 200, height: 100 },
      data: {
        label: id.toUpperCase(),
        templateName: null,
        templateCode: null,
        code: "value = 1\nreturn value: ANY\n",
        schema: makeSchema({
          node_id: id,
          inputs,
          outputs,
        }),
        widgetValues: {},
        isTrusted: true,
        color: "default",
      },
    };
  }

  function makeEdge(
    source: string,
    sourceHandle: string,
    target: string,
    targetHandle: string,
  ): AppEdge {
    return {
      id: `${source}:${sourceHandle}->${target}:${targetHandle}`,
      source,
      sourceHandle,
      target,
      targetHandle,
    };
  }

  function setGraph(): void {
    const source = makeNode(
      "src",
      [],
      [{ name: "df", data_type: "DATAFRAME" }],
      0,
      0,
    );
    const a = makeNode(
      "a",
      [{ name: "df", data_type: "DATAFRAME" }],
      [{ name: "mid", data_type: "ANY" }],
      200,
      40,
      true,
    );
    const b = makeNode(
      "b",
      [{ name: "mid", data_type: "ANY" }],
      [{ name: "result", data_type: "ANY" }],
      400,
      80,
      true,
    );
    const sink = makeNode(
      "sink",
      [{ name: "result", data_type: "ANY" }],
      [],
      600,
      40,
    );
    useGraphStore.setState({
      nodes: [source, a, b, sink],
      edges: [
        makeEdge("src", "df", "a", "df"),
        makeEdge("a", "mid", "b", "mid"),
        makeEdge("b", "result", "sink", "result"),
      ],
      subgraphs: {},
      scopeStack: [],
      rootGraph: null,
      past: [],
      future: [],
      clipboard: null,
    });
  }

  it("creates a subgraph, rewires boundary edges and undoes it", () => {
    setGraph();
    const instanceId = useGraphStore.getState().createSubgraphFromSelection();
    expect(instanceId).not.toBeNull();

    const state = useGraphStore.getState();
    const instance = state.nodes.find((node) => node.id === instanceId);
    expect(instance?.data.isSubgraph).toBe(true);
    expect(instance?.data.schema.inputs.map((socket) => socket.name)).toEqual([
      "df",
    ]);
    expect(instance?.data.schema.outputs.map((socket) => socket.name)).toEqual([
      "result",
    ]);
    expect(state.nodes.map((node) => node.id).sort()).toEqual(
      [instanceId as string, "sink", "src"].sort(),
    );
    expect(Object.keys(state.subgraphs)).toHaveLength(1);
    const document = state.getWorkflowDocument();
    expect(Object.keys(document.subgraphs ?? {})).toHaveLength(1);
    const savedDefinition = Object.values(document.subgraphs ?? {})[0];
    expect(
      savedDefinition?.nodes.some(
        (node) =>
          node.id === SUBGRAPH_INPUT_NODE_ID ||
          node.id === SUBGRAPH_OUTPUT_NODE_ID,
      ),
    ).toBe(false);

    const flattened = state.getExecutionGraph();
    expect(flattened.graph.nodes.map((node) => node.id).sort()).toEqual(
      [`sub:${instanceId}/a`, `sub:${instanceId}/b`, "sink", "src"].sort(),
    );
    expect(flattened.labels[`sub:${instanceId}/a`]).toBe("A");
    expect(flattened.graph.edges).toHaveLength(3);

    useGraphStore.getState().undo();
    expect(Object.keys(useGraphStore.getState().subgraphs)).toHaveLength(0);
    expect(useGraphStore.getState().nodes).toHaveLength(4);
    expect(useGraphStore.getState().nodes.some((node) => node.data.isSubgraph)).toBe(
      false,
    );
  });

  it("enters and exits the nested canvas with shared managers", () => {
    setGraph();
    const instanceId = useGraphStore.getState().createSubgraphFromSelection();
    expect(instanceId).not.toBeNull();

    expect(useGraphStore.getState().enterSubgraph(instanceId as string)).toBe(
      true,
    );
    const nested = useGraphStore.getState();
    expect(nested.scopeStack).toHaveLength(1);
    expect(nested.nodes.map((node) => node.id).sort()).toEqual([
      SUBGRAPH_INPUT_NODE_ID,
      SUBGRAPH_OUTPUT_NODE_ID,
      "a",
      "b",
    ]);
    expect(nested.edges.some((edge) => edge.source === SUBGRAPH_INPUT_NODE_ID)).toBe(
      true,
    );
    expect(nested.edges.some((edge) => edge.target === SUBGRAPH_OUTPUT_NODE_ID)).toBe(
      true,
    );

    const flatWhileNested = nested.getExecutionGraph();
    expect(
      flatWhileNested.graph.nodes.map((node) => node.id).sort(),
    ).toEqual(
      [`sub:${instanceId}/a`, `sub:${instanceId}/b`, "sink", "src"].sort(),
    );

    expect(useGraphStore.getState().exitSubgraph()).toBe(true);
    expect(useGraphStore.getState().scopeStack).toHaveLength(0);
    expect(useGraphStore.getState().nodes).toHaveLength(3);
  });

  it("reuses definitions and flattens nested subgraphs with path ids", () => {
    setGraph();
    const outerInstance = useGraphStore
      .getState()
      .createSubgraphFromSelection();
    expect(outerInstance).not.toBeNull();
    const outerDefinition = useGraphStore.getState().nodes.find(
      (node) => node.id === outerInstance,
    )?.data.subgraphId as string;

    const second = useGraphStore.getState().addSubgraphInstance(
      outerDefinition,
      { x: 800, y: 200 },
    );
    expect(second).not.toBeNull();
    expect(
      useGraphStore.getState().getExecutionGraph().graph.nodes.filter((node) =>
        node.id.startsWith("sub:"),
      ),
    ).toHaveLength(4);

    useGraphStore.getState().enterSubgraph(outerInstance as string);
    useGraphStore.setState((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === "a" || node.id === "b"
          ? { ...node, selected: true }
          : node,
      ),
    }));
    const innerInstance = useGraphStore
      .getState()
      .createSubgraphFromSelection();
    expect(innerInstance).not.toBeNull();
    useGraphStore.getState().exitSubgraph();

    const state = useGraphStore.getState();
    const document = state.getWorkflowDocument();
    expect(Object.keys(document.subgraphs ?? {})).toHaveLength(2);
    const flattened = state.getExecutionGraph();
    const ids = flattened.graph.nodes.map((node) => node.id);
    expect(
      ids.some((id) =>
        id.startsWith(`sub:${outerInstance}/${innerInstance}/`),
      ),
    ).toBe(true);
  });

  it("adds manager ports and infers them from new-handle connections", () => {
    setGraph();
    const instanceId = useGraphStore.getState().createSubgraphFromSelection();
    expect(instanceId).not.toBeNull();
    useGraphStore.getState().enterSubgraph(instanceId as string);

    const inputPort = useGraphStore
      .getState()
      .addSubgraphPort("input", "ANY", "extra");
    const outputPort = useGraphStore
      .getState()
      .addSubgraphPort("output", "ANY", "extra");
    expect(inputPort).toBe("extra");
    expect(outputPort).toBe("extra");

    const free = makeNode(
      "free",
      [
        { name: "x", data_type: "DATAFRAME" },
        { name: "y", data_type: "ANY" },
      ],
      [{ name: "out", data_type: "METRICS" }],
      800,
      200,
    );
    useGraphStore.setState((state) => ({
      nodes: [...state.nodes, free],
    }));

    useGraphStore.getState().onConnect({
      source: SUBGRAPH_INPUT_NODE_ID,
      sourceHandle: NEW_SUBGRAPH_INPUT_HANDLE,
      target: "free",
      targetHandle: "x",
    });
    useGraphStore.getState().onConnect({
      source: "free",
      sourceHandle: "out",
      target: SUBGRAPH_OUTPUT_NODE_ID,
      targetHandle: NEW_SUBGRAPH_OUTPUT_HANDLE,
    });

    const nested = useGraphStore.getState();
    const inputManager = nested.nodes.find(
      (node) => node.id === SUBGRAPH_INPUT_NODE_ID,
    );
    const outputManager = nested.nodes.find(
      (node) => node.id === SUBGRAPH_OUTPUT_NODE_ID,
    );
    expect(
      inputManager?.data.schema.outputs.map((socket) => socket.name),
    ).toEqual(["df", "extra", "x"]);
    expect(
      outputManager?.data.schema.inputs.map((socket) => socket.name),
    ).toEqual(["result", "extra", "out"]);
    expect(
      nested.edges.some(
        (edge) =>
          edge.sourceHandle === "x" &&
          edge.target === "free" &&
          edge.targetHandle === "x",
      ),
    ).toBe(true);
    expect(
      nested.edges.some(
        (edge) =>
          edge.source === "free" &&
          edge.sourceHandle === "out" &&
          edge.targetHandle === "out",
      ),
    ).toBe(true);

    useGraphStore.getState().exitSubgraph();
    const instance = useGraphStore
      .getState()
      .nodes.find((node) => node.id === instanceId);
    expect(
      instance?.data.schema.inputs.map((socket) => socket.name),
    ).toEqual(["df", "extra", "x"]);
    expect(
      instance?.data.schema.outputs.map((socket) => socket.name),
    ).toEqual(["result", "extra", "out"]);
  });

  it("removes a definition when its last instance is deleted", () => {
    setGraph();
    const first = useGraphStore.getState().createSubgraphFromSelection();
    expect(first).not.toBeNull();
    const definitionId = useGraphStore
      .getState()
      .nodes.find((node) => node.id === first)?.data.subgraphId as string;
    const second = useGraphStore.getState().addSubgraphInstance(
      definitionId,
      { x: 800, y: 200 },
    );
    expect(second).not.toBeNull();

    useGraphStore.getState().selectOnlyNode(second as string);
    expect(useGraphStore.getState().deleteSelection()).toBe(true);
    expect(useGraphStore.getState().subgraphs[definitionId]).toBeDefined();

    useGraphStore.getState().selectOnlyNode(first as string);
    expect(useGraphStore.getState().deleteSelection()).toBe(true);
    expect(useGraphStore.getState().subgraphs[definitionId]).toBeUndefined();
    expect(
      useGraphStore.getState().nodes.some((node) => node.data.isSubgraph),
    ).toBe(false);

    useGraphStore.getState().undo();
    expect(useGraphStore.getState().subgraphs[definitionId]).toBeDefined();
    expect(
      useGraphStore.getState().nodes.some((node) => node.id === first),
    ).toBe(true);
    useGraphStore.getState().redo();
    expect(useGraphStore.getState().subgraphs[definitionId]).toBeUndefined();
  });

  it("connects new nodes from the manager search handles", async () => {
    const template = await makeTemplate();
    setGraph();
    const instanceId = useGraphStore.getState().createSubgraphFromSelection();
    expect(instanceId).not.toBeNull();
    useGraphStore.getState().enterSubgraph(instanceId as string);

    const inputResult = useGraphStore
      .getState()
      .addConnectedNodeFromTemplate(
        template,
        { x: 900, y: 100 },
        {
          nodeId: SUBGRAPH_INPUT_NODE_ID,
          handleId: NEW_SUBGRAPH_INPUT_HANDLE,
          handleType: "source",
          dataType: "ANY",
        },
      );
    expect(inputResult).not.toBeNull();

    const outputResult = useGraphStore
      .getState()
      .addConnectedNodeFromTemplate(
        template,
        { x: 900, y: 300 },
        {
          nodeId: SUBGRAPH_OUTPUT_NODE_ID,
          handleId: NEW_SUBGRAPH_OUTPUT_HANDLE,
          handleType: "target",
          dataType: "ANY",
        },
      );
    expect(outputResult).not.toBeNull();

    const state = useGraphStore.getState();
    const inputManager = state.nodes.find(
      (node) => node.id === SUBGRAPH_INPUT_NODE_ID,
    );
    const outputManager = state.nodes.find(
      (node) => node.id === SUBGRAPH_OUTPUT_NODE_ID,
    );
    expect(
      inputManager?.data.schema.outputs.map((socket) => socket.name),
    ).toContain("df_2");
    expect(
      outputManager?.data.schema.inputs.map((socket) => socket.name),
    ).toContain("k");
    expect(
      state.edges.some(
        (edge) =>
          edge.source === SUBGRAPH_INPUT_NODE_ID &&
          edge.sourceHandle === "df_2" &&
          edge.target === inputResult?.nodeId &&
          edge.targetHandle === "df",
      ),
    ).toBe(true);
    expect(
      state.edges.some(
        (edge) =>
          edge.source === outputResult?.nodeId &&
          edge.sourceHandle === "k" &&
          edge.target === SUBGRAPH_OUTPUT_NODE_ID &&
          edge.targetHandle === "k",
      ),
    ).toBe(true);
  });

  it("blocks execution while editing a subgraph", async () => {
    setGraph();
    const instanceId = useGraphStore.getState().createSubgraphFromSelection();
    useGraphStore.getState().enterSubgraph(instanceId as string);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await useGraphStore.getState().runGraph();
    vi.unstubAllGlobals();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(useGraphStore.getState().execution.error?.errorType).toBe(
      "LocalValidation",
    );
    expect(useGraphStore.getState().execution.error?.message).toContain(
      "Exit the subgraph",
    );
  });
});

describe("resetNode", () => {
  it("restores template code, metadata, color and widget defaults", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    useGraphStore.getState().updateWidgetValue(id, "k", 9);
    useGraphStore.getState().setNodeColor([id], "rose");
    useGraphStore.getState().renameNode(id, "Custom name");
    useGraphStore.setState((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === id
          ? {
              ...node,
              data: { ...node.data, code: "# custom\n" },
            }
          : node,
      ),
      execution: {
        ...state.execution,
        nodeStates: { [id]: { status: "completed" } },
      },
    }));

    expect(useGraphStore.getState().resetNode(id)).toBe(true);

    const state = useGraphStore.getState();
    const node = state.nodes.find((candidate) => candidate.id === id);
    expect(node?.data.code).toBe(template.code);
    expect(node?.data.label).toBe(template.title);
    expect(node?.data.widgetValues).toEqual({ k: 5 });
    expect(node?.data.color).toBe("default");
    expect(node?.data.isTrusted).toBe(true);
    expect(state.execution.nodeStates[id]).toBeUndefined();
  });

  it("drops edges that no longer match the restored schema", async () => {
    const template = await makeTemplate();
    const consumer = await makeTemplate({
      name: "sink",
      node_schema: makeSchema({
        inputs: [{ name: "value", data_type: "ANY" }],
      }),
    });
    useGraphStore.setState({ templates: [template, consumer] });
    const sourceId = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const targetId = useGraphStore
      .getState()
      .addNodeFromTemplate(consumer, { x: 200, y: 0 });
    useGraphStore.getState().onConnect({
      source: sourceId,
      target: targetId,
      sourceHandle: "k",
      targetHandle: "value",
    });
    useGraphStore.setState((state) => ({
      edges: [
        ...state.edges,
        {
          id: `${sourceId}:ghost->${targetId}:value`,
          source: sourceId,
          sourceHandle: "ghost",
          target: targetId,
          targetHandle: "value",
        },
      ],
    }));

    useGraphStore.getState().resetNode(sourceId);

    expect(useGraphStore.getState().edges.map((edge) => edge.id)).toEqual([
      `${sourceId}:k->${targetId}:value`,
    ]);
  });

  it("rejects unknown nodes and missing template metadata", async () => {
    expect(useGraphStore.getState().resetNode("missing")).toBe(false);

    const template = await makeTemplate();
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    useGraphStore.setState({ templates: [] });
    expect(useGraphStore.getState().resetNode(id)).toBe(false);
  });
});

describe("seed controls", () => {
  it("advances the seed after an uncached KSampler completion", async () => {
    const template = await makeTemplate({
      node_schema: makeSchema({
        widgets: [
          { name: "seed", data_type: "int", default: 41 },
          {
            name: "control_after_generate",
            data_type: "str",
            default: "increment",
          },
        ],
      }),
    });
    const id = useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    const store = useGraphStore.getState();

    store.applyExecutionEvent({
      type: "NODE_COMPLETED",
      node_id: id,
      data: { cached: false, outputs: {} },
    });
    expect(
      useGraphStore.getState().nodes.find((node) => node.id === id)?.data
        .widgetValues.seed,
    ).toBe(42);

    useGraphStore.getState().applyExecutionEvent({
      type: "NODE_COMPLETED",
      node_id: id,
      data: { cached: true, outputs: {} },
    });
    expect(
      useGraphStore.getState().nodes.find((node) => node.id === id)?.data
        .widgetValues.seed,
    ).toBe(42);
  });
});

