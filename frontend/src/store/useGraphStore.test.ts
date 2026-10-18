import { beforeEach, describe, expect, it, vi } from "vitest";

import { sha256Hex } from "../lib/hash";
import { clearLocalWorkflow, saveLocalWorkflow } from "../lib/persistence";
import type { SavedWorkflow } from "../lib/persistence";
import type { NodeSchema, TemplateInfo } from "../types/graph";
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
    nodes: [],
    edges: [],
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
      cached: {},
      error: null,
    },
  });
}

beforeEach(() => {
  resetStore();
  clearLocalWorkflow();
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
    };
    expect(body.targets).toEqual([sourceId]);
    expect(body.graph.nodes.map((node) => node.id)).toEqual([sourceId]);
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
    };
    expect(body.targets).toBeNull();
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
});

describe("restoreWorkflow", () => {
  it("re-parses schemas, re-checks trust and drops orphaned edges", async () => {
    const template = await makeTemplate();
    useGraphStore.setState({ templates: [template] });
    const parsedSchema = makeSchema({
      inputs: [{ name: "k", data_type: "ANY" }],
      widgets: [{ name: "k", data_type: "int", default: 5 }],
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
      ],
    };

    await useGraphStore.getState().restoreWorkflow(saved);
    vi.unstubAllGlobals();

    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(2);

    const restored = state.nodes.find((node) => node.id === "n1");
    expect(restored?.data.isTrusted).toBe(true);
    expect(restored?.data.templateCode).toBe(template.code);
    expect(restored?.data.widgetValues).toEqual({ k: 7 });

    const custom = state.nodes.find((node) => node.id === "n2");
    expect(custom?.data.isTrusted).toBe(false);
    expect(custom?.data.templateCode).toBeNull();

    expect(state.edges).toHaveLength(1);
    expect(state.edges[0]?.id).toBe("e1");
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

    const before = useGraphStore.getState().nodes;
    await useGraphStore.getState().restoreFromStorage();
    expect(useGraphStore.getState().nodes).toBe(before);
  });
});
