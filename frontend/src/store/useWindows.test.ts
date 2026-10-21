// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearLocalWorkflow,
  clearLocalWorkspace,
  saveLocalWorkspace,
} from "../lib/persistence";
import type { NodeSchema, TemplateInfo } from "../types/graph";
import type {
  WorkflowWindow,
} from "./useGraphStore";
import { useGraphStore } from "./useGraphStore";

const schema: NodeSchema = {
  node_id: "demo",
  inputs: [],
  widgets: [],
  outputs: [{ name: "k", data_type: "ANY" }],
  dependencies: [],
  syntax_errors: [],
};

const template: TemplateInfo = {
  name: "demo",
  title: "Demo",
  category: "Values",
  description: "Demo template",
  hidden: false,
  code: "# demo\nreturn k: ANY\n",
  code_hash: "demo-hash",
  node_schema: schema,
};

function idleExecution() {
  return {
    status: "idle" as const,
    runId: null,
    order: [],
    nodeStates: {},
    previews: {},
    richOutputs: {},
    cached: {},
    error: null,
  };
}

function makeWindow(id: string, title = id): WorkflowWindow {
  return {
    id,
    title,
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

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  clearLocalWorkflow();
  clearLocalWorkspace();
  let counter = 0;
  vi.stubGlobal("crypto", {
    randomUUID: () => `window-${++counter}`,
    subtle: {
      digest: async () => new Uint8Array(32).buffer,
    },
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse(schema)),
  );
  const window = makeWindow("initial", "Untitled 1");
  useGraphStore.setState({
    templates: [template],
    templatesLoaded: true,
    templatesError: null,
    workflows: [],
    workflowsLoaded: true,
    workflowsError: null,
    projects: [],
    projectsLoaded: true,
    projectsError: null,
    windows: [window],
    activeWindowId: window.id,
    executionWindowId: null,
    nodes: [],
    edges: [],
    subgraphs: {},
    scopeStack: [],
    rootGraph: null,
    past: [],
    future: [],
    clipboard: null,
    historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },
    projectName: null,
    projectSlug: null,
    editingNodeId: null,
    pendingApproval: null,
    approvalError: null,
    execution: idleExecution(),
  });
});

describe("workflow windows", () => {
  it("keeps graph and history independent per window", () => {
    const store = useGraphStore.getState();
    store.addNodeFromTemplate(template, { x: 0, y: 0 });
    expect(useGraphStore.getState().past).toHaveLength(1);

    useGraphStore.getState().newWorkflowWindow();
    expect(useGraphStore.getState().nodes).toHaveLength(0);
    expect(useGraphStore.getState().past).toHaveLength(0);
    expect(useGraphStore.getState().windows).toHaveLength(2);

    const first = useGraphStore.getState().windows[0];
    expect(first).toBeDefined();
    useGraphStore.getState().activateWorkflowWindow(first!.id);
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    expect(useGraphStore.getState().past).toHaveLength(1);
  });

  it("opens templates in a new window", () => {
    const workflow = {
      name: "demo_flow",
      title: "Demo Flow",
      category: "Getting Started",
      description: "Demo",
      nodes: [
        {
          id: "n1",
          template: "demo",
          position: { x: 0, y: 0 },
          widget_values: {},
        },
      ],
      edges: [],
    };

    expect(
      useGraphStore.getState().openWorkflowTemplateInWindow(workflow),
    ).toBe(true);
    const state = useGraphStore.getState();
    expect(state.windows).toHaveLength(2);
    expect(state.nodes.map((node) => node.id)).toEqual(["n1"]);
    expect(state.windows[1]?.title).toBe("Demo Flow");
  });

  it("opens an imported workflow in a new window", async () => {
    const opened = await useGraphStore.getState().openWorkflowInWindow({
      name: "Imported",
      version: 1,
      savedAt: "",
      nodes: [
        {
          id: "n1",
          label: "Demo",
          position: { x: 0, y: 0 },
          code: template.code,
          widgetValues: {},
          templateName: "demo",
        },
      ],
      edges: [],
    });

    expect(opened).toBe(true);
    const state = useGraphStore.getState();
    expect(state.windows).toHaveLength(2);
    expect(state.nodes).toHaveLength(1);
    expect(state.projectName).toBe("Imported");
    expect(state.windows[1]?.title).toBe("Imported");
  });

  it("closes a window and activates a neighbour", () => {
    useGraphStore.getState().newWorkflowWindow();
    const secondId = useGraphStore.getState().activeWindowId;
    expect(useGraphStore.getState().windows).toHaveLength(2);

    expect(useGraphStore.getState().closeWorkflowWindow(secondId)).toBe(true);
    const state = useGraphStore.getState();
    expect(state.windows).toHaveLength(1);
    expect(state.activeWindowId).toBe("initial");
  });

  it("routes execution events to an inactive running window", () => {
    const running: WorkflowWindow = {
      ...makeWindow("running", "Running"),
      execution: { ...idleExecution(), status: "running", runId: "run-1" },
    };
    useGraphStore.setState((state) => ({
      windows: [state.windows[0]!, running],
      activeWindowId: "initial",
      executionWindowId: "running",
    }));

    useGraphStore.getState().applyExecutionEvent({
      type: "NODE_STARTED",
      node_id: "n1",
      data: {},
    });

    const state = useGraphStore.getState();
    expect(state.execution.nodeStates).toEqual({});
    expect(
      state.windows.find((window) => window.id === "running")?.execution
        .nodeStates.n1?.status,
    ).toBe("running");
  });

  it("restores all saved windows from workspace storage", async () => {
    saveLocalWorkspace({
      version: 1,
      activeWindowId: "saved-two",
      windows: [
        {
          id: "saved-one",
          title: "First",
          projectName: "First",
          projectSlug: null,
          workflow: {
            name: "First",
            version: 1,
            savedAt: "",
            nodes: [
              {
                id: "n1",
                label: "Demo",
                position: { x: 0, y: 0 },
                code: template.code,
                widgetValues: {},
                templateName: "demo",
              },
            ],
            edges: [],
          },
        },
        {
          id: "saved-two",
          title: "Second",
          projectName: null,
          projectSlug: null,
          workflow: {
            version: 1,
            savedAt: "",
            nodes: [],
            edges: [],
          },
        },
      ],
    });

    await useGraphStore.getState().restoreFromStorage();

    const state = useGraphStore.getState();
    expect(state.windows.map((window) => window.id)).toEqual([
      "saved-one",
      "saved-two",
    ]);
    expect(state.activeWindowId).toBe("saved-two");
    expect(state.nodes).toHaveLength(0);

    useGraphStore.getState().activateWorkflowWindow("saved-one");
    expect(useGraphStore.getState().nodes).toHaveLength(1);
  });
});
