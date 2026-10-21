import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ProjectSummary } from "../lib/api";
import {
  clearLocalProjectIdentity,
  clearLocalWorkflow,
  loadLocalProjectIdentity,
} from "../lib/persistence";
import type { SavedWorkflow } from "../lib/persistence";
import type { NodeSchema } from "../types/graph";
import { useGraphStore } from "./useGraphStore";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function makeSchema(): NodeSchema {
  return {
    node_id: "",
    inputs: [],
    widgets: [],
    outputs: [],
    dependencies: [],
    syntax_errors: [],
  };
}

function makeWorkflow(name: string | null = null): SavedWorkflow {
  return {
    ...(name ? { name } : {}),
    version: 1,
    savedAt: "2026-10-20T12:00:00Z",
    nodes: [
      {
        id: "saved-node",
        label: "Saved Node",
        position: { x: 12, y: 24 },
        code: "x = 1\nreturn x: ANY\n",
        widgetValues: {},
        templateName: null,
      },
    ],
    edges: [],
  };
}

function makeSummary(
  overrides: Partial<ProjectSummary> = {},
): ProjectSummary {
  return {
    slug: "iris-demo",
    name: "Iris Demo",
    saved_at: "2026-10-20T12:00:00Z",
    node_count: 1,
    edge_count: 0,
    version_count: 0,
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
    past: [],
    future: [],
    clipboard: null,
    historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },
    editingNodeId: null,
    pendingApproval: null,
    approvalBusy: false,
    approvalError: null,
    approvedHashes: new Set<string>(),
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
  let idCounter = 0;
  vi.stubGlobal("crypto", {
    randomUUID: () => `project-node-${++idCounter}`,
    subtle: {
      digest: async () => new Uint8Array(32).buffer,
    },
  });
});

describe("project save", () => {
  it("prompts for a name, creates the project, then updates silently", async () => {
    const created = makeSummary();
    const updated = makeSummary({ version_count: 1 });
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url === "/api/parse") {
          return jsonResponse(makeSchema());
        }
        if (url === "/api/projects" && init?.method === "POST") {
          return jsonResponse({ status: "created", project: created });
        }
        return jsonResponse({ status: "updated", project: updated });
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    useGraphStore.getState().addScratchNode({ x: 0, y: 0 });

    useGraphStore.getState().requestSave();
    expect(useGraphStore.getState().saveDialogOpen).toBe(true);

    const saved = await useGraphStore.getState().saveProject("Iris Demo");
    expect(saved).toBe(true);
    const projectCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).startsWith("/api/projects"),
    );
    expect(projectCalls).toHaveLength(1);
    const [url, init] = projectCalls[0] ?? [];
    expect(url).toBe("/api/projects");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toMatchObject({
      name: "Iris Demo",
      overwrite: false,
    });

    let state = useGraphStore.getState();
    expect(state.projectName).toBe("Iris Demo");
    expect(state.projectSlug).toBe("iris-demo");
    expect(state.saveDialogOpen).toBe(false);
    expect(loadLocalProjectIdentity()).toEqual({
      name: "Iris Demo",
      slug: "iris-demo",
    });

    await useGraphStore.getState().saveProject();
    const saveCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).startsWith("/api/projects"),
    );
    expect(saveCalls).toHaveLength(2);
    expect(saveCalls[1]?.[0]).toBe("/api/projects/iris-demo");
    expect(saveCalls[1]?.[1]?.method).toBe("PUT");

    state = useGraphStore.getState();
    expect(state.projects[0]?.version_count).toBe(1);
    expect(state.saveDialogOpen).toBe(false);
  });

  it("keeps the save dialog open on a name conflict", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse(
          { detail: "A project named 'Taken' already exists" },
          409,
        ),
      ),
    );
    useGraphStore.getState().addScratchNode({ x: 0, y: 0 });

    const saved = await useGraphStore.getState().saveProject("Taken");

    expect(saved).toBe(false);
    const state = useGraphStore.getState();
    expect(state.projectSlug).toBeNull();
    expect(state.saveDialogOpen).toBe(true);
    expect(state.saveError).toContain("already exists");
    expect(state.saveBusy).toBe(false);
  });
});

describe("project loading and recovery", () => {
  it("opens a project and restores its identity", async () => {
    const detail = {
      ...makeSummary(),
      workflow: makeWorkflow("Iris Demo"),
    };
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL) =>
        String(input) === "/api/projects/iris-demo"
          ? jsonResponse(detail)
          : jsonResponse(makeSchema()),
    );
    vi.stubGlobal("fetch", fetchMock);

    const opened = await useGraphStore.getState().openProject("iris-demo");

    expect(opened).toBe(true);
    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0]?.data.label).toBe("Saved Node");
    expect(state.projectName).toBe("Iris Demo");
    expect(state.projectSlug).toBe("iris-demo");
    expect(state.past).toHaveLength(0);
    expect(loadLocalProjectIdentity()).toEqual({
      name: "Iris Demo",
      slug: "iris-demo",
    });
  });

  it("falls back to the workspace session when localStorage is empty", async () => {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL) =>
        String(input) === "/api/session"
          ? jsonResponse({
              saved_at: "2026-10-20T12:00:00Z",
              project_slug: "recovered",
              project_name: "Recovered",
              workflow: makeWorkflow("Recovered"),
            })
          : jsonResponse(makeSchema()),
    );
    vi.stubGlobal("fetch", fetchMock);

    await useGraphStore.getState().restoreFromStorage();

    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(1);
    expect(state.projectName).toBe("Recovered");
    expect(state.projectSlug).toBe("recovered");
    expect(fetchMock.mock.calls.some(([url]) => url === "/api/session")).toBe(
      true,
    );
  });

  it("clears project identity when the current project is deleted", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ status: "deleted", slug: "iris-demo" })),
    );
    useGraphStore.setState({
      projects: [makeSummary()],
      projectName: "Iris Demo",
      projectSlug: "iris-demo",
    });

    const deleted = await useGraphStore.getState().deleteProject("iris-demo");

    expect(deleted).toBe(true);
    const state = useGraphStore.getState();
    expect(state.projects).toHaveLength(0);
    expect(state.projectName).toBeNull();
    expect(state.projectSlug).toBeNull();
    expect(loadLocalProjectIdentity()).toEqual({ name: null, slug: null });
  });
});
