// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { startWorkflowAutosave } from "../hooks/useWorkflowAutosave";
import { useGraphStore } from "../store/useGraphStore";
import type {
  AppEdge,
  AppNode,
  NodeSchema,
  SubgraphDefinition,
} from "../types/graph";
import {
  SOCKET_TYPE_NAMES,
  SUBGRAPH_INPUT_NODE_ID,
  SUBGRAPH_OUTPUT_NODE_ID,
} from "../types/graph";
import { createSubgraphInstanceNode } from "./subgraph";
import {
  AUTOSAVE_DELAY_MS,
  MAX_NODE_CODE_CHARS,
  MAX_WORKFLOW_NODES,
  STORAGE_KEY,
  WORKSPACE_STORAGE_KEY,
  clearLocalProjectIdentity,
  clearLocalWorkflow,
  clearLocalWorkspace,
  downloadTextFile,
  loadLocalProjectIdentity,
  loadLocalWorkflow,
  loadLocalWorkspace,
  parseWorkflowJson,
  saveLocalProjectIdentity,
  saveLocalWorkflow,
  saveLocalWorkspace,
  serialiseWorkflow,
} from "./persistence";

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

function makeNode(id: string): AppNode {
  return {
    id,
    type: "complex",
    position: { x: 1, y: 2 },
    data: {
      label: "Load Iris",
      templateName: "load_iris",
      templateCode: "# template\n",
      code: "x = 1\n",
      schema: makeSchema(),
      widgetValues: { k: 5 },
      isTrusted: true,
      color: "default",
    },
  };
}

function makeFrame(id: string): AppNode {
  return {
    id,
    type: "complex",
    position: { x: 100, y: 100 },
    width: 640,
    height: 420,
    zIndex: -1,
    data: {
      label: "Frame 1",
      templateName: null,
      templateCode: null,
      code: "",
      schema: makeSchema(),
      widgetValues: {},
      isTrusted: true,
      color: "sky",
      isFrame: true,
    },
  };
}

function makeEdge(): AppEdge {
  return {
    id: "e1",
    source: "a",
    sourceHandle: "df",
    target: "b",
    targetHandle: "df",
  };
}

describe("serialiseWorkflow / parseWorkflowJson", () => {
  it("round-trips through JSON", () => {
    const saved = serialiseWorkflow(
      [makeNode("a"), makeNode("b")],
      [makeEdge()],
    );
    const reparsed = parseWorkflowJson(JSON.stringify(saved));

    expect(reparsed.version).toBe(saved.version);
    expect(reparsed.nodes).toEqual(saved.nodes);
    expect(reparsed.edges).toEqual(saved.edges);
  });

  it("persists node colors and defaults unknown or missing values", () => {
    const node = makeNode("a");
    node.data.color = "amber";
    const saved = serialiseWorkflow([node], []);

    expect(saved.nodes[0]?.color).toBe("amber");
    expect(
      parseWorkflowJson(JSON.stringify(saved)).nodes[0]?.color,
    ).toBe("amber");

    const invalid = JSON.parse(JSON.stringify(saved)) as {
      nodes: Array<Record<string, unknown>>;
    };
    invalid.nodes[0]!.color = "hotpink";
    expect(
      parseWorkflowJson(JSON.stringify(invalid)).nodes[0]?.color,
    ).toBe("default");

    delete invalid.nodes[0]!.color;
    expect(
      parseWorkflowJson(JSON.stringify(invalid)).nodes[0]?.color,
    ).toBe("default");
  });

  it("round-trips frame dimensions and child relationships", () => {
    const child = makeNode("child");
    child.position = { x: 24, y: 40 };
    child.parentId = "frame";
    child.extent = "parent";
    const saved = serialiseWorkflow([makeFrame("frame"), child], []);

    expect(saved.nodes[0]?.isFrame).toBe(true);
    expect(saved.nodes[0]?.width).toBe(640);
    expect(saved.nodes[0]?.height).toBe(420);
    expect(saved.nodes[1]?.parentId).toBe("frame");

    const reparsed = parseWorkflowJson(JSON.stringify(saved));
    expect(reparsed.nodes[0]?.isFrame).toBe(true);
    expect(reparsed.nodes[0]?.width).toBe(640);
    expect(reparsed.nodes[1]?.parentId).toBe("frame");
  });

  it("round-trips explicit dimensions for regular nodes", () => {
    const resized = makeNode("resized");
    resized.width = 420;
    resized.height = 300;
    const saved = serialiseWorkflow([resized], []);

    expect(saved.nodes[0]?.width).toBe(420);
    expect(saved.nodes[0]?.height).toBe(300);
    const reparsed = parseWorkflowJson(JSON.stringify(saved));
    expect(reparsed.nodes[0]?.width).toBe(420);
    expect(reparsed.nodes[0]?.height).toBe(300);
  });

  it("round-trips reusable subgraphs and strips implicit manager nodes", () => {
    const definition: SubgraphDefinition = {
      id: "def-1",
      name: "Preprocess",
      inputs: [{ name: "df", data_type: "DATAFRAME", optional: true }],
      outputs: [{ name: "result", data_type: "ANY" }],
      nodes: [makeNode("inside")],
      edges: [
        {
          id: "in",
          source: SUBGRAPH_INPUT_NODE_ID,
          sourceHandle: "df",
          target: "inside",
          targetHandle: "df",
        },
        {
          id: "out",
          source: "inside",
          sourceHandle: "result",
          target: SUBGRAPH_OUTPUT_NODE_ID,
          targetHandle: "result",
        },
      ],
      inputPosition: { x: -260, y: 0 },
      outputPosition: { x: 300, y: 0 },
    };
    const instance = createSubgraphInstanceNode("instance", definition, {
      x: 100,
      y: 100,
    });
    const saved = serialiseWorkflow(
      [instance],
      [],
      "Nested",
      { "def-1": definition },
    );

    expect(saved.subgraphs?.["def-1"]?.nodes.map((node) => node.id)).toEqual([
      "inside",
    ]);
    const reparsed = parseWorkflowJson(JSON.stringify(saved));
    expect(reparsed.subgraphs?.["def-1"]?.inputs).toEqual([
      { name: "df", data_type: "DATAFRAME", optional: true },
    ]);
    expect(reparsed.nodes[0]?.isSubgraph).toBe(true);
    expect(reparsed.nodes[0]?.subgraphId).toBe("def-1");
  });

  it("round-trips every socket type on subgraph boundary ports", () => {
    const types = [...SOCKET_TYPE_NAMES];
    const definition: SubgraphDefinition = {
      id: "all-types",
      name: "All Types",
      inputs: types.map((data_type, index) => ({
        name: `input_${index}`,
        data_type,
      })),
      outputs: types.map((data_type, index) => ({
        name: `output_${index}`,
        data_type,
      })),
      nodes: [makeNode("inside")],
      edges: [],
      inputPosition: { x: -260, y: 0 },
      outputPosition: { x: 300, y: 0 },
    };
    const instance = createSubgraphInstanceNode("instance", definition, {
      x: 100,
      y: 100,
    });
    const saved = serialiseWorkflow(
      [instance],
      [],
      "All Types",
      { "all-types": definition },
    );
    const reparsed = parseWorkflowJson(JSON.stringify(saved));

    expect(reparsed.subgraphs?.["all-types"]?.inputs).toEqual(
      definition.inputs,
    );
    expect(reparsed.subgraphs?.["all-types"]?.outputs).toEqual(
      definition.outputs,
    );
  });

  it("rejects missing, cyclic and malformed subgraph references", () => {
    const base = {
      version: 1,
      savedAt: "",
      nodes: [
        {
          id: "instance",
          label: "Instance",
          position: { x: 0, y: 0 },
          code: "",
          widgetValues: {},
          templateName: null,
          isSubgraph: true,
          subgraphId: "missing",
        },
      ],
      edges: [],
    };
    expect(() => parseWorkflowJson(JSON.stringify(base))).toThrow(
      "missing subgraph",
    );

    const definition = {
      id: "a",
      name: "A",
      inputs: [],
      outputs: [],
      nodes: [
        {
          id: "b-instance",
          label: "B",
          position: { x: 0, y: 0 },
          code: "",
          widgetValues: {},
          templateName: null,
          isSubgraph: true,
          subgraphId: "b",
        },
      ],
      edges: [],
    };
    const cyclic = {
      ...base,
      nodes: [
        {
          ...base.nodes[0],
          subgraphId: "a",
        },
      ],
      subgraphs: {
        a: definition,
        b: {
          ...definition,
          id: "b",
          name: "B",
          nodes: [
            {
              ...definition.nodes[0],
              id: "a-instance",
              subgraphId: "a",
            },
          ],
        },
      },
    };
    expect(() => parseWorkflowJson(JSON.stringify(cyclic))).toThrow(
      "subgraph cycle",
    );

    expect(() =>
      parseWorkflowJson(
        JSON.stringify({
          ...base,
          nodes: [],
          subgraphs: {
            a: {
              ...definition,
              id: "other",
            },
          },
        }),
      ),
    ).toThrow("invalid id");
  });

  it("rejects invalid frame relationships and edges", () => {
    const frame = makeFrame("frame");
    const child = makeNode("child");
    const other = makeNode("other");

    const missingParent = serialiseWorkflow([child], []);
    missingParent.nodes[0]!.parentId = "missing";
    expect(() => parseWorkflowJson(JSON.stringify(missingParent))).toThrow(
      "missing parent",
    );

    const nonFrameParent = serialiseWorkflow([child, other], []);
    nonFrameParent.nodes[0]!.parentId = "other";
    expect(() => parseWorkflowJson(JSON.stringify(nonFrameParent))).toThrow(
      "parented to a frame",
    );

    const frameEdge = serialiseWorkflow([frame, child], []);
    frameEdge.edges = [
      {
        id: "frame-edge",
        source: "frame",
        sourceHandle: "out",
        target: "child",
        targetHandle: "df",
      },
    ];
    expect(() => parseWorkflowJson(JSON.stringify(frameEdge))).toThrow(
      "edges cannot reference frame nodes",
    );
  });

  it("rejects malformed documents", () => {
    expect(() => parseWorkflowJson("{")).toThrow("not valid JSON");
    expect(() =>
      parseWorkflowJson(JSON.stringify({ version: 99, nodes: [], edges: [] })),
    ).toThrow("Unsupported workflow version");
    expect(() => parseWorkflowJson(JSON.stringify({ version: 1 }))).toThrow(
      "nodes/edges",
    );
    expect(() =>
      parseWorkflowJson(
        JSON.stringify({ version: 1, nodes: [{ id: "a" }], edges: [] }),
      ),
    ).toThrow("id/code/position");
    expect(() =>
      parseWorkflowJson(
        JSON.stringify({
          version: 1,
          nodes: [],
          edges: [{ source: "a" }],
        }),
      ),
    ).toThrow("endpoints/handles");
  });

  it("rejects duplicate ids and oversized documents", () => {
    const node = {
      id: "a",
      label: "A",
      position: { x: 0, y: 0 },
      code: "x = 1\n",
      widgetValues: {},
      templateName: null,
    };
    expect(() =>
      parseWorkflowJson(
        JSON.stringify({
          version: 1,
          nodes: [node, { ...node }],
          edges: [],
        }),
      ),
    ).toThrow("duplicate node id");

    const edge = {
      id: "e1",
      source: "a",
      sourceHandle: "x",
      target: "b",
      targetHandle: "x",
    };
    expect(() =>
      parseWorkflowJson(
        JSON.stringify({
          version: 1,
          nodes: [node, { ...node, id: "b" }],
          edges: [edge, { ...edge }],
        }),
      ),
    ).toThrow("duplicate edge id");

    expect(() =>
      parseWorkflowJson(
        JSON.stringify({
          version: 1,
          nodes: [{ ...node, code: "x".repeat(MAX_NODE_CODE_CHARS + 1) }],
          edges: [],
        }),
      ),
    ).toThrow("character code limit");

    expect(() =>
      parseWorkflowJson(
        JSON.stringify({
          version: 1,
          nodes: Array.from({ length: MAX_WORKFLOW_NODES + 1 }, (_, index) => ({
            ...node,
            id: `n${index}`,
          })),
          edges: [],
        }),
      ),
    ).toThrow("too many nodes");

    expect(() =>
      parseWorkflowJson(
        JSON.stringify({
          version: 1,
          nodes: [node, { ...node, id: "b" }],
          edges: [{ ...edge, sourceHandle: "" }],
        }),
      ),
    ).toThrow("endpoints/handles must be");
  });
});

describe("project metadata", () => {
  it("round-trips the project name and local identity", () => {
    const saved = serialiseWorkflow([makeNode("a")], [], "Iris Demo");
    expect(saved.name).toBe("Iris Demo");
    expect(parseWorkflowJson(JSON.stringify(saved)).name).toBe("Iris Demo");

    saveLocalProjectIdentity({ name: "Iris Demo", slug: "iris-demo" });
    expect(loadLocalProjectIdentity()).toEqual({
      name: "Iris Demo",
      slug: "iris-demo",
    });

    clearLocalProjectIdentity();
    expect(loadLocalProjectIdentity()).toBeNull();
  });
});

describe("local workflow storage", () => {
  beforeEach(() => {
    clearLocalWorkflow();
    clearLocalProjectIdentity();
  });

  it("saves, loads and clears workflows", () => {
    expect(loadLocalWorkflow()).toBeNull();

    saveLocalWorkflow(serialiseWorkflow([makeNode("a")], []));

    const loaded = loadLocalWorkflow();
    expect(loaded?.nodes).toHaveLength(1);
    expect(loaded?.nodes[0]?.id).toBe("a");

    clearLocalWorkflow();
    expect(loadLocalWorkflow()).toBeNull();
  });

  it("discards corrupt autosave payloads", () => {
    localStorage.setItem(STORAGE_KEY, "{not json");

    expect(loadLocalWorkflow()).toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});

describe("local workspace storage", () => {
  beforeEach(() => {
    clearLocalWorkspace();
  });

  it("saves and restores multiple workflow windows", () => {
    saveLocalWorkspace({
      version: 1,
      activeWindowId: "two",
      windows: [
        {
          id: "one",
          title: "First",
          projectName: "First",
          projectSlug: null,
          workflow: serialiseWorkflow([makeNode("a")], [], "First"),
        },
        {
          id: "two",
          title: "Second",
          projectName: null,
          projectSlug: null,
          workflow: serialiseWorkflow([], []),
        },
      ],
    });

    const loaded = loadLocalWorkspace();
    expect(loaded?.activeWindowId).toBe("two");
    expect(loaded?.windows.map((window) => window.id)).toEqual([
      "one",
      "two",
    ]);
    expect(loaded?.windows[0]?.workflow.nodes).toHaveLength(1);
  });

  it("drops invalid windows and falls back to the first valid tab", () => {
    localStorage.setItem(
      WORKSPACE_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        activeWindowId: "missing",
        windows: [
          { id: "broken", workflow: { version: 1 } },
          {
            id: "valid",
            title: "Valid",
            projectName: null,
            projectSlug: null,
            workflow: serialiseWorkflow([], []),
          },
        ],
      }),
    );

    const loaded = loadLocalWorkspace();
    expect(loaded?.windows.map((window) => window.id)).toEqual(["valid"]);
    expect(loaded?.activeWindowId).toBe("valid");
  });

  it("discards corrupt workspace payloads", () => {
    localStorage.setItem(WORKSPACE_STORAGE_KEY, "{not json");

    expect(loadLocalWorkspace()).toBeNull();
    expect(localStorage.getItem(WORKSPACE_STORAGE_KEY)).toBeNull();
  });
});

describe("downloadTextFile", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("triggers a named blob download for exported artifacts", () => {
    vi.useFakeTimers();
    const createObjectURL = vi.fn(() => "blob:export");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    const clicked: HTMLAnchorElement[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
      function (this: HTMLAnchorElement) {
        clicked.push(this);
      },
    );

    downloadTextFile("workflow.py", "print('ok')", "text/x-python");

    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob));
    expect(clicked).toHaveLength(1);
    expect(clicked[0]?.download).toBe("workflow.py");
    expect(clicked[0]?.href).toContain("blob:export");
    expect(revokeObjectURL).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1000);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:export");
  });
});

describe("startWorkflowAutosave", () => {
  beforeEach(() => {
    clearLocalWorkflow();
    clearLocalProjectIdentity();
    useGraphStore.setState({ nodes: [], edges: [] });
  });

  it("debounces canvas changes into storage", () => {
    vi.useFakeTimers();
    const dispose = startWorkflowAutosave();
    try {
      useGraphStore.setState({ nodes: [makeNode("a")] });
      expect(loadLocalWorkflow()).toBeNull();

      vi.advanceTimersByTime(AUTOSAVE_DELAY_MS + 50);

      const saved = loadLocalWorkflow();
      expect(saved?.nodes).toHaveLength(1);
      expect(saved?.nodes[0]?.id).toBe("a");
    } finally {
      dispose();
      vi.useRealTimers();
    }
  });

  it("stopping the autosave prevents further writes", () => {
    vi.useFakeTimers();
    const dispose = startWorkflowAutosave();
    try {
      dispose();
      useGraphStore.setState({ nodes: [makeNode("b")] });
      vi.advanceTimersByTime(AUTOSAVE_DELAY_MS + 50);
      expect(loadLocalWorkflow()).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });
});
