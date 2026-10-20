// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";

import { startWorkflowAutosave } from "../hooks/useWorkflowAutosave";
import { useGraphStore } from "../store/useGraphStore";
import type { AppEdge, AppNode, NodeSchema } from "../types/graph";
import {
  AUTOSAVE_DELAY_MS,
  MAX_NODE_CODE_CHARS,
  MAX_WORKFLOW_NODES,
  STORAGE_KEY,
  clearLocalWorkflow,
  loadLocalWorkflow,
  parseWorkflowJson,
  saveLocalWorkflow,
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
    const saved = serialiseWorkflow([makeNode("a")], [makeEdge()]);
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

describe("local workflow storage", () => {
  beforeEach(() => {
    clearLocalWorkflow();
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

describe("startWorkflowAutosave", () => {
  beforeEach(() => {
    clearLocalWorkflow();
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
