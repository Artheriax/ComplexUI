import { beforeEach, describe, expect, it, vi } from "vitest";

import { startWorkflowAutosave } from "../hooks/useWorkflowAutosave";
import { useGraphStore } from "../store/useGraphStore";
import type { AppEdge, AppNode, NodeSchema } from "../types/graph";
import {
  AUTOSAVE_DELAY_MS,
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
