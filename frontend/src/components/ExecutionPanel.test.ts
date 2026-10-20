// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { AppNode, NodeSchema } from "../types/graph";
import { ExecutionPanel } from "./ExecutionPanel";

afterEach(cleanup);

const emptySchema: NodeSchema = {
  node_id: "n1",
  inputs: [],
  widgets: [],
  outputs: [],
  dependencies: [],
  syntax_errors: [],
};

function fakeNode(id: string, label: string): AppNode {
  return {
    id,
    type: "complex",
    position: { x: 0, y: 0 },
    data: {
      label,
      templateName: null,
      templateCode: null,
      code: "",
      schema: { ...emptySchema, node_id: id },
      widgetValues: {},
      isTrusted: false,
      color: "default",
    },
  };
}

test("renders the idle placeholder without crashing", () => {
  const { container } = render(createElement(ExecutionPanel));

  expect(container.textContent).toContain("Execution");
  expect(container.textContent).toContain("idle");
});

test("renders node labels, previews and cached badges after a run", () => {
  useGraphStore.setState({
    nodes: [fakeNode("n1", "Load Iris")],
    execution: {
      status: "completed",
      runId: "abcdef1234567890",
      order: ["n1"],
      nodeStates: { n1: { status: "completed", cached: true } },
      previews: {
        n1: { df: { rows: 150, cols: 5, head: [{ sepal_length: 5.1 }] } },
      },
      richOutputs: {
        n1: [
          {
            kind: "text",
            source: "print",
            text: "training complete",
            truncated: false,
          },
        ],
      },
      cached: { n1: true },
      error: null,
    },
  });

  const { container } = render(createElement(ExecutionPanel));

  expect(container.textContent).toContain("Load Iris");
  expect(container.textContent).toContain("150 rows x 5 cols");
  expect(container.textContent).toContain("training complete");
  expect(container.textContent).toContain("cached");
  expect(container.textContent).toContain("abcdef123456");
});

test("renders nodes that only produced console or rich output", () => {
  useGraphStore.setState({
    nodes: [fakeNode("n1", "Console Node")],
    execution: {
      status: "completed",
      runId: "r1",
      order: ["n1"],
      nodeStates: { n1: { status: "completed" } },
      previews: {},
      richOutputs: {
        n1: [
          {
            kind: "text",
            source: "print",
            text: "only console output",
            truncated: false,
          },
        ],
      },
      cached: {},
      error: null,
    },
  });

  const { container } = render(createElement(ExecutionPanel));

  expect(container.textContent).toContain("Console Node");
  expect(container.textContent).toContain("only console output");
  expect(container.textContent).toContain("print");
});

test("renders cancelled status hint", () => {
  useGraphStore.setState({
    execution: {
      status: "cancelled",
      runId: null,
      order: [],
      nodeStates: {},
      previews: {},
      richOutputs: {},
      cached: {},
      error: null,
    },
  });

  const { container } = render(createElement(ExecutionPanel));

  expect(container.textContent).toContain("cancelled");
  expect(container.textContent).toContain("stopped between nodes");
});
