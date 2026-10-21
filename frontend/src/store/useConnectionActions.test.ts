import { beforeEach, expect, test } from "vitest";

import type { AppEdge, AppNode } from "../types/graph";
import { useGraphStore } from "./useGraphStore";

function node(id: string): AppNode {
  return {
    id,
    type: "complex",
    position: { x: 0, y: 0 },
    data: {
      label: id,
      templateName: null,
      templateCode: null,
      code: "",
      schema: {
        node_id: id,
        inputs: [{ name: "input", data_type: "IMAGE" }],
        widgets: [],
        outputs: [{ name: "image", data_type: "IMAGE" }],
        dependencies: [],
        syntax_errors: [],
      },
      widgetValues: {},
      isTrusted: true,
      color: "default",
    },
  };
}

function edge(source: string, target: string, targetHandle = "input"): AppEdge {
  return {
    id: `${source}:image->${target}:${targetHandle}`,
    source,
    target,
    sourceHandle: "image",
    targetHandle,
  };
}

beforeEach(() => {
  useGraphStore.setState({
    nodes: [
      node("source"),
      node("new-source"),
      node("target"),
      node("target-2"),
      node("free"),
    ],
    edges: [edge("source", "target"), edge("source", "target-2")],
    past: [],
    future: [],
  });
});

test("retargeting an edge is one undo step", () => {
  const original = useGraphStore.getState().edges;
  const retargeted = useGraphStore
    .getState()
    .reconnectEdgeTarget("source:image->target:input", "free", "input");

  expect(retargeted).toBe(true);
  expect(useGraphStore.getState().edges).toContainEqual({
    id: "source:image->free:input",
    source: "source",
    sourceHandle: "image",
    target: "free",
    targetHandle: "input",
  });

  useGraphStore.getState().undo();
  expect(useGraphStore.getState().edges).toEqual(original);
});

test("rejects retargeting to an occupied input", () => {
  expect(
    useGraphStore
      .getState()
      .reconnectEdgeTarget("source:image->target:input", "target-2", "input"),
  ).toBe(false);
});

test("reassigns an output bundle to a new source", () => {
  const ids = useGraphStore.getState().edges.map((candidate) => candidate.id);
  const reassigned = useGraphStore
    .getState()
    .reassignOutputBundle(ids, "new-source", "image");

  expect(reassigned).toBe(true);
  expect(
    useGraphStore.getState().edges.map((candidate) => ({
      source: candidate.source,
      target: candidate.target,
      sourceHandle: candidate.sourceHandle,
    })),
  ).toEqual([
    { source: "new-source", target: "target", sourceHandle: "image" },
    { source: "new-source", target: "target-2", sourceHandle: "image" },
  ]);
});

test("rejects a bundle when any captured edge is missing", () => {
  const ids = useGraphStore.getState().edges.map((candidate) => candidate.id);

  expect(
    useGraphStore
      .getState()
      .reassignOutputBundle([...ids, "missing"], "new-source", "image"),
  ).toBe(false);
});
