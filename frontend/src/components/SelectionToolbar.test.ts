// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { AppNode, NodeSchema } from "../types/graph";
import { SelectionToolbar } from "./SelectionToolbar";

afterEach(cleanup);

const schema: NodeSchema = {
  node_id: "demo",
  inputs: [],
  widgets: [],
  outputs: [],
  dependencies: [],
  syntax_errors: [],
};

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
    measured: { width: 100, height: 50 },
    data: {
      label: id,
      templateName: null,
      templateCode: null,
      code: "",
      schema,
      widgetValues: {},
      isTrusted: false,
      color: "default",
    },
  };
}

beforeEach(() => {
  useGraphStore.setState({
    nodes: [
      makeNode("a", 10, 20),
      makeNode("b", 50, 80),
      makeNode("c", 120, 40),
    ],
    past: [],
    future: [],
  });
});

test("aligns the current selection and clears it", () => {
  render(createElement(SelectionToolbar));

  expect(screen.getByText("3 selected")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Align left" }));

  expect(
    useGraphStore.getState().nodes.map((node) => node.position.x),
  ).toEqual([10, 10, 10]);

  fireEvent.click(screen.getByRole("button", { name: "Clear selection" }));
  expect(screen.queryByRole("toolbar")).toBeNull();
});

test("groups the selection and ungroups the resulting frame", () => {
  render(createElement(SelectionToolbar));

  fireEvent.click(screen.getByRole("button", { name: "Group selection" }));

  const frame = useGraphStore
    .getState()
    .nodes.find((node) => node.data.isFrame);
  expect(frame).toBeDefined();
  expect(screen.getByText("1 selected")).toBeTruthy();
  expect(
    screen.getByRole("button", { name: "Align left" }),
  ).toHaveProperty("disabled", true);

  fireEvent.click(screen.getByRole("button", { name: "Ungroup selection" }));

  expect(
    useGraphStore.getState().nodes.some((node) => node.data.isFrame),
  ).toBe(false);
  expect(
    useGraphStore.getState().nodes.map((node) => node.position.x),
  ).toEqual([10, 50, 120]);
});

test("creates a reusable subgraph from the selection", () => {
  render(createElement(SelectionToolbar));

  fireEvent.click(
    screen.getByRole("button", { name: "Create subgraph" }),
  );

  const state = useGraphStore.getState();
  expect(Object.keys(state.subgraphs)).toHaveLength(1);
  expect(state.nodes).toHaveLength(1);
  expect(state.nodes[0]?.data.isSubgraph).toBe(true);
  expect(state.nodes[0]?.data.schema.inputs).toHaveLength(0);
  expect(state.nodes[0]?.data.schema.outputs).toHaveLength(0);
});

test("enables distribution only for three or more selected nodes", () => {
  useGraphStore.setState({
    nodes: [makeNode("a", 10, 20), makeNode("b", 50, 80)],
  });
  render(createElement(SelectionToolbar));

  expect(screen.getByText("2 selected")).toBeTruthy();
  const distribute = screen.getByRole("button", {
    name: "Distribute horizontally",
  });
  expect(distribute).toHaveProperty("disabled", true);
  fireEvent.click(screen.getByRole("button", { name: "Arrange in a grid" }));

  expect(
    useGraphStore.getState().nodes.map((node) => node.position),
  ).toEqual([
    { x: 10, y: 20 },
    { x: 158, y: 20 },
  ]);
});
