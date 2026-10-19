// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { NodeSchema, TemplateInfo } from "../types/graph";
import { NodePalette } from "./NodePalette";

afterEach(cleanup);

const schema: NodeSchema = {
  node_id: "demo",
  inputs: [],
  widgets: [],
  outputs: [{ name: "k", data_type: "ANY" }],
  dependencies: [],
  syntax_errors: [],
};

const demoTemplate: TemplateInfo = {
  name: "demo",
  title: "Demo",
  category: "Preprocessing",
  description: "Demo template",
  hidden: false,
  code: "# demo\nreturn k: ANY\n",
  code_hash: "hash",
  node_schema: schema,
};

beforeEach(() => {
  useGraphStore.setState({
    templates: [demoTemplate],
    templatesLoaded: true,
    templatesError: null,
    workflows: [],
    workflowsLoaded: true,
    workflowsError: null,
  });
});

test("shows the node library by default and switches to workflows", () => {
  render(createElement(NodePalette));

  expect(screen.getByPlaceholderText("Search templates…")).toBeTruthy();
  expect(screen.queryByPlaceholderText("Search workflows…")).toBeNull();

  fireEvent.click(screen.getByText("Workflows"));

  expect(screen.getByPlaceholderText("Search workflows…")).toBeTruthy();
  expect(screen.queryByPlaceholderText("Search templates…")).toBeNull();
  expect(
    screen.getByText(/Loading a workflow replaces the canvas/),
  ).toBeTruthy();

  fireEvent.click(screen.getByText("Nodes"));

  expect(screen.getByPlaceholderText("Search templates…")).toBeTruthy();
});
