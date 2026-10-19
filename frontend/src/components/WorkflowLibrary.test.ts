// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { NodeSchema, TemplateInfo, WorkflowTemplateInfo } from "../types/graph";
import { WorkflowLibrary } from "./WorkflowLibrary";

afterEach(cleanup);

const schema: NodeSchema = {
  node_id: "demo",
  inputs: [{ name: "df", data_type: "DATAFRAME" }],
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

function makeWorkflow(
  name: string,
  title: string,
  category: string,
  nodeCount: number,
  description = `${title} description`,
): WorkflowTemplateInfo {
  return {
    name,
    title,
    category,
    description,
    nodes: Array.from({ length: nodeCount }, (_, index) => ({
      id: `n${index}`,
      template: "demo",
      position: { x: index * 200, y: 0 },
      widget_values: {},
    })),
    edges: [],
  };
}

const workflows = [
  makeWorkflow("iris_flow", "Iris Flow", "Getting Started", 3),
  makeWorkflow("wine_flow", "Wine Flow", "Classical ML", 4),
  makeWorkflow("messy_flow", "Messy Flow", "Data Quality & EDA", 2),
];

beforeEach(() => {
  useGraphStore.setState({
    templates: [demoTemplate],
    templatesLoaded: true,
    templatesError: null,
    workflows,
    workflowsLoaded: true,
    workflowsError: null,
    nodes: [],
    edges: [],
  });
});

test("groups workflows by category in catalog order", () => {
  const { container } = render(createElement(WorkflowLibrary));

  const text = container.textContent ?? "";
  const gettingStarted = text.indexOf("Getting Started");
  const classical = text.indexOf("Classical ML");
  const quality = text.indexOf("Data Quality & EDA");
  expect(gettingStarted).toBeGreaterThanOrEqual(0);
  expect(gettingStarted).toBeLessThan(classical);
  expect(classical).toBeLessThan(quality);
  expect(screen.getByText("Iris Flow")).toBeTruthy();
  expect(screen.getByText("3 nodes")).toBeTruthy();
});

test("clicking a workflow loads it onto the canvas", () => {
  useGraphStore.setState({
    nodes: [
      {
        id: "old",
        type: "complex",
        position: { x: 0, y: 0 },
        data: {
          label: "Old",
          templateName: null,
          templateCode: null,
          code: "",
          schema,
          widgetValues: {},
          isTrusted: false,
        },
      },
    ],
  });

  render(createElement(WorkflowLibrary));
  fireEvent.click(screen.getByText("Wine Flow"));

  const state = useGraphStore.getState();
  expect(state.nodes.map((node) => node.id)).toEqual(["n0", "n1", "n2", "n3"]);
  expect(state.execution.status).toBe("idle");
});

test("search filters workflow cards", () => {
  render(createElement(WorkflowLibrary));

  fireEvent.change(screen.getByPlaceholderText("Search workflows…"), {
    target: { value: "messy" },
  });

  expect(screen.getByText("Messy Flow")).toBeTruthy();
  expect(screen.queryByText("Iris Flow")).toBeNull();
  expect(screen.queryByText("Getting Started")).toBeNull();
});

test("shows a loading hint before workflows arrive", () => {
  useGraphStore.setState({ workflows: [], workflowsLoaded: false });
  render(createElement(WorkflowLibrary));

  expect(screen.getByText(/Loading workflows/)).toBeTruthy();
});

test("surfaces workflow load errors", () => {
  useGraphStore.setState({ workflowsError: "Backend unreachable" });
  render(createElement(WorkflowLibrary));

  expect(screen.getByText("Backend unreachable")).toBeTruthy();
});
