// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { NodeSchema, TemplateInfo } from "../types/graph";
import { NodeSearchPopup } from "./NodeSearchPopup";

afterEach(cleanup);

function makeSchema(overrides: Partial<NodeSchema> = {}): NodeSchema {
  return {
    node_id: "",
    inputs: [],
    widgets: [],
    outputs: [],
    dependencies: [],
    syntax_errors: [],
    ...overrides,
  };
}

function makeTemplate(
  name: string,
  title: string,
  schema: NodeSchema,
): TemplateInfo {
  return {
    name,
    title,
    category: "Data",
    description: `${title} description`,
    hidden: false,
    code: "",
    code_hash: "",
    node_schema: schema,
  };
}

const producer = makeTemplate(
  "producer",
  "Producer",
  makeSchema({ outputs: [{ name: "data", data_type: "ANY" }] }),
);
const consumer = makeTemplate(
  "consumer",
  "Consumer",
  makeSchema({ inputs: [{ name: "value", data_type: "ANY" }] }),
);
const incompatible = makeTemplate(
  "incompatible",
  "Incompatible",
  makeSchema({ inputs: [{ name: "frame", data_type: "DATAFRAME" }] }),
);

beforeEach(() => {
  useGraphStore.setState({
    nodes: [],
    edges: [],
    templates: [producer, consumer, incompatible],
    templatesLoaded: true,
    templatesError: null,
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
});

test("searches templates and adds the chosen node", () => {
  const onClose = vi.fn();
  render(
    createElement(NodeSearchPopup, {
      anchor: { x: 40, y: 60 },
      flowPosition: { x: 120, y: 80 },
      connection: null,
      onClose,
    }),
  );

  const search = screen.getByRole("combobox", { name: "Search nodes" });
  expect(document.activeElement).toBe(search);
  expect(screen.getByRole("option", { name: /New scratch node/ })).toBeTruthy();

  fireEvent.change(search, { target: { value: "consumer" } });
  fireEvent.click(screen.getByRole("option", { name: /Consumer/ }));

  expect(onClose).toHaveBeenCalledOnce();
  const nodes = useGraphStore.getState().nodes;
  expect(nodes).toHaveLength(1);
  expect(nodes[0]?.data.templateName).toBe("consumer");
  expect(nodes[0]?.position).toEqual({ x: 120, y: 80 });
});

test("filters to compatible templates and connects the chosen node", () => {
  const sourceId = useGraphStore
    .getState()
    .addNodeFromTemplate(producer, { x: 0, y: 0 });
  const onClose = vi.fn();
  render(
    createElement(NodeSearchPopup, {
      anchor: { x: 40, y: 60 },
      flowPosition: { x: 240, y: 0 },
      connection: {
        nodeId: sourceId,
        handleId: "data",
        handleType: "source",
        dataType: "ANY",
      },
      onClose,
    }),
  );

  fireEvent.change(screen.getByRole("combobox", { name: "Search nodes" }), {
    target: { value: "consumer" },
  });
  expect(screen.queryByRole("option", { name: /Incompatible/ })).toBeNull();
  expect(
    screen.getByText("Connect to value · Any"),
  ).toBeTruthy();

  fireEvent.click(screen.getByRole("option", { name: /Consumer/ }));

  const state = useGraphStore.getState();
  expect(state.edges).toHaveLength(1);
  expect(state.edges[0]?.source).toBe(sourceId);
  expect(state.edges[0]?.sourceHandle).toBe("data");
  expect(state.edges[0]?.targetHandle).toBe("value");
  expect(onClose).toHaveBeenCalledOnce();
});

test("can show all nodes when no template is compatible", () => {
  useGraphStore.setState({ templates: [producer, incompatible] });
  const sourceId = useGraphStore
    .getState()
    .addNodeFromTemplate(producer, { x: 0, y: 0 });
  const onClose = vi.fn();
  render(
    createElement(NodeSearchPopup, {
      anchor: { x: 40, y: 60 },
      flowPosition: { x: 240, y: 0 },
      connection: {
        nodeId: sourceId,
        handleId: "data",
        handleType: "source",
        dataType: "TENSOR",
      },
      onClose,
    }),
  );

  expect(screen.getByText("No compatible nodes found.")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Show all nodes" }));
  fireEvent.change(screen.getByRole("combobox", { name: "Search nodes" }), {
    target: { value: "incompatible" },
  });

  expect(
    screen.getByText("No connection will be created"),
  ).toBeTruthy();
  fireEvent.click(screen.getByRole("option", { name: /Incompatible/ }));

  expect(useGraphStore.getState().nodes).toHaveLength(2);
  expect(useGraphStore.getState().edges).toHaveLength(0);
  expect(onClose).toHaveBeenCalledOnce();
});

test("supports keyboard selection and escape dismissal", () => {
  const onClose = vi.fn();
  render(
    createElement(NodeSearchPopup, {
      anchor: { x: 40, y: 60 },
      flowPosition: { x: 120, y: 80 },
      connection: null,
      onClose,
    }),
  );

  const search = screen.getByRole("combobox", { name: "Search nodes" });
  fireEvent.change(search, { target: { value: "producer" } });
  fireEvent.keyDown(search, { key: "Enter" });

  expect(useGraphStore.getState().nodes[0]?.data.templateName).toBe("producer");
  expect(onClose).toHaveBeenCalledOnce();

  fireEvent.keyDown(document, { key: "Escape" });
  expect(onClose).toHaveBeenCalledTimes(2);
});
