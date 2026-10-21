import { expect, test } from "vitest";

import type {
  AppEdge,
  AppNode,
  InputSocket,
  NodeSchema,
  OutputSocket,
  SubgraphDefinition,
  SubgraphMap,
} from "../types/graph";
import {
  SUBGRAPH_INPUT_NODE_ID,
  SUBGRAPH_OUTPUT_NODE_ID,
} from "../types/graph";
import {
  buildSubgraphFromSelection,
  createSubgraphInstanceNode,
  flattenWorkflow,
  SubgraphError,
} from "./subgraph";

function schema(
  nodeId: string,
  inputs: InputSocket[] = [],
  outputs: OutputSocket[] = [],
): NodeSchema {
  return {
    node_id: nodeId,
    inputs,
    outputs,
    widgets: [],
    dependencies: [],
    syntax_errors: [],
  };
}

function makeNode(
  id: string,
  inputs: InputSocket[] = [],
  outputs: OutputSocket[] = [],
  x = 0,
  y = 0,
): AppNode {
  return {
    id,
    type: "complex",
    position: { x, y },
    measured: { width: 200, height: 100 },
    data: {
      label: id.toUpperCase(),
      templateName: null,
      templateCode: null,
      code: `value = 1\nreturn value: ANY`,
      schema: schema(id, inputs, outputs),
      widgetValues: {},
      isTrusted: true,
      color: "default",
    },
  };
}

function edge(
  source: string,
  sourceHandle: string,
  target: string,
  targetHandle: string,
): AppEdge {
  return {
    id: `${source}:${sourceHandle}->${target}:${targetHandle}`,
    source,
    sourceHandle,
    target,
    targetHandle,
  };
}

const dfIn: InputSocket = { name: "df", data_type: "DATAFRAME" };
const anyOut: OutputSocket = { name: "result", data_type: "ANY" };
const anyIn: InputSocket = { name: "result", data_type: "ANY" };

test("builds a subgraph from boundary edges and flattens it back", () => {
  const source = makeNode("src", [], [anyOut]);
  const selected = makeNode("work", [dfIn], [anyOut], 100, 50);
  const sink = makeNode("sink", [anyIn], []);
  const edges = [
    edge("src", "result", "work", "df"),
    edge("work", "result", "sink", "result"),
  ];

  const built = buildSubgraphFromSelection({
    definitionId: "def-1",
    instanceId: "instance-1",
    name: "Subgraph 1",
    nodes: [source, selected, sink],
    edges,
    selectedIds: new Set(["work"]),
  });

  expect(built.instance.data.subgraphId).toBe("def-1");
  expect(built.instance.data.schema.inputs).toEqual([
    { name: "df", data_type: "DATAFRAME" },
  ]);
  expect(built.instance.data.schema.outputs).toEqual([
    { name: "result", data_type: "ANY" },
  ]);
  expect(built.nodes.map((node) => node.id)).toEqual([
    "src",
    "sink",
    "instance-1",
  ]);
  expect(built.edges.map((current) => current.id)).toEqual([
    "src:result->instance-1:df",
    "instance-1:result->sink:result",
  ]);
  expect(built.definition.nodes[0]?.position).toEqual({ x: 0, y: 0 });
  expect(
    built.definition.edges.some(
      (current) => current.source === SUBGRAPH_INPUT_NODE_ID,
    ),
  ).toBe(true);
  expect(
    built.definition.edges.some(
      (current) => current.target === SUBGRAPH_OUTPUT_NODE_ID,
    ),
  ).toBe(true);

  const flattened = flattenWorkflow(
    built.nodes,
    built.edges,
    { "def-1": built.definition },
  );
  expect(flattened.graph.nodes.map((node) => node.id).sort()).toEqual([
    "sink",
    "src",
    "sub:instance-1/work",
  ]);
  expect(flattened.labels["sub:instance-1/work"]).toBe("WORK");
  expect(
    flattened.graph.nodes.find(
      (node) => node.id === "sub:instance-1/work",
    )?.approval_id,
  ).toBe("work");
  expect(flattened.graph.edges).toEqual(
    expect.arrayContaining([
      {
        source_node_id: "src",
        source_handle: "result",
        target_node_id: "sub:instance-1/work",
        target_handle: "df",
      },
      {
        source_node_id: "sub:instance-1/work",
        source_handle: "result",
        target_node_id: "sink",
        target_handle: "result",
      },
    ]),
  );
});

test("creates typed boundary ports for widget connections", () => {
  const source = makeNode(
    "src",
    [],
    [{ name: "result", data_type: "FLOAT" }],
  );
  const selected = makeNode("work", [dfIn], [anyOut]);
  selected.data.schema.widgets = [
    { name: "test_size", data_type: "float", default: 0.2 },
  ];
  selected.data.widgetValues = { test_size: 0.2 };

  const built = buildSubgraphFromSelection({
    definitionId: "def-widget",
    instanceId: "instance-widget",
    name: "Widget Subgraph",
    nodes: [source, selected],
    edges: [edge("src", "result", "work", "test_size")],
    selectedIds: new Set(["work"]),
  });

  expect(built.instance.data.schema.inputs).toEqual([
    { name: "test_size", data_type: "FLOAT" },
  ]);
  expect(built.edges.map((current) => current.id)).toEqual([
    "src:result->instance-widget:test_size",
  ]);

  const flattened = flattenWorkflow(
    built.nodes,
    built.edges,
    { "def-widget": built.definition },
  );
  expect(flattened.graph.edges).toEqual([
    {
      source_node_id: "src",
      source_handle: "result",
      target_node_id: "sub:instance-widget/work",
      target_handle: "test_size",
    },
  ]);
});

test("flattens nested instances with path-scoped ids", () => {
  const inner: SubgraphDefinition = {
    id: "inner",
    name: "Inner",
    inputs: [{ name: "in", data_type: "DATAFRAME" }],
    outputs: [{ name: "out", data_type: "ANY" }],
    nodes: [
      makeNode(
        "leaf",
        [{ name: "in", data_type: "DATAFRAME" }],
        [{ name: "out", data_type: "ANY" }],
      ),
    ],
    edges: [
      edge(SUBGRAPH_INPUT_NODE_ID, "in", "leaf", "in"),
      edge("leaf", "out", SUBGRAPH_OUTPUT_NODE_ID, "out"),
    ],
    inputPosition: { x: -200, y: 0 },
    outputPosition: { x: 300, y: 0 },
  };
  const innerInstance = createSubgraphInstanceNode(
    "inner-instance",
    inner,
    { x: 0, y: 0 },
  );
  const outer: SubgraphDefinition = {
    id: "outer",
    name: "Outer",
    inputs: [{ name: "in", data_type: "DATAFRAME" }],
    outputs: [{ name: "out", data_type: "ANY" }],
    nodes: [innerInstance],
    edges: [
      edge(SUBGRAPH_INPUT_NODE_ID, "in", "inner-instance", "in"),
      edge("inner-instance", "out", SUBGRAPH_OUTPUT_NODE_ID, "out"),
    ],
    inputPosition: { x: -200, y: 0 },
    outputPosition: { x: 300, y: 0 },
  };
  const outerInstance = createSubgraphInstanceNode(
    "outer-instance",
    outer,
    { x: 100, y: 100 },
  );
  const source = makeNode("src", [], [anyOut]);
  const sink = makeNode("sink", [anyIn], []);
  const subgraphs: SubgraphMap = { inner, outer };

  const flattened = flattenWorkflow(
    [source, outerInstance, sink],
    [
      edge("src", "result", "outer-instance", "in"),
      edge("outer-instance", "out", "sink", "result"),
    ],
    subgraphs,
  );

  expect(flattened.graph.nodes.map((node) => node.id).sort()).toEqual([
    "sink",
    "src",
    "sub:outer-instance/inner-instance/leaf",
  ]);
  expect(flattened.graph.edges).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        source_node_id: "src",
        target_node_id: "sub:outer-instance/inner-instance/leaf",
      }),
      expect.objectContaining({
        source_node_id: "sub:outer-instance/inner-instance/leaf",
        target_node_id: "sink",
      }),
    ]),
  );
});

test("rejects recursive subgraph cycles", () => {
  const a: SubgraphDefinition = {
    id: "a",
    name: "A",
    inputs: [],
    outputs: [],
    nodes: [],
    edges: [],
    inputPosition: { x: 0, y: 0 },
    outputPosition: { x: 0, y: 0 },
  };
  const b: SubgraphDefinition = {
    ...a,
    id: "b",
    name: "B",
  };
  a.nodes = [createSubgraphInstanceNode("b-instance", b, { x: 0, y: 0 })];
  b.nodes = [createSubgraphInstanceNode("a-instance", a, { x: 0, y: 0 })];

  expect(() =>
    flattenWorkflow(
      [createSubgraphInstanceNode("a-instance", a, { x: 0, y: 0 })],
      [],
      { a, b },
    ),
  ).toThrow(SubgraphError);
});
