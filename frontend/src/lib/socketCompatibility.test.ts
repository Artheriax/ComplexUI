import { expect, test } from "vitest";

import {
  findCompatibleInput,
  findCompatibleOutput,
  isInputTaken,
  matchTemplateConnection,
  socketTypesCompatible,
} from "./socketCompatibility";
import type { AppEdge, NodeSchema, TemplateInfo } from "../types/graph";

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

function makeTemplate(schema: NodeSchema): TemplateInfo {
  return {
    name: "demo",
    title: "Demo",
    category: "Models",
    description: "Demo",
    hidden: false,
    code: "",
    code_hash: "",
    node_schema: schema,
  };
}

test("matches exact and ANY socket types", () => {
  expect(socketTypesCompatible("DATAFRAME", "DATAFRAME")).toBe(true);
  expect(socketTypesCompatible("ANY", "ML_MODEL")).toBe(true);
  expect(socketTypesCompatible("METRICS", "ANY")).toBe(true);
  expect(socketTypesCompatible("METRICS", "TENSOR")).toBe(false);
});

test("finds compatible free inputs and compatible outputs", () => {
  const schema = makeSchema({
    inputs: [
      { name: "df", data_type: "DATAFRAME" },
      { name: "rows", data_type: "ANY" },
    ],
    outputs: [
      { name: "metrics", data_type: "METRICS" },
      { name: "result", data_type: "ANY" },
    ],
  });

  expect(findCompatibleInput(schema, "DATASET")?.name).toBe("rows");
  expect(
    findCompatibleInput(schema, "DATASET", (name) => name === "rows"),
  ).toBeNull();
  expect(findCompatibleOutput(schema, "TENSOR")?.name).toBe("result");
});

test("matches templates for both connection directions", () => {
  const template = makeTemplate(
    makeSchema({
      inputs: [{ name: "df", data_type: "DATAFRAME" }],
      outputs: [{ name: "metrics", data_type: "METRICS" }],
    }),
  );

  expect(matchTemplateConnection(template, "source", "DATAFRAME")).toEqual({
    socketName: "df",
    socketType: "DATAFRAME",
  });
  expect(matchTemplateConnection(template, "target", "ANY")).toEqual({
    socketName: "metrics",
    socketType: "METRICS",
  });
  expect(matchTemplateConnection(template, "source", "TENSOR")).toBeNull();
});

test("detects occupied input handles", () => {
  const edges: AppEdge[] = [
    {
      id: "a:df->b:df",
      source: "a",
      sourceHandle: "df",
      target: "b",
      targetHandle: "df",
    },
  ];

  expect(isInputTaken(edges, "b", "df")).toBe(true);
  expect(isInputTaken(edges, "b", "other")).toBe(false);
  expect(isInputTaken(edges, "c", "df")).toBe(false);
});
