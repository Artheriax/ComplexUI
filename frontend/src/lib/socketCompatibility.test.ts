import { expect, test } from "vitest";

import {
  findCompatibleInput,
  findCompatibleOutput,
  isInputTaken,
  matchTemplateConnection,
  resolveTargetSocket,
  socketTypesCompatible,
} from "./socketCompatibility";
import type { AppEdge, NodeSchema, TemplateInfo } from "../types/graph";
import { widgetSocketType } from "../types/graph";

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

test("maps widget python types onto socket types", () => {
  expect(widgetSocketType("int")).toBe("INT");
  expect(widgetSocketType("float")).toBe("FLOAT");
  expect(widgetSocketType("bool")).toBe("BOOL");
  expect(widgetSocketType("str")).toBe("STRING");
  expect(widgetSocketType("file")).toBe("STRING");
  expect(widgetSocketType("path")).toBe("STRING");
  expect(widgetSocketType("ImageFile")).toBe("STRING");
  expect(widgetSocketType("VideoFile")).toBe("STRING");
  expect(widgetSocketType("OutputFile")).toBe("STRING");
  expect(widgetSocketType("DATAFRAME")).toBe("DATAFRAME");
  expect(widgetSocketType("IMAGE")).toBe("IMAGE");
  expect(widgetSocketType("unknown")).toBe("ANY");
});

test("matches exact and ANY socket types", () => {
  expect(socketTypesCompatible("DATAFRAME", "DATAFRAME")).toBe(true);
  expect(socketTypesCompatible("ANY", "ML_MODEL")).toBe(true);
  expect(socketTypesCompatible("METRICS", "ANY")).toBe(true);
  expect(socketTypesCompatible("INT", "FLOAT")).toBe(true);
  expect(socketTypesCompatible("FLOAT", "INT")).toBe(false);
  expect(socketTypesCompatible("METRICS", "TENSOR")).toBe(false);
  expect(socketTypesCompatible("IMAGE", "MEDIA")).toBe(true);
  expect(socketTypesCompatible("VIDEO", "MEDIA")).toBe(true);
  expect(socketTypesCompatible("MEDIA", "IMAGE")).toBe(false);
  expect(socketTypesCompatible("IMAGE", "VIDEO")).toBe(false);
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
    targetKind: "input",
  });
  expect(matchTemplateConnection(template, "target", "ANY")).toEqual({
    socketName: "metrics",
    socketType: "METRICS",
    targetKind: "output",
  });
  expect(matchTemplateConnection(template, "source", "TENSOR")).toBeNull();
});

test("matches primitive outputs to widget parameters", () => {
  const template = makeTemplate(
    makeSchema({
      widgets: [
        { name: "test_size", data_type: "float", default: 0.2 },
        { name: "random_state", data_type: "int", default: 42 },
      ],
    }),
  );

  expect(matchTemplateConnection(template, "source", "FLOAT")).toEqual({
    socketName: "test_size",
    socketType: "FLOAT",
    targetKind: "widget",
  });
  expect(matchTemplateConnection(template, "source", "INT")).toEqual({
    socketName: "test_size",
    socketType: "FLOAT",
    targetKind: "widget",
  });
  expect(matchTemplateConnection(template, "source", "STRING")).toBeNull();

  const intTemplate = makeTemplate(
    makeSchema({
      widgets: [{ name: "random_state", data_type: "int", default: 42 }],
    }),
  );
  expect(matchTemplateConnection(intTemplate, "source", "INT")).toEqual({
    socketName: "random_state",
    socketType: "INT",
    targetKind: "widget",
  });
});

test("resolves input and widget handles to sockets", () => {
  const schema = makeSchema({
    inputs: [{ name: "df", data_type: "DATAFRAME" }],
    widgets: [{ name: "k", data_type: "int", default: 5 }],
  });

  expect(resolveTargetSocket(schema, "df")).toEqual({
    name: "df",
    data_type: "DATAFRAME",
  });
  expect(resolveTargetSocket(schema, "k")).toEqual({
    name: "k",
    data_type: "INT",
  });
  expect(resolveTargetSocket(schema, "missing")).toBeNull();
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
