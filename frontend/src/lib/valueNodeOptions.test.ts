import { expect, test } from "vitest";

import type { AppEdge, AppNode, ComplexNodeData } from "../types/graph";
import {
  deriveValueNodeOptions,
  normalizeValueChoice,
  normalizeValueNodeValues,
} from "./valueNodeOptions";

function makeNode(
  id: string,
  data: Partial<ComplexNodeData> = {},
): AppNode {
  return {
    id,
    type: "complex",
    position: { x: 0, y: 0 },
    data: {
      label: id,
      templateName: id,
      templateCode: "official",
      code: "official",
      schema: {
        node_id: id,
        inputs: [],
        widgets: [],
        outputs: [],
        dependencies: [],
        syntax_errors: [],
      },
      widgetValues: {},
      isTrusted: true,
      color: "default",
      ...data,
    },
  };
}

function edge(
  source: string,
  target: string,
  targetHandle: string,
): AppEdge {
  return {
    id: `${source}-${target}-${targetHandle}`,
    source,
    target,
    sourceHandle: "result",
    targetHandle,
  };
}

test("inherits fixed choices from a connected widget", () => {
  const value = makeNode("string_value", {
    templateName: "string_value",
    templateCode: "value: str = 'text'",
    code: "value: str = 'text'",
    schema: {
      node_id: "string_value",
      inputs: [],
      widgets: [{ name: "value", data_type: "str", default: "text" }],
      outputs: [{ name: "result", data_type: "STRING" }],
      dependencies: [],
      syntax_errors: [],
    },
    widgetValues: { value: "text" },
  });
  const target = makeNode("filter", {
    schema: {
      node_id: "filter",
      inputs: [],
      widgets: [
        {
          name: "level",
          data_type: "str",
          default: "low",
          options: ["low", "medium", "high"],
        },
      ],
      outputs: [],
      dependencies: [],
      syntax_errors: [],
    },
  });
  const edges = [edge(value.id, target.id, "level")];

  expect(deriveValueNodeOptions(value.id, value.data, [value, target], edges)).toEqual([
    "low",
    "medium",
    "high",
  ]);
  expect(normalizeValueChoice("text", ["low", "medium", "high"])).toBe("low");
  expect(
    normalizeValueNodeValues([value, target], edges)[0].data.widgetValues.value,
  ).toBe("low");
});

test("intersects choices across multiple constrained targets", () => {
  const value = makeNode("string_value");
  const first = makeNode("first", {
    schema: {
      node_id: "first",
      inputs: [],
      widgets: [
        {
          name: "level",
          data_type: "str",
          default: "low",
          options: ["low", "medium"],
        },
      ],
      outputs: [],
      dependencies: [],
      syntax_errors: [],
    },
  });
  const second = makeNode("second", {
    schema: {
      node_id: "second",
      inputs: [],
      widgets: [
        {
          name: "level",
          data_type: "str",
          default: "medium",
          options: ["medium", "high"],
        },
      ],
      outputs: [],
      dependencies: [],
      syntax_errors: [],
    },
  });

  expect(
    deriveValueNodeOptions(value.id, value.data, [value, first, second], [
      edge(value.id, first.id, "level"),
      edge(value.id, second.id, "level"),
    ]),
  ).toEqual(["medium"]);
});

test("edited value nodes do not inherit choices", () => {
  const value = makeNode("string_value", { code: "custom code" });
  const target = makeNode("target", {
    schema: {
      node_id: "target",
      inputs: [],
      widgets: [
        {
          name: "level",
          data_type: "str",
          default: "low",
          options: ["low"],
        },
      ],
      outputs: [],
      dependencies: [],
      syntax_errors: [],
    },
  });

  expect(
    deriveValueNodeOptions(value.id, value.data, [value, target], [
      edge(value.id, target.id, "level"),
    ]),
  ).toBeNull();
});
