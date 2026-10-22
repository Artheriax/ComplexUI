import { describe, expect, it } from "vitest";

import type { AppEdge, NodeSchema, WidgetParam } from "../types/graph";
import {
  mergeWidgetValues,
  reconcileEdges,
  widgetValueAllowed,
} from "./graphEditing";

const widget = (overrides: Partial<WidgetParam>): WidgetParam => ({
  name: "value",
  data_type: "float",
  default: 0.5,
  ...overrides,
});

const schema: NodeSchema = {
  node_id: "node",
  inputs: [{ name: "df", data_type: "DATAFRAME" }],
  widgets: [
    widget({ name: "count", data_type: "int", default: 2 }),
    widget({ name: "mode", data_type: "str", default: "fast", options: ["fast", "slow"] }),
  ],
  outputs: [{ name: "result", data_type: "ANY" }],
  dependencies: [],
  syntax_errors: [],
};

const edge = (overrides: Partial<AppEdge>): AppEdge => ({
  id: "edge",
  source: "node",
  sourceHandle: "result",
  target: "other",
  targetHandle: "in",
  ...overrides,
});

describe("graphEditing", () => {
  it("validates widget types and Literal choices", () => {
    expect(widgetValueAllowed(schema.widgets[1]!, "fast")).toBe(true);
    expect(widgetValueAllowed(schema.widgets[1]!, "other")).toBe(false);
    expect(widgetValueAllowed(widget({ data_type: "int" }), 2.5)).toBe(false);
    expect(widgetValueAllowed(widget({ data_type: "float" }), 2)).toBe(true);
    expect(widgetValueAllowed(widget({ data_type: "bool" }), true)).toBe(true);
  });

  it("keeps compatible saved values and resets invalid ones", () => {
    expect(
      mergeWidgetValues(schema, { count: 4, mode: "slow", removed: "x" }),
    ).toEqual({ count: 4, mode: "slow" });
    expect(
      mergeWidgetValues(schema, { count: "bad", mode: "other" }),
    ).toEqual({ count: 2, mode: "fast" });
  });

  it("drops edges that no longer match sockets or widgets", () => {
    const edges = [
      edge({ id: "keep", sourceHandle: "result", targetHandle: "df" }),
      edge({ id: "bad-source", sourceHandle: "missing" }),
      edge({ id: "bad-target", target: "node", targetHandle: "missing" }),
      edge({ id: "other", source: "other", target: "third" }),
    ];

    expect(reconcileEdges(edges, "node", schema).map((item) => item.id)).toEqual(
      ["keep", "other"],
    );
  });
});
