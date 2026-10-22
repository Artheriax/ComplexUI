import { describe, expect, it } from "vitest";

import type { AppNode, NodeSchema } from "../types/graph";
import { advanceSeedNodes } from "./seedNodes";

const schema: NodeSchema = {
  node_id: "",
  inputs: [],
  widgets: [],
  outputs: [],
  dependencies: [],
  syntax_errors: [],
};

function makeNode(widgetValues: Record<string, unknown>): AppNode {
  return {
    id: "node",
    type: "complex",
    position: { x: 0, y: 0 },
    data: {
      label: "Sampler",
      templateName: null,
      templateCode: null,
      code: "",
      schema,
      widgetValues,
      isTrusted: true,
      color: "default",
    },
  };
}

describe("advanceSeedNodes", () => {
  it("increments both legacy and generated seed controls", () => {
    const legacy = makeNode({
      seed: 7,
      control_after_generate: "increment",
    });
    const generated = makeNode({
      noise_seed: 7,
      control_after_generated: "increment",
    });
    const event = {
      type: "NODE_COMPLETED" as const,
      node_id: "node",
      data: { cached: false },
    };

    expect(advanceSeedNodes([legacy], event)[0]?.data.widgetValues.seed).toBe(
      8,
    );
    expect(
      advanceSeedNodes([generated], event)[0]?.data.widgetValues.noise_seed,
    ).toBe(8);
  });

  it("leaves nodes unchanged for cached completions or fixed controls", () => {
    const nodes = [
      makeNode({ seed: 7, control_after_generated: "fixed" }),
    ];
    const fixed = advanceSeedNodes(nodes, {
      type: "NODE_COMPLETED",
      node_id: "node",
      data: { cached: false },
    });
    const cached = advanceSeedNodes(nodes, {
      type: "NODE_COMPLETED",
      node_id: "node",
      data: { cached: true },
    });

    expect(fixed).toBe(nodes);
    expect(cached).toBe(nodes);
  });
});
