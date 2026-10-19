// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { createElement } from "react";
import { afterEach, expect, test } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { AppNode } from "../types/graph";
import { ComplexNode } from "./ComplexNode";

afterEach(cleanup);

const data: AppNode["data"] = {
  label: "Console Node",
  templateName: null,
  templateCode: null,
  code: "",
  schema: {
    node_id: "n1",
    inputs: [],
    widgets: [],
    outputs: [],
    dependencies: [],
    syntax_errors: [],
  },
  widgetValues: {},
  isTrusted: true,
};

function nodeProps(): NodeProps<AppNode> {
  return {
    id: "n1",
    type: "complex",
    data,
    selected: false,
    isConnectable: true,
    zIndex: 0,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
    dragging: false,
  } as unknown as NodeProps<AppNode>;
}

test("renders inline rich outputs from the execution state", () => {
  useGraphStore.setState({
    execution: {
      status: "completed",
      runId: "r1",
      order: ["n1"],
      nodeStates: {
        n1: {
          status: "completed",
          richOutputs: [
            {
              kind: "text",
              source: "print",
              text: "node console line",
              truncated: false,
            },
          ],
        },
      },
      previews: {},
      richOutputs: {
        n1: [
          {
            kind: "text",
            source: "print",
            text: "node console line",
            truncated: false,
          },
        ],
      },
      cached: {},
      error: null,
    },
  });

  const { container } = render(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, nodeProps()),
    ),
  );

  expect(container.textContent).toContain("Console Node");
  expect(container.textContent).toContain("Output");
  expect(container.textContent).toContain("node console line");
});
