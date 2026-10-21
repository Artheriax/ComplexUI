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
  color: "default",
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

test("renders the persisted node color accent", () => {
  const { container } = render(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, {
        ...nodeProps(),
        data: { ...data, color: "violet" },
      }),
    ),
  );

  const accent = container.querySelector('[data-node-color="violet"]');
  expect(accent).not.toBeNull();
  expect((accent as HTMLElement).style.backgroundColor).toBe(
    "rgb(167, 139, 250)",
  );
});

test("renders a shared subgraph manager without code controls", () => {
  const managerData = {
    ...data,
    label: "Inputs",
    subgraphRole: "input" as const,
    schema: {
      ...data.schema,
      node_id: "__subgraph_input__",
      outputs: [{ name: "df", data_type: "DATAFRAME" as const }],
    },
  };

  const { container } = render(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, { ...nodeProps(), data: managerData }),
    ),
  );

  expect(container.textContent).toContain("Inputs");
  expect(container.textContent).toContain("shared");
  expect(container.textContent).not.toContain("Edit code");
});

test("renders dataframe previews as a table inside the node", () => {
  useGraphStore.setState({
    execution: {
      status: "completed",
      runId: "r1",
      order: ["n1"],
      nodeStates: {
        n1: {
          status: "completed",
          outputs: {
            df: {
              rows: 150,
              cols: 5,
              head: [{ sepal_length: 5.1, species: "setosa" }],
            },
          },
        },
      },
      previews: {},
      richOutputs: {},
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

  expect(container.textContent).toContain("Preview");
  expect(container.textContent).toContain("df");
  expect(container.textContent).toContain("150 rows x 5 cols");
  expect(container.textContent).toContain("sepal_length");
  expect(container.textContent).toContain("setosa");
  expect(container.querySelector("table")).not.toBeNull();
});
