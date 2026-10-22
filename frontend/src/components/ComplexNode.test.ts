// @vitest-environment jsdom
import { cleanup, fireEvent, render } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { createElement } from "react";
import { afterEach, expect, test, vi } from "vitest";

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

test("renders node progress while running", () => {
  useGraphStore.setState({
    execution: {
      status: "running",
      runId: "r-progress",
      order: ["n1"],
      nodeStates: {
        n1: {
          status: "running",
          progress: 0.5,
          progressMessage: "step 2/4",
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

  expect(container.textContent).toContain("step 2/4");
  expect(container.textContent).toContain("50%");
  const progressbar = container.querySelector('[role="progressbar"]');
  expect(progressbar?.getAttribute("aria-valuenow")).toBe("50");
});

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


test("does not bubble widget-stepper double-clicks to the node editor", () => {
  const valueData = {
    ...data,
    label: "Int Value",
    templateName: "int_value",
    templateCode: "value: int = 0\nstep: int = 1\nresult = value\nreturn result: INT",
    code: "value: int = 0\nstep: int = 1\nresult = value\nreturn result: INT",
    schema: {
      ...data.schema,
      widgets: [
        { name: "value", data_type: "int", default: 0, options: [] },
        { name: "step", data_type: "int", default: 1, options: [] },
      ],
      outputs: [{ name: "result", data_type: "INT" as const }],
    },
    widgetValues: { value: 0, step: 1 },
  };
  const onDoubleClick = vi.fn();
  const { getByLabelText } = render(
    createElement(
      "div",
      { onDoubleClick },
      createElement(
        ReactFlowProvider,
        null,
        createElement(ComplexNode, { ...nodeProps(), data: valueData }),
      ),
    ),
  );

  fireEvent.doubleClick(getByLabelText("Increase value by 1"));

  expect(onDoubleClick).not.toHaveBeenCalled();
});


test("renders inherited fixed choices for a connected value node", () => {
  const valueData = {
    ...data,
    label: "String Value",
    templateName: "string_value",
    templateCode: "value: str = 'text'",
    code: "value: str = 'text'",
    schema: {
      ...data.schema,
      widgets: [{ name: "value", data_type: "str", default: "text", options: [] }],
      outputs: [{ name: "result", data_type: "STRING" as const }],
    },
    widgetValues: { value: "low" },
  };
  const valueNode: AppNode = {
    id: "n1",
    type: "complex",
    position: { x: 0, y: 0 },
    data: valueData,
  };
  const target: AppNode = {
    id: "target",
    type: "complex",
    position: { x: 300, y: 0 },
    data: {
      ...data,
      label: "Target",
      schema: {
        ...data.schema,
        node_id: "target",
        widgets: [
          {
            name: "level",
            data_type: "str",
            default: "low",
            options: ["low", "medium", "high"],
          },
        ],
      },
    },
  };
  useGraphStore.setState({
    nodes: [valueNode, target],
    edges: [
      {
        id: "n1-result-target-level",
        source: "n1",
        target: "target",
        sourceHandle: "result",
        targetHandle: "level",
      },
    ],
  });

  const { getByRole } = render(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, { ...nodeProps(), data: valueData }),
    ),
  );
  const select = getByRole("combobox");
  expect(select.textContent).toContain("low");
  expect(select.textContent).toContain("medium");
  expect(select.textContent).toContain("high");
});
