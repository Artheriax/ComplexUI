// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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

test("tints the node body with the selected dark surface", () => {
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

  const surface = container.querySelector('[data-node-surface="violet"]');
  expect(surface).not.toBeNull();
  expect((surface as HTMLElement).style.backgroundColor).toBe(
    "rgb(28, 21, 51)",
  );
});

test("gives prompt widgets a flexible textarea row", () => {
  const promptData = {
    ...data,
    label: "CLIP Text Encode",
    templateName: "clip_text_encode",
    schema: {
      ...data.schema,
      widgets: [
        {
          name: "text",
          data_type: "Prompt",
          default: "",
        },
      ],
    },
    widgetValues: { text: "a cinematic landscape" },
  };
  const { container } = render(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, { ...nodeProps(), data: promptData }),
    ),
  );

  const prompt = container.querySelector("[data-cu-prompt]");
  expect(prompt?.tagName).toBe("TEXTAREA");
  expect(prompt?.className).toContain("flex-1");
  expect(
    container.querySelector('[data-prompt-row="true"]')?.className,
  ).toContain("flex-1");
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

test("renders the Save Media editor for the save_media schema", () => {
  useGraphStore.setState({
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
  const saveData: AppNode["data"] = {
    label: "Save Media",
    templateName: "save_media",
    templateCode: null,
    code: "",
    schema: {
      node_id: "save",
      inputs: [{ name: "media", data_type: "MEDIA" }],
      widgets: [
        { name: "folder", data_type: "str", default: "%date%" },
        { name: "filename", data_type: "str", default: "%model%_%counter%" },
        { name: "model_name", data_type: "str", default: "image" },
        {
          name: "format",
          data_type: "str",
          default: "png",
          options: ["png", "jpg", "webp"],
        },
        { name: "quality", data_type: "int", default: 95 },
        { name: "compress_level", data_type: "int", default: 6 },
        { name: "optimize", data_type: "bool", default: true },
        { name: "save_nonce", data_type: "int", default: 0 },
      ],
      outputs: [
        { name: "saved_path", data_type: "STRING" },
        { name: "saved", data_type: "MEDIA" },
      ],
      dependencies: [],
      syntax_errors: [],
    },
    widgetValues: {
      folder: "%date%",
      filename: "%model%_%counter%",
      model_name: "model",
      format: "png",
      quality: 95,
      compress_level: 6,
      optimize: true,
      save_nonce: 0,
    },
    isTrusted: true,
    color: "default",
  };

  render(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, {
        ...nodeProps(),
        id: "save",
        data: saveData,
      }),
    ),
  );

  expect(screen.getByLabelText("Save folder")).toBeTruthy();
  expect(screen.getByLabelText("Filename template")).toBeTruthy();
  expect(screen.getByText("Will save as")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Save" })).toBeTruthy();
});

test("renders the switch editor with named inputs", () => {
  useGraphStore.setState({
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
  const switchData: AppNode["data"] = {
    label: "Switch (5 Inputs)",
    templateName: "switch",
    templateCode: null,
    code: "",
    schema: {
      node_id: "switch",
      inputs: [],
      widgets: [
        { name: "input_1", data_type: "ANY", default: null },
        { name: "input_2", data_type: "ANY", default: null },
        { name: "input_3", data_type: "ANY", default: null },
        { name: "input_4", data_type: "ANY", default: null },
        { name: "input_5", data_type: "ANY", default: null },
        {
          name: "choices",
          data_type: "str",
          default: "Input 1|Input 2|Input 3|Input 4|Input 5",
        },
        { name: "selected", data_type: "int", default: 0 },
      ],
      outputs: [{ name: "chosen", data_type: "ANY" }],
      dependencies: [],
      syntax_errors: [],
    },
    widgetValues: {
      input_1: null,
      input_2: null,
      input_3: null,
      input_4: null,
      input_5: null,
      choices: "Alpha|Beta|Gamma|Delta|Epsilon",
      selected: 0,
    },
    isTrusted: true,
    color: "default",
  };

  render(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, {
        ...nodeProps(),
        id: "switch",
        data: switchData,
      }),
    ),
  );

  expect(screen.getByLabelText("Switch selection")).toBeTruthy();
  expect(screen.getByLabelText("Input 1 name")).toBeTruthy();
  expect(screen.getByLabelText("Input 5 name")).toBeTruthy();
  expect(screen.queryByText("model_name")).toBeNull();
});

test("renders only widgets whose visibility conditions match", () => {
  useGraphStore.setState({
    nodes: [],
    edges: [],
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
  const conditionalData: AppNode["data"] = {
    ...data,
    label: "Conditional Node",
    schema: {
      ...data.schema,
      widgets: [
        {
          name: "size_mode",
          data_type: "str",
          default: "Resolution",
          options: ["Resolution", "Megapixels"],
        },
        {
          name: "width",
          data_type: "int",
          default: 512,
          visible_when: [
            { widget: "size_mode", values: ["Resolution"] },
          ],
        },
        {
          name: "megapixel_preset",
          data_type: "str",
          default: "1.00 MP - 1024p",
          options: ["1.00 MP - 1024p", "Custom"],
          visible_when: [
            { widget: "size_mode", values: ["Megapixels"] },
          ],
        },
        {
          name: "custom_megapixels",
          data_type: "float",
          default: 1,
          visible_when: [
            { widget: "size_mode", values: ["Megapixels"] },
            { widget: "megapixel_preset", values: ["Custom"] },
          ],
        },
      ],
    },
    widgetValues: {
      size_mode: "Resolution",
      width: 512,
      megapixel_preset: "1.00 MP - 1024p",
      custom_megapixels: 1,
    },
  };
  const view = render(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, {
        ...nodeProps(),
        data: conditionalData,
      }),
    ),
  );

  expect(screen.getByText("width")).toBeTruthy();
  expect(screen.queryByText("megapixel_preset")).toBeNull();
  expect(screen.queryByText("custom_megapixels")).toBeNull();

  view.rerender(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, {
        ...nodeProps(),
        data: {
          ...conditionalData,
          widgetValues: {
            ...conditionalData.widgetValues,
            size_mode: "Megapixels",
          },
        },
      }),
    ),
  );
  expect(screen.queryByText("width")).toBeNull();
  expect(screen.getByText("megapixel_preset")).toBeTruthy();
  expect(screen.queryByText("custom_megapixels")).toBeNull();

  view.rerender(
    createElement(
      ReactFlowProvider,
      null,
      createElement(ComplexNode, {
        ...nodeProps(),
        data: {
          ...conditionalData,
          widgetValues: {
            ...conditionalData.widgetValues,
            size_mode: "Megapixels",
            megapixel_preset: "Custom",
          },
        },
      }),
    ),
  );
  expect(screen.getByText("custom_megapixels")).toBeTruthy();
});
