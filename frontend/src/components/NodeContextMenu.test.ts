// @vitest-environment jsdom
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { TemplateInfo } from "../types/graph";
import { NodeContextMenu } from "./NodeContextMenu";

afterEach(cleanup);

const template: TemplateInfo = {
  name: "demo",
  title: "Demo Node",
  category: "Data",
  description: "Demo",
  hidden: false,
  code: "value = 1\n",
  code_hash: "hash",
  node_schema: {
    node_id: "demo",
    inputs: [],
    widgets: [],
    outputs: [{ name: "value", data_type: "ANY" }],
    dependencies: [],
    syntax_errors: [],
  },
};

beforeEach(() => {
  useGraphStore.setState({
    nodes: [],
    edges: [],
    templates: [template],
    clipboard: null,
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
});

function renderMenu(
  nodeId: string,
  onClose = vi.fn(),
  onRename = vi.fn(),
  onSaveTemplate = vi.fn(),
  onSweep = vi.fn(),
) {
  render(
    createElement(NodeContextMenu, {
      nodeId,
      anchor: { x: 30, y: 40 },
      flowPosition: { x: 50, y: 60 },
      onClose,
      onRename,
      onSaveTemplate,
      onSweep,
    }),
  );
  return { onClose, onRename, onSaveTemplate, onSweep };
}

test("applies colors and opens rename for the clicked node", () => {
  const id = useGraphStore
    .getState()
    .addNodeFromTemplate(template, { x: 0, y: 0 });
  useGraphStore.getState().selectOnlyNode(id);
  const { onClose, onRename } = renderMenu(id);

  expect(
    screen.getByRole("menu", { name: "Node actions for Demo Node" }),
  ).toBeTruthy();
  const rose = screen.getByRole("button", { name: "Set color Rose" });
  expect(rose.getAttribute("aria-pressed")).toBe("false");
  fireEvent.click(rose);
  expect(useGraphStore.getState().nodes[0]?.data.color).toBe("rose");

  fireEvent.click(screen.getByRole("menuitem", { name: /Rename/ }));
  expect(onRename).toHaveBeenCalledWith(id);
  expect(onClose).toHaveBeenCalledOnce();
});

test("copies the selection and pastes it at the menu position", async () => {
  const id = useGraphStore
    .getState()
    .addNodeFromTemplate(template, { x: 0, y: 0 });
  useGraphStore.getState().selectOnlyNode(id);
  renderMenu(id);

  fireEvent.click(screen.getByRole("menuitem", { name: /Copy/ }));
  expect(useGraphStore.getState().clipboard?.nodes).toHaveLength(1);

  fireEvent.click(screen.getByRole("menuitem", { name: /Paste here/ }));
  await waitFor(() => expect(useGraphStore.getState().nodes).toHaveLength(2));
  expect(useGraphStore.getState().nodes[1]?.position).toEqual({ x: 50, y: 60 });
});

test("disables reset when the template is unavailable and deletes nodes", () => {
  const id = useGraphStore
    .getState()
    .addNodeFromTemplate(template, { x: 0, y: 0 });
  useGraphStore.getState().selectOnlyNode(id);
  useGraphStore.setState({ templates: [] });
  const { onClose } = renderMenu(id);

  const reset = screen.getByRole("menuitem", { name: /Reset node/ });
  expect(reset.hasAttribute("disabled")).toBe(true);

  fireEvent.click(screen.getByRole("menuitem", { name: /Delete/ }));
  expect(useGraphStore.getState().nodes).toHaveLength(0);
  expect(onClose).toHaveBeenCalledOnce();
});

test("opens sweep and save-as-template actions", () => {
  const id = useGraphStore
    .getState()
    .addNodeFromTemplate(template, { x: 0, y: 0 });
  useGraphStore.getState().selectOnlyNode(id);
  const { onSaveTemplate, onSweep } = renderMenu(id);

  fireEvent.click(screen.getByText("Sweep…"));
  expect(onSweep).toHaveBeenCalledWith(id);

  fireEvent.click(screen.getByText("Save as template…"));
  expect(onSaveTemplate).toHaveBeenCalledWith(id);
});
