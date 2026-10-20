// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { TemplateInfo } from "../types/graph";
import { RenameNodeDialog } from "./RenameNodeDialog";

afterEach(cleanup);

const template: TemplateInfo = {
  name: "demo",
  title: "Demo Node",
  category: "Data",
  description: "Demo",
  hidden: false,
  code: "",
  code_hash: "hash",
  node_schema: {
    node_id: "demo",
    inputs: [],
    widgets: [],
    outputs: [],
    dependencies: [],
    syntax_errors: [],
  },
};

beforeEach(() => {
  useGraphStore.setState({
    nodes: [],
    edges: [],
    templates: [template],
  });
});

test("validates and saves a trimmed node name", () => {
  const id = useGraphStore
    .getState()
    .addNodeFromTemplate(template, { x: 0, y: 0 });
  const onClose = vi.fn();
  render(createElement(RenameNodeDialog, { nodeId: id, onClose }));

  expect(screen.getByRole("dialog", { name: "Rename node" })).toBeTruthy();
  const input = screen.getByRole("textbox", { name: "Node name" });
  expect((input as HTMLInputElement).value).toBe("Demo Node");

  fireEvent.change(input, { target: { value: "   " } });
  fireEvent.click(screen.getByRole("button", { name: /Save/ }));

  expect(
    screen.getByText("Enter a node name between 1 and 80 characters."),
  ).toBeTruthy();
  expect(useGraphStore.getState().nodes[0]?.data.label).toBe("Demo Node");
  expect(onClose).not.toHaveBeenCalled();

  fireEvent.change(input, { target: { value: "  Tuned Node  " } });
  fireEvent.click(screen.getByRole("button", { name: /Save/ }));

  expect(useGraphStore.getState().nodes[0]?.data.label).toBe("Tuned Node");
  expect(onClose).toHaveBeenCalledOnce();
});
