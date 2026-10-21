// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { AppNode, NodeSchema } from "../types/graph";
import { CanvasContextMenu } from "./CanvasContextMenu";

afterEach(cleanup);

const schema: NodeSchema = {
  node_id: "a",
  inputs: [],
  widgets: [],
  outputs: [],
  dependencies: [],
  syntax_errors: [],
};

function makeNode(id: string): AppNode {
  return {
    id,
    type: "complex",
    position: { x: 0, y: 0 },
    data: {
      label: id,
      templateName: null,
      templateCode: null,
      code: "",
      schema,
      widgetValues: {},
      isTrusted: true,
      color: "default",
    },
  };
}

function renderMenu(overrides: {
  onClose?: () => void;
  onDismissAll?: () => void;
  onFocusSearch?: () => void;
  onFitView?: () => void;
  onImport?: () => void;
} = {}) {
  const onClose = overrides.onClose ?? vi.fn();
  const onDismissAll = overrides.onDismissAll ?? vi.fn();
  const onFocusSearch = overrides.onFocusSearch ?? vi.fn();
  const onFitView = overrides.onFitView ?? vi.fn();
  const onImport = overrides.onImport ?? vi.fn();
  const result = render(
    createElement(CanvasContextMenu, {
      anchor: { x: 400, y: 300 },
      flowPosition: { x: 10, y: 20 },
      onClose,
      onDismissAll,
      onFocusSearch,
      onFitView,
      onImport,
    }),
  );
  return {
    ...result,
    onClose,
    onDismissAll,
    onFocusSearch,
    onFitView,
    onImport,
  };
}

beforeEach(() => {
  useGraphStore.setState({
    nodes: [makeNode("a"), makeNode("b")],
    edges: [],
    subgraphs: {},
    scopeStack: [],
    rootGraph: null,
    past: [],
    future: [],
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

test("opens at the cursor and closes outside", () => {
  const { onDismissAll } = renderMenu();

  const menu = screen.getByRole("menu", { name: "Canvas actions" });
  expect(menu).toHaveProperty("style.left", "400px");
  expect(menu).toHaveProperty("style.top", "300px");
  fireEvent.pointerDown(document.body);
  expect(onDismissAll).toHaveBeenCalledTimes(1);
});

test("runs canvas actions and closes all overlays", () => {
  const { onDismissAll, onFitView } = renderMenu();

  fireEvent.click(
    screen.getByRole("menuitem", { name: /^Select all/ }),
  );
  expect(useGraphStore.getState().nodes.every((node) => node.selected)).toBe(
    true,
  );
  expect(onDismissAll).toHaveBeenCalled();

  fireEvent.click(screen.getByRole("menuitem", { name: "Fit view" }));
  expect(onFitView).toHaveBeenCalledTimes(1);
});

test("add node focuses the search and keeps it open", () => {
  const { onClose, onDismissAll, onFocusSearch } = renderMenu();

  fireEvent.click(screen.getByRole("menuitem", { name: "Add node…" }));

  expect(onFocusSearch).toHaveBeenCalledTimes(1);
  expect(onClose).toHaveBeenCalledTimes(1);
  expect(onDismissAll).not.toHaveBeenCalled();
});

test("opens the File submenu with workflow actions", () => {
  const { onImport, onDismissAll } = renderMenu();

  fireEvent.click(screen.getByRole("menuitem", { name: "File" }));

  const fileMenu = screen.getByRole("menu", { name: "File actions" });
  expect(fileMenu).toBeTruthy();
  expect(screen.getByRole("menuitem", { name: /Save project/ })).toBeTruthy();
  expect(
    screen.getByRole("menuitem", { name: "Export workflow JSON" }),
  ).toBeTruthy();
  expect(
    screen.getByRole("menuitem", { name: "Export Python script" }),
  ).toBeTruthy();
  expect(
    screen.getByRole("menuitem", { name: "Export Jupyter notebook" }),
  ).toBeTruthy();
  expect(screen.getByRole("menuitem", { name: "Clear canvas" })).toBeTruthy();

  fireEvent.click(screen.getByRole("menuitem", { name: "Import workflow…" }));
  expect(onImport).toHaveBeenCalledTimes(1);
  expect(onDismissAll).toHaveBeenCalled();
});

test("disables undo and redo when history is empty", () => {
  renderMenu();
  const undo = screen.getByRole("menuitem", { name: /^Undo/ });
  const redo = screen.getByRole("menuitem", { name: /^Redo/ });
  expect((undo as HTMLButtonElement).disabled).toBe(true);
  expect((redo as HTMLButtonElement).disabled).toBe(true);
});
