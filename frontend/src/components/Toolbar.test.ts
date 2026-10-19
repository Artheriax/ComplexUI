// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { NodeSchema, TemplateInfo } from "../types/graph";
import { Toolbar } from "./Toolbar";

const schema: NodeSchema = {
  node_id: "demo",
  inputs: [],
  widgets: [],
  outputs: [{ name: "k", data_type: "ANY" }],
  dependencies: [],
  syntax_errors: [],
};

const template: TemplateInfo = {
  name: "demo",
  title: "Demo",
  category: "Preprocessing",
  description: "Demo template",
  hidden: false,
  code: "# demo\nreturn k: ANY\n",
  code_hash: "demo-hash",
  node_schema: schema,
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

beforeEach(() => {
  useGraphStore.setState({
    templates: [template],
    nodes: [],
    edges: [],
    past: [],
    future: [],
    clipboard: null,
    historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },
    wsConnected: false,
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
  let idCounter = 0;
  vi.stubGlobal("crypto", {
    randomUUID: () => `test-node-${++idCounter}`,
    subtle: { digest: async () => new Uint8Array(32).buffer },
  });
});

test("undo and redo buttons follow the history state", async () => {
  render(createElement(Toolbar));
  const undo = screen.getByTitle("Undo (Ctrl+Z)") as HTMLButtonElement;
  const redo = screen.getByTitle("Redo (Ctrl+Shift+Z)") as HTMLButtonElement;
  expect(undo.disabled).toBe(true);
  expect(redo.disabled).toBe(true);

  await act(async () => {
    useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
  });

  expect(undo.disabled).toBe(false);
  fireEvent.click(undo);
  expect(useGraphStore.getState().nodes).toHaveLength(0);
  expect(redo.disabled).toBe(false);

  fireEvent.click(redo);
  expect(useGraphStore.getState().nodes).toHaveLength(1);
});

test("history buttons are disabled while a run is active", async () => {
  render(createElement(Toolbar));
  await act(async () => {
    useGraphStore
      .getState()
      .addNodeFromTemplate(template, { x: 0, y: 0 });
    useGraphStore.setState((state) => ({
      execution: { ...state.execution, status: "running" },
    }));
  });

  expect(
    (screen.getByTitle("Undo (Ctrl+Z)") as HTMLButtonElement).disabled,
  ).toBe(true);
});

test("icon buttons and backend status expose accessible names", () => {
  render(createElement(Toolbar));

  expect(screen.getByRole("button", { name: "Undo" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Redo" })).toBeTruthy();
  expect(screen.getByRole("status").textContent).toContain("Backend offline");

  act(() => {
    useGraphStore.getState().setWsConnected(true);
  });
  expect(screen.getByRole("status").textContent).toContain("Backend connected");
});
