// @vitest-environment jsdom
import { cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { NodeSchema, TemplateInfo } from "../types/graph";
import { useGraphShortcuts } from "./useGraphShortcuts";

const schema: NodeSchema = {
  node_id: "demo",
  inputs: [{ name: "df", data_type: "DATAFRAME" }],
  widgets: [{ name: "k", data_type: "int", default: 5 }],
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
  code: "# demo\nk: int = 5\nreturn k: ANY\n",
  code_hash: "demo-hash",
  node_schema: schema,
};

function resetStore(): void {
  useGraphStore.setState({
    templates: [template],
    nodes: [],
    edges: [],
    subgraphs: {},
    scopeStack: [],
    rootGraph: null,
    past: [],
    future: [],
    clipboard: null,
    historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },
    editingNodeId: null,
    pendingApproval: null,
    approvalError: null,
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
}

function addNode(): string {
  return useGraphStore
    .getState()
    .addNodeFromTemplate(template, { x: 0, y: 0 });
}

function selectAll(): void {
  useGraphStore.setState((state) => ({
    nodes: state.nodes.map((node) => ({ ...node, selected: true })),
  }));
}

function press(
  init: KeyboardEventInit,
  target: EventTarget = window,
): KeyboardEvent {
  const event = new KeyboardEvent("keydown", {
    bubbles: true,
    cancelable: true,
    ...init,
  });
  target.dispatchEvent(event);
  return event;
}

beforeEach(() => {
  resetStore();
  let idCounter = 0;
  vi.stubGlobal("crypto", {
    randomUUID: () => `test-node-${++idCounter}`,
    subtle: {
      digest: async () => new Uint8Array(32).buffer,
    },
  });
  renderHook(() => useGraphShortcuts());
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("useGraphShortcuts", () => {
  it("undoes and redoes with keyboard shortcuts", () => {
    addNode();
    expect(useGraphStore.getState().nodes).toHaveLength(1);

    const undoEvent = press({ key: "z", ctrlKey: true });
    expect(undoEvent.defaultPrevented).toBe(true);
    expect(useGraphStore.getState().nodes).toHaveLength(0);

    press({ key: "z", ctrlKey: true, shiftKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(1);

    press({ key: "z", ctrlKey: true });
    press({ key: "y", ctrlKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(1);
  });

  it("selects every node with Ctrl+A", () => {
    addNode();
    addNode();

    const event = press({ key: "a", ctrlKey: true });

    expect(event.defaultPrevented).toBe(true);
    expect(
      useGraphStore.getState().nodes.every((node) => node.selected),
    ).toBe(true);
  });

  it("groups and ungroups the selection with Ctrl+G", () => {
    addNode();
    addNode();
    selectAll();

    const groupEvent = press({ key: "g", ctrlKey: true });
    expect(groupEvent.defaultPrevented).toBe(true);
    expect(
      useGraphStore.getState().nodes.some((node) => node.data.isFrame),
    ).toBe(true);

    const ungroupEvent = press({
      key: "g",
      ctrlKey: true,
      shiftKey: true,
    });
    expect(ungroupEvent.defaultPrevented).toBe(true);
    expect(
      useGraphStore.getState().nodes.some((node) => node.data.isFrame),
    ).toBe(false);
  });

  it("copies, pastes and duplicates the selection", async () => {
    addNode();
    selectAll();

    press({ key: "c", ctrlKey: true });
    expect(useGraphStore.getState().clipboard).not.toBeNull();

    press({ key: "v", ctrlKey: true });
    await vi.waitFor(() => {
      expect(useGraphStore.getState().nodes).toHaveLength(2);
    });

    selectAll();
    press({ key: "d", ctrlKey: true });
    await vi.waitFor(() => {
      expect(useGraphStore.getState().nodes).toHaveLength(4);
    });
  });

  it("cuts and pastes the selection", async () => {
    const original = addNode();
    selectAll();

    press({ key: "x", ctrlKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(0);
    expect(useGraphStore.getState().clipboard).not.toBeNull();

    press({ key: "v", ctrlKey: true });
    await vi.waitFor(() => {
      expect(useGraphStore.getState().nodes).toHaveLength(1);
    });
    expect(useGraphStore.getState().nodes[0]?.id).not.toBe(original);
  });

  it("saves with Ctrl+S even when a text field is focused", () => {
    const requestSave = vi.fn();
    useGraphStore.setState({ requestSave });
    const input = document.createElement("input");
    document.body.appendChild(input);

    const event = press({ key: "s", ctrlKey: true }, input);

    expect(event.defaultPrevented).toBe(true);
    expect(requestSave).toHaveBeenCalledTimes(1);
    input.remove();
  });

  it("ignores events from text fields", () => {
    addNode();
    const input = document.createElement("input");
    document.body.appendChild(input);

    input.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "z",
        ctrlKey: true,
        bubbles: true,
        cancelable: true,
      }),
    );
    input.remove();
    expect(useGraphStore.getState().nodes).toHaveLength(1);
  });

  it("ignores events while a modal is open", () => {
    addNode();
    useGraphStore.setState({ editingNodeId: "some-node" });

    press({ key: "z", ctrlKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(1);
  });

  it("leaves shortcuts untouched when they do not apply", () => {
    const copyEvent = press({ key: "c", ctrlKey: true });
    expect(copyEvent.defaultPrevented).toBe(false);

    const cutEvent = press({ key: "x", ctrlKey: true });
    expect(cutEvent.defaultPrevented).toBe(false);

    const plainEvent = press({ key: "z" });
    expect(plainEvent.defaultPrevented).toBe(false);
  });
});
