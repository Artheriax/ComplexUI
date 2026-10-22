// @vitest-environment jsdom
import { cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import {
  DEFAULT_SETTINGS,
  useSettingsStore,
} from "../store/useSettingsStore";
import type { AppNode, NodeSchema } from "../types/graph";
import { useWorkflowFileActions } from "./useWorkflowFileActions";

const emptySchema: NodeSchema = {
  node_id: "n1",
  inputs: [],
  widgets: [],
  outputs: [],
  dependencies: [],
  syntax_errors: [],
};

const node: AppNode = {
  id: "n1",
  type: "complex",
  position: { x: 0, y: 0 },
  data: {
    label: "Node",
    templateName: null,
    templateCode: null,
    code: "",
    schema: emptySchema,
    widgetValues: {},
    isTrusted: true,
    color: "default",
  },
};

const clearCanvas = vi.fn();

beforeEach(() => {
  clearCanvas.mockClear();
  useSettingsStore.setState({
    settings: { ...DEFAULT_SETTINGS },
    isOpen: false,
  });
  useGraphStore.setState({
    nodes: [node],
    edges: [],
    subgraphs: {},
    scopeStack: [],
    clearCanvas,
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("clears the canvas without confirmation when disabled", () => {
  useSettingsStore
    .getState()
    .updateSetting("confirmClearCanvas", false);
  const confirm = vi.fn(() => true);
  vi.stubGlobal("confirm", confirm);
  const { result } = renderHook(() => useWorkflowFileActions());

  result.current.clearCanvas();

  expect(clearCanvas).toHaveBeenCalledTimes(1);
  expect(confirm).not.toHaveBeenCalled();
});

test("honours the confirmation prompt when enabled", () => {
  const confirm = vi.fn(() => false);
  vi.stubGlobal("confirm", confirm);
  const { result } = renderHook(() => useWorkflowFileActions());

  result.current.clearCanvas();

  expect(confirm).toHaveBeenCalledTimes(1);
  expect(clearCanvas).not.toHaveBeenCalled();

  confirm.mockReturnValue(true);
  result.current.clearCanvas();

  expect(clearCanvas).toHaveBeenCalledTimes(1);
});
