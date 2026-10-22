// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import {
  DEFAULT_SETTINGS,
  useSettingsStore,
} from "../store/useSettingsStore";
import { useUiStore } from "../store/useUiStore";
import type { AppNode, NodeSchema } from "../types/graph";
import { CanvasStatusBar } from "./CanvasStatusBar";

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

const runGraph = vi.fn(async () => undefined);

beforeEach(() => {
  runGraph.mockClear();
  useSettingsStore.setState({
    settings: { ...DEFAULT_SETTINGS },
    isOpen: false,
  });
  useUiStore.setState({ consoleOpen: false });
  useGraphStore.setState({
    wsConnected: true,
    nodes: [node],
    windows: [],
    activeWindowId: "window-1",
    scopeStack: [],
    runGraph,
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

afterEach(cleanup);

test("opens the console on run when enabled", () => {
  render(createElement(CanvasStatusBar));

  fireEvent.click(screen.getByRole("button", { name: "Run workflow" }));

  expect(runGraph).toHaveBeenCalledTimes(1);
  expect(useUiStore.getState().consoleOpen).toBe(true);
});

test("leaves the console closed on run when disabled", () => {
  useSettingsStore
    .getState()
    .updateSetting("autoOpenConsole", false);
  render(createElement(CanvasStatusBar));

  fireEvent.click(screen.getByRole("button", { name: "Run workflow" }));

  expect(runGraph).toHaveBeenCalledTimes(1);
  expect(useUiStore.getState().consoleOpen).toBe(false);
});
