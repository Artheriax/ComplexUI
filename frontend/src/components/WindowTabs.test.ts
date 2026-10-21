// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type {
  GraphExecutionState,
  WorkflowWindow,
} from "../store/useGraphStore";
import { WindowTabs } from "./WindowTabs";

const idleExecution = (): GraphExecutionState => ({
  status: "idle",
  runId: null,
  order: [],
  nodeStates: {},
  previews: {},
  richOutputs: {},
  cached: {},
  error: null,
});

function makeWindow(id: string, title: string): WorkflowWindow {
  return {
    id,
    title,
    nodes: [],
    edges: [],
    subgraphs: {},
    scopeStack: [],
    rootGraph: null,
    past: [],
    future: [],
    execution: idleExecution(),
    projectName: null,
    projectSlug: null,
  };
}

beforeEach(() => {
  let counter = 0;
  vi.stubGlobal("crypto", {
    randomUUID: () => `window-${++counter}`,
  });
  vi.stubGlobal("confirm", vi.fn(() => true));
  useGraphStore.setState({
    windows: [makeWindow("one", "First"), makeWindow("two", "Second")],
    activeWindowId: "one",
    nodes: [],
    edges: [],
    subgraphs: {},
    scopeStack: [],
    rootGraph: null,
    past: [],
    future: [],
    projectName: null,
    projectSlug: null,
    execution: idleExecution(),
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("renders one tab per workflow window and switches", () => {
  render(createElement(WindowTabs));
  expect(
    screen.getByRole("tablist", { name: "Open workflows" }),
  ).toBeTruthy();
  const first = screen.getByRole("tab", { name: /First/ });
  const second = screen.getByRole("tab", { name: /Second/ });
  expect(first.getAttribute("aria-selected")).toBe("true");

  fireEvent.click(second);
  expect(useGraphStore.getState().activeWindowId).toBe("two");
  expect(screen.getByRole("tab", { name: /Second/ }).getAttribute(
    "aria-selected",
  )).toBe("true");
});

test("creates a new blank workflow window", () => {
  render(createElement(WindowTabs));
  fireEvent.click(screen.getByRole("button", { name: "New workflow window" }));

  const state = useGraphStore.getState();
  expect(state.windows).toHaveLength(3);
  expect(state.activeWindowId).toBe("window-1");
  expect(state.windows[2]?.title).toBe("Untitled 1");
});

test("closes a window and activates a neighbour", () => {
  useGraphStore.setState({ activeWindowId: "two" });
  render(createElement(WindowTabs));

  fireEvent.click(screen.getByRole("button", { name: "Close Second" }));

  const state = useGraphStore.getState();
  expect(state.windows.map((window) => window.id)).toEqual(["one"]);
  expect(state.activeWindowId).toBe("one");
});

test("prevents closing a running window", () => {
  useGraphStore.setState((state) => ({
    execution: { ...state.execution, status: "running" },
  }));
  render(createElement(WindowTabs));

  const close = screen.getByRole("button", {
    name: "Close First",
  }) as HTMLButtonElement;
  expect(close.disabled).toBe(true);
});
