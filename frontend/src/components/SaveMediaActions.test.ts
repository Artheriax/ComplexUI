// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import type { NodeExecutionState } from "../types/graph";
import { useGraphStore } from "../store/useGraphStore";
import { SaveMediaActions } from "./SaveMediaActions";

const originalRunGraph = useGraphStore.getState().runGraph;
const originalUpdateWidgetValue = useGraphStore.getState().updateWidgetValue;
const runGraph = vi.fn(async () => undefined);
const updateWidgetValue = vi.fn();

const savedExecution: NodeExecutionState = {
  status: "completed",
  outputs: {
    saved_path: "outputs/2026-10-23/model_001.png",
    saved: {
      kind: "image",
      width: 512,
      height: 512,
      mode: "RGB",
      thumbnail: null,
    },
  },
};

beforeEach(() => {
  useGraphStore.setState({
    runGraph,
    updateWidgetValue,
  });
});

afterEach(() => {
  cleanup();
  useGraphStore.setState({
    runGraph: originalRunGraph,
    updateWidgetValue: originalUpdateWidgetValue,
  });
  vi.clearAllMocks();
});

function renderActions(overrides: {
  execution?: NodeExecutionState;
  running?: boolean;
  isTrusted?: boolean;
  hasMediaInput?: boolean;
  values?: Record<string, unknown>;
} = {}) {
  return render(
    createElement(SaveMediaActions, {
      nodeId: "save",
      values: {
        filename: "%model%_%counter%",
        save_nonce: 4,
        ...overrides.values,
      },
      execution: overrides.execution,
      running: overrides.running ?? false,
      isTrusted: overrides.isTrusted ?? true,
      hasMediaInput: overrides.hasMediaInput ?? true,
    }),
  );
}

test("saves by bumping the nonce and running the node", () => {
  renderActions();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  expect(updateWidgetValue).toHaveBeenCalledWith("save", "save_nonce", 5);
  expect(runGraph).toHaveBeenCalledWith(["save"]);
});

test("disables save for invalid templates, missing input and untrusted code", () => {
  renderActions({ values: { filename: "image" } });
  expect(
    screen.getByRole("button", { name: "Save" }).hasAttribute("disabled"),
  ).toBe(true);
  cleanup();

  renderActions({ hasMediaInput: false });
  expect(
    screen.getByRole("button", { name: "Save" }).hasAttribute("disabled"),
  ).toBe(true);
  cleanup();

  renderActions({ isTrusted: false });
  expect(
    screen.getByRole("button", { name: "Save" }).hasAttribute("disabled"),
  ).toBe(true);
});

test("enables output actions after a saved path exists", async () => {
  renderActions({ execution: savedExecution });
  expect(
    screen.getByRole("button", { name: "Preview" }).hasAttribute("disabled"),
  ).toBe(false);
  expect(
    screen
      .getByRole("button", { name: "Copy model_001.png to clipboard" })
      .hasAttribute("disabled"),
  ).toBe(false);
  expect(
    screen.getByRole("button", { name: "Open model_001.png" }).hasAttribute(
      "disabled",
    ),
  ).toBe(false);

  fireEvent.click(screen.getByRole("button", { name: "Preview" }));
  const dialog = await screen.findByRole("dialog", {
    name: /Asset preview — model_001.png/,
  });
  expect(dialog).toBeTruthy();
  expect(
    screen.getByAltText("model_001.png").getAttribute("src"),
  ).toBe("/api/outputs/2026-10-23/model_001.png/raw");
});
