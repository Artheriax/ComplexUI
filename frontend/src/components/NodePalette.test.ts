// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test } from "vitest";

import { useDatasetsStore } from "../store/useDatasetsStore";
import { useGraphStore } from "../store/useGraphStore";
import { useSettingsStore } from "../store/useSettingsStore";
import { useUiStore } from "../store/useUiStore";
import type { NodeSchema, TemplateInfo } from "../types/graph";
import { NodePalette } from "./NodePalette";

afterEach(cleanup);

const schema: NodeSchema = {
  node_id: "demo",
  inputs: [],
  widgets: [],
  outputs: [{ name: "k", data_type: "ANY" }],
  dependencies: [],
  syntax_errors: [],
};

const demoTemplate: TemplateInfo = {
  name: "demo",
  title: "Demo",
  category: "Preprocessing",
  description: "Demo template",
  hidden: false,
  code: "# demo\nreturn k: ANY\n",
  code_hash: "hash",
  node_schema: schema,
};

beforeEach(() => {
  useGraphStore.setState({
    templates: [demoTemplate],
    templatesLoaded: true,
    templatesError: null,
    workflows: [],
    workflowsLoaded: true,
    workflowsError: null,
    projects: [],
    projectsLoaded: true,
    projectsError: null,
  });
  useDatasetsStore.setState({
    datasets: [],
    datasetsLoaded: true,
    datasetsLoading: false,
    datasetsError: null,
    uploading: false,
  });
  useUiStore.setState({
    sidebarPanel: null,
    consoleOpen: false,
    shortcutsOpen: false,
  });
  useSettingsStore.setState({ isOpen: false });
});

test("starts as a compact rail and expands a panel on demand", () => {
  render(createElement(NodePalette));

  expect(screen.getByRole("navigation", { name: "Workspace panels" })).toBeTruthy();
  expect(screen.queryByPlaceholderText("Search templates…")).toBeNull();

  fireEvent.click(screen.getByRole("button", { name: "Nodes" }));
  expect(screen.getByPlaceholderText("Search templates…")).toBeTruthy();
  fireEvent.click(
    screen.getByRole("button", { name: /Preprocessing/ }),
  );
  expect(screen.getByText("Demo")).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Nodes" }));
  expect(screen.queryByPlaceholderText("Search templates…")).toBeNull();
});

test("switches between Workflows, templates and assets", () => {
  render(createElement(NodePalette));

  fireEvent.click(screen.getByRole("button", { name: "Workflows" }));
  expect(screen.getByPlaceholderText("Search workflows…")).toBeTruthy();
  expect(screen.getByText("Refresh workflow list")).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Templates" }));
  expect(screen.getByPlaceholderText("Search templates…")).toBeTruthy();
  expect(screen.getByText(/Opening a template adds it as a new/)).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Assets" }));
  expect(screen.getByPlaceholderText("Search assets…")).toBeTruthy();
  expect(screen.getByLabelText("Upload asset files")).toBeTruthy();
});

test("bottom rail buttons toggle console, shortcuts and settings", () => {
  render(createElement(NodePalette));

  fireEvent.click(screen.getByRole("button", { name: "Console" }));
  expect(useUiStore.getState().consoleOpen).toBe(true);

  fireEvent.click(screen.getByRole("button", { name: "Keyboard shortcuts" }));
  expect(useUiStore.getState().shortcutsOpen).toBe(true);

  fireEvent.click(screen.getByRole("button", { name: "Settings" }));
  expect(useSettingsStore.getState().isOpen).toBe(true);
});

