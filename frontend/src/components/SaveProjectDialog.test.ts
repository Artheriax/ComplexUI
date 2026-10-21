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
import { SaveProjectDialog } from "./SaveProjectDialog";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  useGraphStore.setState({
    nodes: [],
    edges: [],
    past: [],
    future: [],
    clipboard: null,
    historyMeta: { coalesceKey: null, coalesceAt: 0, dragging: false },
    projectName: null,
    projectSlug: null,
    saveDialogOpen: true,
    saveBusy: false,
    saveError: null,
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
    randomUUID: () => `dialog-node-${++idCounter}`,
    subtle: { digest: async () => new Uint8Array(32).buffer },
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("names and saves an untitled project", async () => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) =>
    String(input) === "/api/parse"
      ? jsonResponse({
          node_id: "",
          inputs: [],
          widgets: [],
          outputs: [],
          dependencies: [],
          syntax_errors: [],
        })
      : jsonResponse({
          status: "created",
          project: {
            slug: "iris-demo",
            name: "Iris Demo",
            saved_at: "2026-10-20T12:00:00Z",
            node_count: 1,
            edge_count: 0,
            version_count: 0,
          },
        }),
  );
  vi.stubGlobal("fetch", fetchMock);
  useGraphStore.getState().addScratchNode({ x: 0, y: 0 });

  render(createElement(SaveProjectDialog));

  expect(screen.getByRole("dialog", { name: "Name this project" })).toBeTruthy();
  expect(screen.getByRole("checkbox")).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Project name"), {
    target: { value: "Iris Demo" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save project" }));

  await waitFor(() => {
    expect(useGraphStore.getState().projectSlug).toBe("iris-demo");
  });
  expect(useGraphStore.getState().saveDialogOpen).toBe(false);
  expect(
    fetchMock.mock.calls.filter(([url]) => String(url) === "/api/projects"),
  ).toHaveLength(1);
});

test("shows a conflict and offers overwrite", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      String(input) === "/api/parse"
        ? jsonResponse({
            node_id: "",
            inputs: [],
            widgets: [],
            outputs: [],
            dependencies: [],
            syntax_errors: [],
          })
        : jsonResponse({ detail: "A project named 'Taken' already exists" }, 409),
    ),
  );
  useGraphStore.getState().addScratchNode({ x: 0, y: 0 });

  render(createElement(SaveProjectDialog));
  fireEvent.change(screen.getByLabelText("Project name"), {
    target: { value: "Taken" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save project" }));

  expect((await screen.findByRole("alert")).textContent).toContain(
    "already exists",
  );
  expect(screen.getByRole("checkbox")).toBeTruthy();
  expect(useGraphStore.getState().saveDialogOpen).toBe(true);
});
