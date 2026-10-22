// @vitest-environment jsdom
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import type { OutputSummary } from "../lib/api";
import { useGraphStore } from "../store/useGraphStore";
import { useOutputsStore } from "../store/useOutputsStore";
import { OutputsLibrary } from "./OutputsLibrary";

const output: OutputSummary = {
  id: "result.png",
  name: "result.png",
  path: "outputs/result.png",
  format: "PNG",
  kind: "image",
  size_bytes: 2048,
  modified_at: "2026-10-22T00:00:00Z",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function resetOutputs(): void {
  useOutputsStore.setState({
    outputs: [],
    outputsLoaded: false,
    outputsLoading: false,
    outputsError: null,
  });
}

beforeEach(() => {
  resetOutputs();
  useGraphStore.setState({
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
  vi.unstubAllGlobals();
});

afterEach(cleanup);

test("lists generated outputs and opens the media preview", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse([output])),
  );

  render(createElement(OutputsLibrary));

  expect(await screen.findByText("result.png")).toBeTruthy();
  expect(screen.getByText("outputs/result.png")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Preview result.png" }));

  const dialog = await screen.findByRole("dialog", {
    name: /Asset preview — result.png/,
  });
  const image = within(dialog).getByAltText("result.png");
  expect(image.getAttribute("src")).toBe("/api/outputs/result.png/raw");
});

test("deletes an output after confirmation", async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "DELETE") {
      return jsonResponse({ status: "deleted", output });
    }
    return jsonResponse([output]);
  });
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("confirm", vi.fn(() => true));

  render(createElement(OutputsLibrary));
  expect(await screen.findByText("result.png")).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Delete result.png" }));

  await waitFor(() =>
    expect(useOutputsStore.getState().outputs).toEqual([]),
  );
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/outputs/result.png",
    expect.objectContaining({ method: "DELETE" }),
  );
});
