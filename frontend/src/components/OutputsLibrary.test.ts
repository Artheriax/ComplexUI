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

function imageResponse(blob: Blob): Response {
  return {
    ok: true,
    status: 200,
    blob: async () => blob,
  } as Response;
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
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: undefined,
  });
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

const datedOutput: OutputSummary = {
  id: "2026-10-23/result.png",
  name: "result.png",
  path: "outputs/2026-10-23/result.png",
  format: "PNG",
  kind: "image",
  size_bytes: 2048,
  modified_at: "2026-10-23T00:00:00Z",
};

test("serves previews from dated output folders", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse([datedOutput])),
  );

  render(createElement(OutputsLibrary));
  expect(await screen.findByText("result.png")).toBeTruthy();
  expect(screen.getByText("outputs/2026-10-23/result.png")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Preview result.png" }));

  const dialog = await screen.findByRole("dialog", {
    name: /Asset preview — result.png/,
  });
  const image = within(dialog).getByAltText("result.png");
  expect(image.getAttribute("src")).toBe(
    "/api/outputs/2026-10-23/result.png/raw",
  );
});

test("copies an output image to the clipboard", async () => {
  const write = vi.fn(async () => undefined);
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { write },
  });
  class FakeClipboardItem {
    constructor(readonly items: Record<string, Blob>) {}
  }
  vi.stubGlobal("ClipboardItem", FakeClipboardItem);
  const png = new Blob([new Uint8Array([1, 2, 3])], { type: "image/png" });
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/raw")) {
        return imageResponse(png);
      }
      return jsonResponse([output]);
    }),
  );

  render(createElement(OutputsLibrary));
  expect(await screen.findByText("result.png")).toBeTruthy();
  fireEvent.click(
    screen.getByRole("button", {
      name: "Copy result.png to clipboard",
    }),
  );

  expect(await screen.findByText("Copied image")).toBeTruthy();
  expect(write).toHaveBeenCalledTimes(1);
});

test("opens and reveals outputs through the API", async () => {
  const fetchMock = vi.fn(
    async (_input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === "POST") {
        return jsonResponse({ status: "opened" });
      }
      return jsonResponse([output]);
    },
  );
  vi.stubGlobal("fetch", fetchMock);

  render(createElement(OutputsLibrary));
  fireEvent.click(
    await screen.findByRole("button", { name: "Open result.png" }),
  );
  await waitFor(() =>
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/outputs/result.png/open",
      expect.objectContaining({ method: "POST" }),
    ),
  );

  fireEvent.click(
    screen.getByRole("button", { name: "Show result.png in folder" }),
  );
  await waitFor(() =>
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/outputs/result.png/reveal",
      expect.objectContaining({ method: "POST" }),
    ),
  );
});

test("deletes dated outputs with their encoded id", async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "DELETE") {
      return jsonResponse({ status: "deleted", output: datedOutput });
    }
    return jsonResponse([datedOutput]);
  });
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("confirm", vi.fn(() => true));

  render(createElement(OutputsLibrary));
  expect(await screen.findByText("result.png")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Delete result.png" }));

  await waitFor(() =>
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/outputs/2026-10-23/result.png",
      expect.objectContaining({ method: "DELETE" }),
    ),
  );
});
