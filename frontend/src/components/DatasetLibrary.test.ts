// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import type { DatasetSummary } from "../lib/api";
import { useDatasetsStore } from "../store/useDatasetsStore";
import { DatasetLibrary } from "./DatasetLibrary";

afterEach(cleanup);

const iris: DatasetSummary = {
  id: "iris.csv",
  name: "iris.csv",
  path: "data/uploads/iris.csv",
  format: "csv",
  kind: "table",
  size_bytes: 2_048,
  modified_at: "2026-10-20T12:00:00Z",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function resetStore(): void {
  useDatasetsStore.setState({
    datasets: [],
    datasetsLoaded: false,
    datasetsLoading: false,
    datasetsError: null,
    uploading: false,
  });
}

beforeEach(() => {
  resetStore();
  vi.unstubAllGlobals();
});

test("lists datasets and opens a read-only preview", async () => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/preview")) {
      return jsonResponse({
        dataset: iris,
        columns: ["sepal_length", "species"],
        rows: [[5.1, "setosa"]],
        truncated: true,
        row_limit: 20,
      });
    }
    return jsonResponse([iris]);
  });
  vi.stubGlobal("fetch", fetchMock);

  render(createElement(DatasetLibrary));

  expect(await screen.findByText("iris.csv")).toBeTruthy();
  expect(screen.getByText("data/uploads/iris.csv")).toBeTruthy();
  fireEvent.click(screen.getByText("Preview"));

  expect(await screen.findByText("sepal_length")).toBeTruthy();
  expect(screen.getByText("5.1")).toBeTruthy();
  expect(screen.getByText("setosa")).toBeTruthy();
  expect(screen.getByText(/more rows available/)).toBeTruthy();
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/uploads/iris.csv/preview?limit=20",
    expect.any(Object),
  );
});

test("previews image assets inline without a table preview request", async () => {
  const photo: DatasetSummary = {
    ...iris,
    id: "photo.png",
    name: "photo.png",
    path: "data/uploads/photo.png",
    format: "png",
    kind: "image",
  };
  const fetchMock = vi.fn(async (_input: RequestInfo | URL) =>
    jsonResponse([photo]),
  );
  vi.stubGlobal("fetch", fetchMock);

  render(createElement(DatasetLibrary));

  expect(await screen.findByText("photo.png")).toBeTruthy();
  fireEvent.click(screen.getByText("Preview"));

  const image = await screen.findByAltText("photo.png");
  expect(image.getAttribute("src")).toBe("/api/uploads/photo.png/raw");
  expect(
    fetchMock.mock.calls.some((call) => String(call[0]).includes("/preview")),
  ).toBe(false);
});

test("uploads a selected file into the library", async () => {
  const uploaded: DatasetSummary = {
    ...iris,
    id: "wine.csv",
    name: "wine.csv",
    path: "data/uploads/wine.csv",
  };
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "POST") {
      return jsonResponse({ status: "uploaded", dataset: uploaded }, 201);
    }
    return jsonResponse([]);
  });
  vi.stubGlobal("fetch", fetchMock);

  render(createElement(DatasetLibrary));
  expect(await screen.findByText(/No assets yet/)).toBeTruthy();

  const file = new File(["a,b\n1,2\n"], "wine.csv", { type: "text/csv" });
  fireEvent.change(screen.getByLabelText("Upload asset files"), {
    target: { files: [file] },
  });

  await waitFor(() => expect(screen.getByText("wine.csv")).toBeTruthy());
  const post = fetchMock.mock.calls.find(
    ([, init]) => (init as RequestInit | undefined)?.method === "POST",
  );
  expect(post).toBeTruthy();
  expect((post?.[1]?.body as FormData).get("file")).toMatchObject({
    name: "wine.csv",
  });
});

test("deletes a dataset after confirmation", async () => {
  const fetchMock = vi.fn(async () =>
    jsonResponse({ status: "deleted", dataset: iris }),
  );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("confirm", vi.fn(() => true));
  useDatasetsStore.setState({
    datasets: [iris],
    datasetsLoaded: true,
    datasetsLoading: false,
    datasetsError: null,
    uploading: false,
  });

  render(createElement(DatasetLibrary));
  fireEvent.click(screen.getByLabelText("Delete iris.csv"));

  await waitFor(() =>
    expect(useDatasetsStore.getState().datasets).toEqual([]),
  );
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/uploads/iris.csv",
    expect.objectContaining({ method: "DELETE" }),
  );
});
