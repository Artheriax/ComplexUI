import { beforeEach, describe, expect, it, vi } from "vitest";

import type { DatasetSummary } from "../lib/api";
import { useDatasetsStore } from "./useDatasetsStore";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function makeDataset(overrides: Partial<DatasetSummary> = {}): DatasetSummary {
  return {
    id: "iris.csv",
    name: "iris.csv",
    path: "data/uploads/iris.csv",
    format: "csv",
    kind: "table",
    size_bytes: 42,
    modified_at: "2026-10-20T12:00:00Z",
    ...overrides,
  };
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

describe("dataset store", () => {
  it("loads workspace datasets", async () => {
    const dataset = makeDataset();
    const fetchMock = vi.fn(async () => jsonResponse([dataset]));
    vi.stubGlobal("fetch", fetchMock);

    await useDatasetsStore.getState().loadDatasets();

    expect(fetchMock).toHaveBeenCalledWith("/api/uploads", expect.any(Object));
    const state = useDatasetsStore.getState();
    expect(state.datasets).toEqual([dataset]);
    expect(state.datasetsLoaded).toBe(true);
    expect(state.datasetsError).toBeNull();
  });

  it("uploads a file as multipart data and prepends it", async () => {
    const uploaded = makeDataset({ id: "wine.csv", name: "wine.csv" });
    let posted: FormData | null = null;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === "POST") {
        posted = init.body as FormData;
        return jsonResponse({ status: "uploaded", dataset: uploaded }, 201);
      }
      return jsonResponse([]);
    });
    vi.stubGlobal("fetch", fetchMock);

    const file = new File(["a,b\n1,2\n"], "wine.csv", { type: "text/csv" });
    const result = await useDatasetsStore.getState().uploadFiles([file]);

    expect(posted).toBeInstanceOf(FormData);
    expect((posted as unknown as FormData).get("file")).toMatchObject({
      name: "wine.csv",
    });
    expect(result).toEqual([uploaded]);
    expect(useDatasetsStore.getState().datasets).toEqual([uploaded]);
  });

  it("removes a dataset after the API succeeds", async () => {
    const dataset = makeDataset();
    const fetchMock = vi.fn(async () =>
      jsonResponse({ status: "deleted", dataset }),
    );
    vi.stubGlobal("fetch", fetchMock);
    useDatasetsStore.setState({ datasets: [dataset], datasetsLoaded: true });

    const removed = await useDatasetsStore.getState().removeDataset(dataset.id);

    expect(removed).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/uploads/iris.csv",
      expect.objectContaining({ method: "DELETE" }),
    );
    expect(useDatasetsStore.getState().datasets).toEqual([]);
  });

  it("records load failures without marking the list loaded", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ detail: "disk unavailable" }, 500)),
    );

    await useDatasetsStore.getState().loadDatasets();

    const state = useDatasetsStore.getState();
    expect(state.datasetsError).toContain("disk unavailable");
    expect(state.datasetsLoaded).toBe(false);
    expect(state.datasetsLoading).toBe(false);
  });
});
