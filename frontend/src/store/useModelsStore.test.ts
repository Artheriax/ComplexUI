import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelSummary, ModelTypeSummary } from "../lib/api";
import { useModelsStore } from "./useModelsStore";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const yoloType: ModelTypeSummary = {
  name: "yolo",
  path: "models/yolo",
  count: 1,
  default: true,
};
const model: ModelSummary = {
  id: "yolo/custom.pt",
  name: "custom.pt",
  path: "models/yolo/custom.pt",
  model_type: "yolo",
  format: "PyTorch",
  size_bytes: 1_024,
  modified_at: "2026-10-20T12:00:00Z",
  mtime_ns: 123,
  fingerprint: "1024:123",
  executable: true,
  is_directory: false,
};

function resetStore(): void {
  useModelsStore.setState({
    types: [],
    typesLoaded: false,
    modelsByType: {},
    loadedTypes: {},
    loading: false,
    refreshing: false,
    error: null,
    lastUpdated: null,
  });
}

beforeEach(() => {
  resetStore();
  vi.unstubAllGlobals();
});

describe("models store", () => {
  it("loads types and the whole library", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      return jsonResponse(url.includes("/types") ? [yoloType] : [model]);
    });
    vi.stubGlobal("fetch", fetchMock);

    await useModelsStore.getState().loadLibrary();

    const state = useModelsStore.getState();
    expect(state.types).toEqual([yoloType]);
    expect(state.modelsByType.yolo).toEqual([model]);
    expect(state.loadedTypes.yolo).toBe(true);
    expect(state.hasModel("yolo", model.path)).toBe(true);
    expect(state.lastUpdated).not.toBeNull();
  });

  it("loads and refreshes one type", async () => {
    const fetchMock = vi.fn(async () => jsonResponse([model]));
    vi.stubGlobal("fetch", fetchMock);

    await useModelsStore.getState().loadType("yolo");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/models?type=yolo",
      expect.any(Object),
    );

    await useModelsStore.getState().loadType("yolo", true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("reports load failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ detail: "models unavailable" }, 500)),
    );

    await useModelsStore.getState().loadType("yolo");

    expect(useModelsStore.getState().error).toContain("models unavailable");
    expect(useModelsStore.getState().loadedTypes.yolo).toBeUndefined();
  });
});
