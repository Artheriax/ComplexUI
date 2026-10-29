// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { useModelsStore } from "../store/useModelsStore";
import type { ModelSummary } from "../lib/api";
import { ModelsLibrary } from "./ModelsLibrary";

vi.mock("../hooks/useModelsAutoRefresh", () => ({
  useModelsAutoRefresh: () => {},
}));

const model: ModelSummary = {
  id: "diffusion/int8",
  name: "int8.safetensors",
  path: "models/diffusion/int8.safetensors",
  model_type: "diffusion",
  format: "SafeTensors",
  size_bytes: 2 * 1024 ** 3,
  modified_at: "2026-01-01T00:00:00Z",
  mtime_ns: 1,
  fingerprint: "size:mtime",
  executable: false,
  is_directory: false,
  parameter_count: 2_567_459_428,
  quantization: {
    format: "int8_w8a8",
    label: "INT8 W8A8",
    runtime: "comfy_kitchen",
    source: "comfy_quant",
    mode: "pre_quantized",
    weight_bits: 8,
    group_size: 128,
    valid: true,
    warnings: [],
  },
  memory: {
    weights_bytes: 2 * 1024 ** 3,
    quantization_overhead_bytes: 0,
    resident_estimate_bytes: 3 * 1024 ** 3,
    offload_ram_estimate_bytes: 2.5 * 1024 ** 3,
    confidence: "medium",
    assumptions: [],
    unknown: false,
  },
};

afterEach(() => {
  cleanup();
  useModelsStore.setState({
    types: [],
    modelsByType: {},
    loading: false,
    refreshing: false,
    error: null,
    lastUpdated: null,
  });
});

describe("ModelsLibrary metadata", () => {
  test("shows quantization, parameters and memory estimates", () => {
    useModelsStore.setState({
      types: [{ name: "diffusion", path: "models/diffusion", count: 1, default: true }],
      modelsByType: { diffusion: [model] },
      loading: false,
      refreshing: false,
      error: null,
      lastUpdated: null,
    });

    render(<ModelsLibrary />);

    expect(screen.getByText("int8.safetensors")).toBeTruthy();
    expect(screen.getByText("INT8 W8A8")).toBeTruthy();
    expect(screen.getByText(/2\.57B params/)).toBeTruthy();
    expect(screen.getByText("~3.00 GB resident")).toBeTruthy();
    expect(screen.getByText(/2\.50 GB offload RAM/)).toBeTruthy();
  });

  test("marks an invalid quantization runtime", () => {
    useModelsStore.setState({
      types: [{ name: "diffusion", path: "models/diffusion", count: 1, default: true }],
      modelsByType: {
        diffusion: [
          {
            ...model,
            quantization: { ...model.quantization!, valid: false },
          },
        ],
      },
      loading: false,
      refreshing: false,
      error: null,
      lastUpdated: null,
    });

    render(<ModelsLibrary />);

    const tag = screen.getByText("INT8 W8A8");
    expect(tag.className).toContain("text-red-300");
  });
});
