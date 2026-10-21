// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import type { ModelSummary, ModelTypeSummary } from "../lib/api";
import { useModelsStore } from "../store/useModelsStore";
import type { WidgetParam } from "../types/graph";
import { WidgetField } from "./WidgetField";

afterEach(cleanup);

const widget: WidgetParam = {
  name: "local_model",
  data_type: "ModelSelector",
  default: "",
  model_type: "yolo",
};
const model: ModelSummary = {
  id: "yolo/custom.pt",
  name: "custom.pt",
  path: "models/yolo/custom.pt",
  model_type: "yolo",
  format: "PyTorch",
  size_bytes: 2_048,
  modified_at: "2026-10-20T12:00:00Z",
  mtime_ns: 123,
  fingerprint: "2048:123",
  executable: true,
};
const modelType: ModelTypeSummary = {
  name: "yolo",
  path: "models/yolo",
  count: 1,
  default: true,
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  vi.unstubAllGlobals();
  useModelsStore.setState({
    types: [modelType],
    typesLoaded: true,
    modelsByType: {},
    loadedTypes: {},
    loading: false,
    refreshing: false,
    error: null,
    lastUpdated: null,
  });
});

test("renders the built-in fallback and opens the model picker", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      jsonResponse(String(input).includes("/types") ? [modelType] : [model]),
    ),
  );
  const onChange = vi.fn();

  render(
    createElement(WidgetField, {
      widget,
      value: "",
      onChange,
    }),
  );

  expect(screen.getByText("Built-in fallback")).toBeTruthy();
  fireEvent.click(screen.getByLabelText("Choose model for local_model"));

  expect(await screen.findByText("custom.pt")).toBeTruthy();
  expect(screen.getByText("pickle")).toBeTruthy();
  fireEvent.click(screen.getByText("custom.pt"));

  expect(onChange).toHaveBeenCalledWith("models/yolo/custom.pt");
});

test("warns when the selected local model is missing", () => {
  useModelsStore.setState({
    modelsByType: { yolo: [] },
    loadedTypes: { yolo: true },
  });

  render(
    createElement(WidgetField, {
      widget,
      value: "models/yolo/missing.pt",
      onChange: vi.fn(),
    }),
  );

  const button = screen.getByLabelText("Choose model for local_model");
  expect(button.getAttribute("aria-invalid")).toBe("true");
  expect(screen.getByText("missing.pt")).toBeTruthy();
});
