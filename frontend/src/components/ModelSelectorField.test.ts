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
  is_directory: false,
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

test("searches included model types alongside the primary type", async () => {
  const vaeModel: ModelSummary = {
    ...model,
    id: "vae/autoencoder.safetensors",
    name: "autoencoder.safetensors",
    path: "models/vae/autoencoder.safetensors",
    model_type: "vae",
    format: "SafeTensors",
    executable: false,
  };
  const folder: ModelSummary = {
    ...model,
    id: "diffusion/example-flux",
    name: "example-flux",
    path: "models/diffusion/example-flux",
    model_type: "diffusion",
    format: "Folder",
    executable: false,
    is_directory: true,
  };
  const diffusionType: ModelTypeSummary = {
    name: "diffusion",
    path: "models/diffusion",
    count: 1,
    default: true,
  };
  const vaeType: ModelTypeSummary = {
    name: "vae",
    path: "models/vae",
    count: 1,
    default: true,
  };
  const includedWidget: WidgetParam = {
    ...widget,
    name: "vae",
    model_type: "vae",
    model_include: ["diffusion"],
    model_select: "any",
    model_required: true,
  };
  useModelsStore.setState({
    types: [vaeType, diffusionType],
    typesLoaded: true,
    modelsByType: { vae: [vaeModel], diffusion: [folder] },
    loadedTypes: { vae: true, diffusion: true },
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/types")) {
        return jsonResponse([vaeType, diffusionType]);
      }
      return jsonResponse(url.includes("type=vae") ? [vaeModel] : [folder]);
    }),
  );
  const onChange = vi.fn();

  render(
    createElement(WidgetField, {
      widget: includedWidget,
      value: "",
      onChange,
    }),
  );

  fireEvent.click(screen.getByLabelText("Choose model for vae"));

  expect(await screen.findByText("autoencoder.safetensors")).toBeTruthy();
  expect(screen.getByText("example-flux")).toBeTruthy();
  fireEvent.click(screen.getByText("example-flux"));

  expect(onChange).toHaveBeenCalledWith("models/diffusion/example-flux");
});

test("required folder selectors hide the fallback and filter files", async () => {
  const diffusionType: ModelTypeSummary = {
    name: "diffusion",
    path: "models/diffusion",
    count: 2,
    default: true,
  };
  const folder: ModelSummary = {
    id: "diffusion/example-flux",
    name: "example-flux",
    path: "models/diffusion/example-flux",
    model_type: "diffusion",
    format: "Folder",
    size_bytes: 4_096,
    modified_at: "2026-10-20T12:00:00Z",
    mtime_ns: 456,
    fingerprint: "folder:abc",
    executable: false,
    is_directory: true,
  };
  const file: ModelSummary = {
    ...model,
    id: "diffusion/single.safetensors",
    name: "single.safetensors",
    path: "models/diffusion/single.safetensors",
    model_type: "diffusion",
    format: "SafeTensors",
    executable: false,
  };
  const requiredWidget: WidgetParam = {
    ...widget,
    name: "diffusion_model",
    model_type: "diffusion",
    model_select: "folder",
    model_required: true,
  };
  useModelsStore.setState({
    types: [diffusionType],
    typesLoaded: true,
    modelsByType: { diffusion: [file, folder] },
    loadedTypes: { diffusion: true },
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      jsonResponse(
        String(input).includes("/types") ? [diffusionType] : [file, folder],
      ),
    ),
  );
  const onChange = vi.fn();

  render(
    createElement(WidgetField, {
      widget: requiredWidget,
      value: "",
      onChange,
    }),
  );

  expect(screen.getByText("Select model…")).toBeTruthy();
  expect(screen.getByTitle("A local model is required")).toBeTruthy();
  fireEvent.click(screen.getByLabelText("Choose model for diffusion_model"));

  expect(await screen.findByText("example-flux")).toBeTruthy();
  expect(screen.getByText("folder")).toBeTruthy();
  expect(screen.queryByText("single.safetensors")).toBeNull();
  expect(screen.queryByText("Built-in fallback")).toBeNull();
  fireEvent.click(screen.getByText("example-flux"));

  expect(onChange).toHaveBeenCalledWith("models/diffusion/example-flux");
});
