// @vitest-environment jsdom
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test, vi } from "vitest";

import { pickFolder } from "../lib/api";
import type { WidgetParam } from "../types/graph";
import { SaveMediaFields } from "./SaveMediaFields";

vi.mock("../lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api")>()),
  pickFolder: vi.fn(),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const widgets: WidgetParam[] = [
  { name: "folder", data_type: "str", default: "%date%" },
  { name: "filename", data_type: "str", default: "%model%_%counter%" },
  { name: "model_name", data_type: "str", default: "image" },
  {
    name: "format",
    data_type: "str",
    default: "png",
    options: ["png", "jpg", "webp", "bmp"],
  },
  { name: "quality", data_type: "int", default: 95 },
  { name: "compress_level", data_type: "int", default: 6 },
  { name: "optimize", data_type: "bool", default: true },
  { name: "save_nonce", data_type: "int", default: 0 },
];

function values(overrides: Record<string, unknown> = {}) {
  return {
    folder: "%date%",
    filename: "%model%_%counter%",
    model_name: "model",
    format: "png",
    quality: 95,
    compress_level: 6,
    optimize: true,
    save_nonce: 0,
    ...overrides,
  };
}

function renderFields(
  overrides: Record<string, unknown> = {},
  onChange = vi.fn(),
) {
  render(
    createElement(SaveMediaFields, {
      widgets,
      values: values(overrides),
      connected: new Set<string>(),
      onChange,
      renderHandle: () => null,
      upstreamPreview: null,
    }),
  );
  return onChange;
}

test("shows the default output preview", () => {
  renderFields();
  expect(
    screen.getByText(/outputs\/\d{4}-\d{2}-\d{2}\/model_001\.png/),
  ).toBeTruthy();
});

test("inserts filename tokens at the cursor", () => {
  const onChange = renderFields({ filename: "" });
  fireEvent.click(screen.getByRole("button", { name: "+ Counter" }));
  expect(onChange).toHaveBeenCalledWith("filename", "%counter%");
});

test("adds a date folder token", () => {
  const onChange = renderFields({ folder: "" });
  fireEvent.click(screen.getByRole("button", { name: "+ Date folder" }));
  expect(onChange).toHaveBeenCalledWith("folder", "%date%");
});

test("updates format buttons and reset defaults", () => {
  const onChange = renderFields();
  fireEvent.click(screen.getByRole("button", { name: "Use JPG format" }));
  expect(onChange).toHaveBeenCalledWith("format", "jpg");

  fireEvent.click(screen.getByRole("button", { name: "Reset" }));
  expect(onChange).toHaveBeenCalledWith("folder", "%date%");
  expect(onChange).toHaveBeenCalledWith("filename", "%model%_%counter%");
  expect(onChange).toHaveBeenCalledWith("format", "png");
});

test("browses for a native folder and stores the selected path", async () => {
  vi.mocked(pickFolder).mockResolvedValue("/tmp/chosen");
  const onChange = renderFields({ folder: "" });
  fireEvent.click(screen.getByRole("button", { name: "Browse" }));
  await waitFor(() =>
    expect(onChange).toHaveBeenCalledWith("folder", "/tmp/chosen"),
  );
});

test("warns when the counter token is missing", () => {
  renderFields({ filename: "image" });
  expect(
    screen.getByText("Filename template must include %counter%."),
  ).toBeTruthy();
});

test("opens advanced settings and keeps metadata tokens disabled", () => {
  renderFields();
  expect(
    screen.getByRole("button", { name: "+ Seed" }).hasAttribute("disabled"),
  ).toBe(true);
  fireEvent.click(
    screen.getByRole("button", { name: "Save Media settings" }),
  );
  expect(screen.getByText("model_name")).toBeTruthy();
  expect(screen.getByText("quality")).toBeTruthy();
});
