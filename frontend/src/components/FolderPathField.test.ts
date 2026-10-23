// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test, vi } from "vitest";

import { pickFolder } from "../lib/api";
import type { WidgetParam } from "../types/graph";
import { FolderPathField } from "./FolderPathField";

vi.mock("../lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api")>()),
  pickFolder: vi.fn(),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const widget: WidgetParam = {
  name: "folder",
  data_type: "FolderPath",
  default: "",
};

test("accepts a typed folder path", () => {
  const onChange = vi.fn();
  render(
    createElement(FolderPathField, { widget, value: "", onChange }),
  );
  fireEvent.change(screen.getByLabelText("Choose folder for folder"), {
    target: { value: "C:\\images" },
  });
  expect(onChange).toHaveBeenCalledWith("C:\\images");
});

test("browses for a native folder", async () => {
  vi.mocked(pickFolder).mockResolvedValue("C:\\picked");
  const onChange = vi.fn();
  render(
    createElement(FolderPathField, {
      widget,
      value: "C:\\images",
      onChange,
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Browse for folder" }));
  await waitFor(() =>
    expect(onChange).toHaveBeenCalledWith("C:\\picked"),
  );
  expect(pickFolder).toHaveBeenCalledWith("C:\\images");
});
