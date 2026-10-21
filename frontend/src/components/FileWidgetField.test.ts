// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useDatasetsStore } from "../store/useDatasetsStore";
import type { WidgetParam } from "../types/graph";
import { WidgetField } from "./WidgetField";

afterEach(cleanup);

const widget: WidgetParam = {
  name: "path",
  data_type: "File",
  default: "",
};

beforeEach(() => {
  useDatasetsStore.setState({
    datasets: [
      {
        id: "iris.csv",
        name: "iris.csv",
        path: "data/uploads/iris.csv",
        format: "csv",
        size_bytes: 2_048,
        modified_at: "2026-10-20T12:00:00Z",
      },
    ],
    datasetsLoaded: true,
    datasetsLoading: false,
    datasetsError: null,
    uploading: false,
  });
});

test("shows the basename for a selected workspace file", () => {
  render(
    createElement(WidgetField, {
      widget,
      value: "data/uploads/iris.csv",
      onChange: vi.fn(),
    }),
  );

  expect(screen.getByText("iris.csv")).toBeTruthy();
});

test("picks a dataset and emits its workspace-relative path", () => {
  const onChange = vi.fn();
  render(
    createElement(WidgetField, { widget, value: "", onChange }),
  );

  fireEvent.click(screen.getByLabelText("Choose file for path"));
  fireEvent.click(screen.getByRole("button", { name: /iris\.csv/ }));

  expect(onChange).toHaveBeenCalledWith("data/uploads/iris.csv");
  expect(screen.queryByRole("dialog")).toBeNull();
});
