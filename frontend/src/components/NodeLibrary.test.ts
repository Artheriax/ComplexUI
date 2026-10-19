// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import type { TemplateInfo } from "../types/graph";
import { NodeLibrary } from "./NodeLibrary";

afterEach(cleanup);

function makeTemplate(
  name: string,
  title: string,
  category: string,
  description = `${title} description`,
  hidden = false,
): TemplateInfo {
  return {
    name,
    title,
    category,
    description,
    hidden,
    code: "",
    code_hash: "",
    node_schema: {
      node_id: name,
      inputs: [{ name: "df", data_type: "DATAFRAME" }],
      widgets: [],
      outputs: [],
      dependencies: [],
      syntax_errors: [],
    },
  };
}

const templates = [
  makeTemplate("load_csv", "Load CSV", "Data Loaders"),
  makeTemplate("load_iris", "Load Iris", "Data Loaders", undefined, true),
  makeTemplate(
    "tensor_dataset",
    "Synthetic Tensor Dataset",
    "PyTorch",
    undefined,
    true,
  ),
  makeTemplate("drop_missing", "Drop Missing Rows", "Data Cleaning"),
  makeTemplate("knn_train", "KNN Classifier", "Models"),
  makeTemplate("evaluate_metrics", "Evaluate Classifier", "Evaluation"),
  makeTemplate("custom_thing", "Custom Thing", "Other"),
];

beforeEach(() => {
  useGraphStore.setState({
    templates,
    templatesLoaded: true,
    templatesError: null,
  });
});

test("groups templates by category in catalog order", () => {
  const { container } = render(createElement(NodeLibrary));

  const text = container.textContent ?? "";
  const loaders = text.indexOf("Data Loaders");
  const cleaning = text.indexOf("Data Cleaning");
  const models = text.indexOf("Models");
  const evaluation = text.indexOf("Evaluation");
  const other = text.indexOf("Other");
  expect(loaders).toBeGreaterThanOrEqual(0);
  expect(loaders).toBeLessThan(cleaning);
  expect(cleaning).toBeLessThan(models);
  expect(models).toBeLessThan(evaluation);
  expect(evaluation).toBeLessThan(other);

  expect(screen.getByTitle("Expand Models").textContent).toContain("Models");
  expect(screen.getByTitle("Expand Models").textContent).toContain("1");
  expect(screen.getByTitle("Expand Other").textContent).toContain("Other");
});

test("search filters cards and hides empty categories", () => {
  render(createElement(NodeLibrary));

  fireEvent.change(screen.getByPlaceholderText("Search templates…"), {
    target: { value: "csv" },
  });

  expect(screen.getByText("Load CSV")).toBeTruthy();
  expect(screen.queryByText("KNN Classifier")).toBeNull();
  expect(screen.queryByTitle("Collapse Models")).toBeNull();
  expect(screen.queryByText(/No templates match/)).toBeNull();
});

test("hidden templates are excluded from the palette and search", () => {
  const { container } = render(createElement(NodeLibrary));

  expect(screen.queryByText("Load Iris")).toBeNull();
  expect(screen.queryByText("Synthetic Tensor Dataset")).toBeNull();
  expect(container.textContent).not.toContain("PyTorch");
  expect(screen.getByTitle("Expand Data Loaders").textContent).toContain("1");

  fireEvent.change(screen.getByPlaceholderText("Search templates…"), {
    target: { value: "iris" },
  });
  expect(screen.queryByText("Load Iris")).toBeNull();
  expect(screen.getByText(/No templates match “iris”/)).toBeTruthy();

  fireEvent.change(screen.getByPlaceholderText("Search templates…"), {
    target: { value: "tensor" },
  });
  expect(screen.getByText(/No templates match “tensor”/)).toBeTruthy();
});

test("search with no results shows the empty state", () => {
  render(createElement(NodeLibrary));

  fireEvent.change(screen.getByPlaceholderText("Search templates…"), {
    target: { value: "zzz" },
  });

  expect(screen.getByText(/No templates match “zzz”/)).toBeTruthy();
});

test("categories start collapsed, expand on click, and search force-expands them", () => {
  render(createElement(NodeLibrary));

  expect(screen.queryByText("KNN Classifier")).toBeNull();
  expect(screen.getByTitle("Expand Models")).toBeTruthy();

  fireEvent.click(screen.getByTitle("Expand Models"));
  expect(screen.getByText("KNN Classifier")).toBeTruthy();
  expect(screen.getByTitle("Collapse Models")).toBeTruthy();

  fireEvent.click(screen.getByTitle("Collapse Models"));
  expect(screen.queryByText("KNN Classifier")).toBeNull();

  fireEvent.change(screen.getByPlaceholderText("Search templates…"), {
    target: { value: "knn" },
  });
  expect(screen.getByText("KNN Classifier")).toBeTruthy();
});
