import { expect, test } from "vitest";

import {
  filterTemplates,
  groupTemplates,
  orderTemplates,
  searchTemplates,
} from "./templateSearch";
import type { TemplateInfo } from "../types/graph";

function makeTemplate(
  name: string,
  title: string,
  category: string,
  hidden = false,
): TemplateInfo {
  return {
    name,
    title,
    category,
    description: `${title} description`,
    hidden,
    code: "",
    code_hash: "",
    node_schema: {
      node_id: name,
      inputs: [],
      widgets: [],
      outputs: [],
      dependencies: [],
      syntax_errors: [],
    },
  };
}

const templates = [
  makeTemplate("knn_train", "KNN Classifier", "Models"),
  makeTemplate("load_csv", "Load CSV", "Data Loaders"),
  makeTemplate("secret", "Hidden Template", "Models", true),
  makeTemplate("custom", "Custom Thing", "Other"),
  makeTemplate("drop_rows", "Drop Rows", "Data Cleaning"),
];

test("filters hidden templates and matches query fields", () => {
  expect(filterTemplates(templates, "").map((item) => item.name)).toEqual([
    "knn_train",
    "load_csv",
    "custom",
    "drop_rows",
  ]);
  expect(filterTemplates(templates, "csv").map((item) => item.name)).toEqual([
    "load_csv",
  ]);
  expect(filterTemplates(templates, "models").map((item) => item.name)).toEqual([
    "knn_train",
  ]);
  expect(filterTemplates(templates, "hidden")).toEqual([]);
});

test("orders templates by catalog category and groups them", () => {
  expect(
    orderTemplates(filterTemplates(templates, "")).map((item) => item.category),
  ).toEqual([
    "Data Loaders",
    "Data Cleaning",
    "Models",
    "Other",
  ]);

  const groups = groupTemplates(templates, "");
  expect(groups.map((group) => group.category)).toEqual([
    "Data Loaders",
    "Data Cleaning",
    "Models",
    "Other",
  ]);
  expect(groups[2]?.items.map((item) => item.name)).toEqual(["knn_train"]);
});

test("search combines filtering, ordering and grouping", () => {
  expect(searchTemplates(templates, "load").map((item) => item.name)).toEqual([
    "load_csv",
  ]);
  expect(groupTemplates(templates, "zzz")).toEqual([]);
});
