// @vitest-environment jsdom
import { expect, test } from "vitest";

import { isInteractiveControlTarget } from "./interactiveTarget";

test("blocks node editing for controls and nested control contents", () => {
  const button = document.createElement("button");
  const icon = document.createElement("span");
  button.appendChild(icon);
  document.body.appendChild(button);

  expect(isInteractiveControlTarget(button)).toBe(true);
  expect(isInteractiveControlTarget(icon)).toBe(true);
  expect(isInteractiveControlTarget(document.createElement("input"))).toBe(true);
  expect(
    isInteractiveControlTarget(document.createElement("select")),
  ).toBe(true);
  expect(
    isInteractiveControlTarget(document.createElement("textarea")),
  ).toBe(true);
});

test("allows node editing for ordinary node content", () => {
  const label = document.createElement("span");
  label.textContent = "Node title";
  document.body.appendChild(label);

  expect(isInteractiveControlTarget(label)).toBe(false);
  expect(isInteractiveControlTarget(null)).toBe(false);
});
