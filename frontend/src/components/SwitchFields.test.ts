// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test, vi } from "vitest";

import type { WidgetParam } from "../types/graph";
import { SwitchFields } from "./SwitchFields";

afterEach(cleanup);

const widgets: WidgetParam[] = [
  { name: "input_1", data_type: "ANY", default: null },
  { name: "input_2", data_type: "ANY", default: null },
  { name: "input_3", data_type: "ANY", default: null },
  { name: "input_4", data_type: "ANY", default: null },
  { name: "input_5", data_type: "ANY", default: null },
  { name: "choices", data_type: "str", default: "Input 1|Input 2|Input 3|Input 4|Input 5" },
  { name: "selected", data_type: "int", default: 0 },
];

function renderFields(
  values: Record<string, unknown> = {},
  connected = new Set<string>(),
  onChange = vi.fn(),
) {
  render(
    createElement(SwitchFields, {
      widgets,
      values: {
        input_1: null,
        input_2: null,
        input_3: null,
        input_4: null,
        input_5: null,
        choices: "Alpha|Beta|Gamma|Delta|Epsilon",
        selected: 1,
        ...values,
      },
      connected,
      onChange,
      renderHandle: () => null,
    }),
  );
  return onChange;
}

test("shows named inputs and the selected input warning", () => {
  renderFields({}, new Set(["input_1"]));
  expect(
    (screen.getByLabelText("Input 1 name") as HTMLInputElement).value,
  ).toBe("Alpha");
  expect(
    (screen.getByLabelText("Switch selection") as HTMLSelectElement).value,
  ).toBe("1");
  expect(
    screen.getByText("The selected input is not connected."),
  ).toBeTruthy();
});

test("renames inputs and changes the selection", () => {
  const onChange = renderFields({}, new Set(["input_1", "input_2"]));
  fireEvent.change(screen.getByLabelText("Input 1 name"), {
    target: { value: "First" },
  });
  expect(onChange).toHaveBeenCalledWith(
    "choices",
    "First|Beta|Gamma|Delta|Epsilon",
  );

  fireEvent.change(screen.getByLabelText("Switch selection"), {
    target: { value: "0" },
  });
  expect(onChange).toHaveBeenCalledWith("selected", 0);
});

test("hides labels behind linked choices", () => {
  renderFields({}, new Set(["choices", "input_1"]));
  expect(
    screen.getByLabelText("Input 1 name").hasAttribute("disabled"),
  ).toBe(true);
});
