// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test, vi } from "vitest";

import type { WidgetParam } from "../types/graph";
import { WidgetField } from "./WidgetField";

afterEach(cleanup);

const floatWidget: WidgetParam = {
  name: "value",
  data_type: "float",
  default: 0.5,
};

test("steps numeric values by the sibling step size", () => {
  const onChange = vi.fn();
  render(
    createElement(WidgetField, {
      widget: floatWidget,
      value: 0.5,
      step: 0.25,
      onChange,
    }),
  );

  fireEvent.click(
    screen.getByRole("button", { name: "Increase value by 0.25" }),
  );
  expect(onChange).toHaveBeenLastCalledWith(0.75);

  fireEvent.click(
    screen.getByRole("button", { name: "Decrease value by 0.25" }),
  );
  expect(onChange).toHaveBeenLastCalledWith(0.25);
});

test("uses integer steps without floating point noise", () => {
  const onChange = vi.fn();
  render(
    createElement(WidgetField, {
      widget: { name: "value", data_type: "int", default: 3 },
      value: 3,
      step: 2,
      onChange,
    }),
  );

  fireEvent.click(
    screen.getByRole("button", { name: "Increase value by 2" }),
  );
  expect(onChange).toHaveBeenLastCalledWith(5);
});

test("renders Literal choices as a select", () => {
  const onChange = vi.fn();
  render(
    createElement(WidgetField, {
      widget: {
        name: "level",
        data_type: "str",
        default: "low",
        options: ["low", "medium", "high"],
      },
      value: "low",
      onChange,
    }),
  );

  const select = screen.getByRole("combobox", { name: "level" });
  expect(select.querySelectorAll("option")).toHaveLength(3);
  fireEvent.change(select, { target: { value: "1" } });
  expect(onChange).toHaveBeenLastCalledWith("medium");
});

test("preserves numeric types for fixed choices", () => {
  const onChange = vi.fn();
  render(
    createElement(WidgetField, {
      widget: {
        name: "count",
        data_type: "int",
        default: 2,
        options: [1, 2, 3],
      },
      value: 2,
      onChange,
    }),
  );

  const select = screen.getByRole("combobox", { name: "count" });
  expect((select as HTMLSelectElement).value).toBe("1");
  fireEvent.change(select, { target: { value: "2" } });
  expect(onChange).toHaveBeenLastCalledWith(3);
});

test("disables fixed choices when linked", () => {
  render(
    createElement(WidgetField, {
      widget: {
        name: "level",
        data_type: "str",
        default: "low",
        options: ["low", "medium", "high"],
      },
      value: "medium",
      connected: true,
      onChange: vi.fn(),
    }),
  );

  expect(
    (screen.getByRole("combobox", { name: /level/ }) as HTMLSelectElement)
      .disabled,
  ).toBe(true);
  expect(screen.getByText("linked")).toBeTruthy();
});

test("disables a widget driven by a connected value", () => {
  render(
    createElement(WidgetField, {
      widget: floatWidget,
      value: 0.5,
      step: 0.5,
      connected: true,
      onChange: vi.fn(),
    }),
  );

  expect(screen.getByText("linked")).toBeTruthy();
  expect(
    (screen.getByRole("spinbutton") as HTMLInputElement).disabled,
  ).toBe(true);
  expect(
    (screen.getByRole("button", {
      name: "Increase value by 0.5",
    }) as HTMLButtonElement).disabled,
  ).toBe(true);
});
