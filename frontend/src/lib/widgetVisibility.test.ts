import { expect, test } from "vitest";

import type { WidgetParam } from "../types/graph";
import { isWidgetVisible } from "./widgetVisibility";

const width: WidgetParam = {
  name: "width",
  data_type: "int",
  default: 512,
  visible_when: [{ widget: "size_mode", values: ["Resolution"] }],
};

const custom: WidgetParam = {
  name: "custom_megapixels",
  data_type: "float",
  default: 1,
  visible_when: [
    { widget: "size_mode", values: ["Megapixels"] },
    { widget: "megapixel_preset", values: ["Custom"] },
  ],
};

const upscale: WidgetParam = {
  name: "allow_upscale",
  data_type: "bool",
  default: true,
  visible_when: [
    { widget: "mode", values: ["By Side", "By Megapixels"] },
  ],
};

test("widgets without conditions are visible", () => {
  expect(isWidgetVisible({ name: "mode", data_type: "str", default: "x" }, {})).toBe(
    true,
  );
});

test("scalar conditions show only their matching values", () => {
  expect(
    isWidgetVisible(width, { size_mode: "Resolution" }),
  ).toBe(true);
  expect(isWidgetVisible(width, { size_mode: "Megapixels" })).toBe(false);
});

test("all conditions must match", () => {
  expect(
    isWidgetVisible(custom, {
      size_mode: "Megapixels",
      megapixel_preset: "Custom",
    }),
  ).toBe(true);
  expect(
    isWidgetVisible(custom, {
      size_mode: "Resolution",
      megapixel_preset: "Custom",
    }),
  ).toBe(false);
  expect(
    isWidgetVisible(custom, {
      size_mode: "Megapixels",
      megapixel_preset: "1.00 MP - 1024p",
    }),
  ).toBe(false);
});

test("list values behave as alternatives", () => {
  expect(isWidgetVisible(upscale, { mode: "By Side" })).toBe(true);
  expect(
    isWidgetVisible(upscale, { mode: "By Megapixels" }),
  ).toBe(true);
  expect(isWidgetVisible(upscale, { mode: "By Dimensions" })).toBe(false);
});

test("boolean conditions compare strictly", () => {
  const flag: WidgetParam = {
    name: "details",
    data_type: "int",
    default: 1,
    visible_when: [{ widget: "enabled", values: [false] }],
  };
  expect(isWidgetVisible(flag, { enabled: false })).toBe(true);
  expect(isWidgetVisible(flag, { enabled: true })).toBe(false);
});

test("connected controls and widgets stay visible", () => {
  expect(
    isWidgetVisible(width, { size_mode: "Megapixels" }, new Set(["width"])),
  ).toBe(true);
  expect(
    isWidgetVisible(
      width,
      { size_mode: "Megapixels" },
      new Set(["size_mode"]),
    ),
  ).toBe(true);
});
