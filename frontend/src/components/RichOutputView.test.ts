// @vitest-environment jsdom
import { cleanup, render, waitFor } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test, vi } from "vitest";

import type { RichOutput } from "../types/graph";
import { RichOutputView } from "./RichOutputView";

const plotly = vi.hoisted(() => ({
  newPlot: vi.fn().mockResolvedValue(undefined),
  purge: vi.fn(),
}));

vi.mock("plotly.js-dist-min", () => ({ default: plotly }));

afterEach(() => {
  cleanup();
  plotly.newPlot.mockClear();
  plotly.purge.mockClear();
});

test("renders console text in a monospaced block with a truncation notice", () => {
  const output: RichOutput = {
    kind: "text",
    source: "print",
    text: "hello\nworld",
    truncated: true,
  };

  const { container } = render(createElement(RichOutputView, { output }));

  expect(container.textContent).toContain("hello\nworld");
  expect(container.textContent).toContain("Text output truncated");
  expect(container.querySelector("pre")).not.toBeNull();
});

test("renders table rows and truncation notice", () => {
  const output: RichOutput = {
    kind: "table",
    source: "display",
    columns: ["name", "score"],
    rows: [
      ["ada", 1.5],
      ["grace", 2.5],
    ],
    total_rows: 100,
    total_cols: 2,
    truncated: true,
  };

  const { container } = render(createElement(RichOutputView, { output }));

  expect(container.textContent).toContain("100 rows x 2 cols");
  expect(container.textContent).toContain("ada");
  expect(container.textContent).toContain("grace");
  expect(container.textContent).toContain("Table preview truncated");
});

test("sanitizes HTML rich output", () => {
  const output: RichOutput = {
    kind: "html",
    source: "display",
    html: '<b>safe</b><script>alert("xss")</script>',
    truncated: false,
  };

  const { container } = render(createElement(RichOutputView, { output }));

  expect(container.innerHTML).toContain("<b>safe</b>");
  expect(container.innerHTML).not.toContain("<script");
});

test("renders Plotly static inline and interactive in the panel", async () => {
  const output: RichOutput = {
    kind: "plotly",
    source: "plotly",
    figure: {
      data: [{ type: "scatter", x: [1, 2], y: [3, 4] }],
      layout: { title: { text: "Demo" } },
    },
  };

  const { rerender } = render(createElement(RichOutputView, { output, compact: true }));

  await waitFor(() => expect(plotly.newPlot).toHaveBeenCalled());
  expect(plotly.newPlot.mock.calls[0]?.[3]).toMatchObject({
    staticPlot: true,
    displayModeBar: false,
  });

  rerender(createElement(RichOutputView, { output, interactive: true }));
  await waitFor(() => expect(plotly.newPlot).toHaveBeenCalledTimes(2));
  expect(plotly.newPlot.mock.calls[1]?.[3]).toMatchObject({
    staticPlot: false,
    displayModeBar: true,
  });
});

test("renders base64 images with their mime type", () => {
  const output: RichOutput = {
    kind: "image",
    source: "matplotlib",
    mime: "image/png",
    data: "iVBORw0KGgo=",
    alt: "Plot output",
  };

  const { container } = render(createElement(RichOutputView, { output }));

  const image = container.querySelector("img");
  expect(image?.getAttribute("src")).toBe("data:image/png;base64,iVBORw0KGgo=");
  expect(image?.getAttribute("alt")).toBe("Plot output");
});
