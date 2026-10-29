// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test, vi } from "vitest";

import type { RunDetail, RunSummary } from "../lib/api";
import { fetchRun, fetchRuns } from "../lib/api";
import { RunHistoryDialog } from "./RunHistoryDialog";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return {
    ...actual,
    fetchRuns: vi.fn(),
    fetchRun: vi.fn(),
  };
});

afterEach(() => {
  cleanup();
  vi.mocked(fetchRuns).mockReset();
  vi.mocked(fetchRun).mockReset();
});

const summary: RunSummary = {
  run_id: "a".repeat(32),
  status: "failed",
  started_at: "2026-10-20T12:00:00.000Z",
  finished_at: "2026-10-20T12:00:02.000Z",
  duration_ms: 2000,
  node_count: 2,
  completed_count: 1,
  cached_count: 0,
  error_type: "ZeroDivisionError",
};

const detail: RunDetail = {
  ...summary,
  order: ["load", "boom"],
  targets: null,
  node_labels: { load: "Load", boom: "Boom" },
  nodes: [
    {
      node_id: "load",
      label: "Load",
      status: "completed",
      duration_ms: 1200,
      cached: false,
      previews: { value: "ok" },
      rich_outputs: [
        {
          kind: "text",
          source: "print",
          text: "historical console output",
          truncated: false,
        },
      ],
      rich_outputs_truncated: false,
      metrics: { cuda_peak_allocated_bytes: 2 * 1024 ** 3 },
      error: null,
    },
    {
      node_id: "boom",
      label: "Boom",
      status: "error",
      duration_ms: 800,
      cached: false,
      previews: {},
      rich_outputs: [],
      rich_outputs_truncated: false,
      error: {
        message: "division by zero",
        error_type: "ZeroDivisionError",
        node_id: "boom",
        line: 2,
        traceback: "trace",
      },
    },
  ],
  error: {
    message: "division by zero",
    error_type: "ZeroDivisionError",
    node_id: "boom",
    line: 2,
    traceback: "trace",
  },
};

test("lists retained runs and renders detailed outputs and failures", async () => {
  vi.mocked(fetchRuns).mockResolvedValue([summary]);
  vi.mocked(fetchRun).mockResolvedValue(detail);

  render(createElement(RunHistoryDialog, { onClose: vi.fn() }));

  await waitFor(() => {
    expect(screen.getAllByText("2.00 s").length).toBeGreaterThan(0);
  });
  expect(await screen.findByText("Load")).toBeTruthy();
  expect(screen.getByText("Boom")).toBeTruthy();
  expect(screen.getByText("ok")).toBeTruthy();
  expect(screen.getByText("historical console output")).toBeTruthy();
  expect(screen.getByText("VRAM 2.00 GiB")).toBeTruthy();
  expect(screen.getAllByText("division by zero").length).toBeGreaterThan(0);
  expect(screen.getByText("order: load → boom")).toBeTruthy();
  expect(fetchRun).toHaveBeenCalledWith(summary.run_id);
});

test("shows an empty state and reports list failures", async () => {
  vi.mocked(fetchRuns).mockResolvedValueOnce([]);
  render(createElement(RunHistoryDialog, { onClose: vi.fn() }));

  expect(await screen.findByText(/No runs recorded yet/)).toBeTruthy();
  cleanup();

  vi.mocked(fetchRuns).mockRejectedValueOnce(new Error("history offline"));
  render(createElement(RunHistoryDialog, { onClose: vi.fn() }));

  expect(await screen.findByText("history offline")).toBeTruthy();
});
