// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import type { SystemMetrics } from "../lib/api";
import {
  DEFAULT_SETTINGS,
  useSettingsStore,
} from "../store/useSettingsStore";

const monitorState = vi.hoisted(() => ({
  metrics: null as SystemMetrics | null,
  error: null as string | null,
  updating: false,
}));

vi.mock("../hooks/useSystemMetrics", () => ({
  useSystemMetrics: () => monitorState,
}));

import { SystemMonitor } from "./SystemMonitor";

const metrics: SystemMetrics = {
  cpu: {
    utilization: 37.5,
    cores: 32,
  },
  memory: {
    total_bytes: 16 * 1024 ** 3,
    used_bytes: 4 * 1024 ** 3,
    utilization: 25,
  },
  gpus: [
    {
      index: 0,
      name: "NVIDIA GeForce RTX 5080",
      utilization: 72,
      vram_total_bytes: 16 * 1024 ** 3,
      vram_used_bytes: 8 * 1024 ** 3,
      temperature_c: 64.4,
    },
  ],
  timestamp: "2026-10-22T00:00:00+00:00",
};

function resetSettings(): void {
  useSettingsStore.setState({
    settings: {
      ...DEFAULT_SETTINGS,
      systemMonitorPosition: { x: 100, y: 100 },
    },
  });
}

beforeEach(() => {
  monitorState.metrics = metrics;
  monitorState.error = null;
  monitorState.updating = false;
  resetSettings();
});

afterEach(cleanup);

test("renders CPU, GPU, VRAM and RAM readings", () => {
  render(createElement(SystemMonitor));

  expect(screen.getByTestId("system-monitor")).toBeTruthy();
  expect(screen.getByText("NVIDIA GeForce RTX 5080")).toBeTruthy();
  expect(
    screen.getByRole("progressbar", { name: "CPU utilization" }).getAttribute(
      "aria-valuenow",
    ),
  ).toBe("38");
  expect(
    screen.getByRole("progressbar", { name: "GPU utilization" }).getAttribute(
      "aria-valuenow",
    ),
  ).toBe("72");
  expect(screen.getByText(/32 cores/)).toBeTruthy();
  expect(screen.getByText(/64°C/)).toBeTruthy();
  expect(screen.getByText(/8.0 GB \/ 16.0 GB/)).toBeTruthy();
  expect(screen.getByText(/4.0 GB \/ 16.0 GB/)).toBeTruthy();
});

test("hides itself through the setting", () => {
  render(createElement(SystemMonitor));

  fireEvent.click(screen.getByRole("button", { name: "Hide system monitor" }));

  expect(useSettingsStore.getState().settings.showSystemMonitor).toBe(false);
});

test("drags and persists its position", () => {
  render(createElement(SystemMonitor));

  const handle = screen.getByRole("button", { name: "Move system monitor" });
  fireEvent.pointerDown(handle, {
    button: 0,
    pointerId: 1,
    clientX: 100,
    clientY: 100,
  });
  fireEvent.pointerMove(window, {
    pointerId: 1,
    clientX: 140,
    clientY: 120,
  });
  fireEvent.pointerUp(window, { pointerId: 1 });

  expect(useSettingsStore.getState().settings.systemMonitorPosition).toEqual({
    x: 140,
    y: 120,
  });
});

test("nudges its position with arrow keys", () => {
  render(createElement(SystemMonitor));

  fireEvent.keyDown(screen.getByRole("button", { name: "Move system monitor" }), {
    key: "ArrowLeft",
  });

  expect(useSettingsStore.getState().settings.systemMonitorPosition).toEqual({
    x: 84,
    y: 100,
  });
});

test("shows an unavailable state when metrics cannot be loaded", () => {
  monitorState.metrics = null;
  monitorState.error = "offline";
  render(createElement(SystemMonitor));

  expect(screen.getByText("System metrics unavailable")).toBeTruthy();
  expect(screen.getByText(/No NVIDIA GPU/)).toBeTruthy();
});
