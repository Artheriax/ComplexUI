// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import {
  DEFAULT_SETTINGS,
  useSettingsStore,
} from "../store/useSettingsStore";
import { SettingsModal } from "./SettingsModal";

const health = {
  status: "ok",
  version: "0.1.0",
  templates: 84,
  workflows: 21,
  models_root: "C:/repo/models",
  accelerator: "cpu",
  effective_accelerator: "cpu",
  torch: "2.14.1+cpu",
  torch_cuda: false,
  nvidia: false,
  limits: {
    node_timeout_seconds: 600,
    run_timeout_seconds: 3600,
    max_memory_mb: 4096,
  },
  optimizations: {
    attention: {
      configured: "auto",
      backends: ["auto", "sdpa", "sdpa-math"],
      packages: { xformers: null },
    },
    precision: {
      default: "keep",
      fp16_accumulation: false,
      quantization_formats: ["fp8-e4m3"],
      quantized_runtime: false,
    },
    memory: {
      profile: "resident",
      allocator: "native",
      alloc_conf: "expandable_segments:True",
      pinned_memory: false,
    },
    compile: {
      mode: "off",
      cache_dir: "C:/workspace/cache/torchinductor",
    },
  },
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

beforeEach(() => {
  useSettingsStore.setState({
    settings: { ...DEFAULT_SETTINGS },
    isOpen: true,
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      new Response(JSON.stringify(health), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ),
  );
});

test("renders settings categories and updates connection options", () => {
  render(createElement(SettingsModal));

  expect(screen.getByRole("dialog", { name: "Settings" })).toBeTruthy();
  expect(screen.getByRole("tab", { name: "Connections" })).toBeTruthy();

  const showEdges = screen.getByRole("switch", {
    name: "Show connection lines",
  });
  expect(showEdges.getAttribute("aria-checked")).toBe("true");
  fireEvent.click(showEdges);
  expect(useSettingsStore.getState().settings.showEdges).toBe(false);

  const dropSearch = screen.getByRole("switch", {
    name: "Node picker on connection drop",
  });
  expect(dropSearch.getAttribute("aria-checked")).toBe("true");
  fireEvent.click(dropSearch);
  expect(useSettingsStore.getState().settings.connectionDropSearch).toBe(false);

  fireEvent.click(screen.getByRole("radio", { name: "Straight" }));
  expect(useSettingsStore.getState().settings.edgeStyle).toBe("straight");

  fireEvent.click(screen.getByRole("radio", { name: "Single color" }));
  fireEvent.click(screen.getByRole("radio", { name: "Amber" }));
  expect(useSettingsStore.getState().settings.edgeColorMode).toBe("single");
  expect(useSettingsStore.getState().settings.accentColor).toBe("#fbbf24");
});

test("switches to canvas options and toggles grid snapping", () => {
  render(createElement(SettingsModal));

  fireEvent.click(screen.getByRole("tab", { name: "Canvas" }));
  expect(screen.getByRole("tabpanel")).toBeTruthy();

  const snap = screen.getByRole("switch", { name: "Snap to grid" });
  expect(snap.getAttribute("aria-checked")).toBe("false");
  fireEvent.click(snap);
  expect(useSettingsStore.getState().settings.snapToGrid).toBe(true);

  fireEvent.click(screen.getByRole("radio", { name: "Lines" }));
  expect(useSettingsStore.getState().settings.backgroundVariant).toBe("lines");
});

test("controls the system monitor and shows runtime diagnostics", async () => {
  useSettingsStore.setState({
    settings: {
      ...DEFAULT_SETTINGS,
      systemMonitorPosition: { x: 10, y: 20 },
    },
  });
  render(createElement(SettingsModal));

  fireEvent.click(screen.getByRole("tab", { name: "System" }));

  const monitor = screen.getByRole("switch", {
    name: "Show system monitor",
  });
  expect(monitor.getAttribute("aria-checked")).toBe("true");
  fireEvent.click(monitor);
  expect(useSettingsStore.getState().settings.showSystemMonitor).toBe(false);

  fireEvent.click(screen.getByRole("radio", { name: "1s" }));
  expect(useSettingsStore.getState().settings.systemMonitorIntervalMs).toBe(1000);

  fireEvent.click(screen.getByRole("button", { name: /Reset position/i }));
  expect(useSettingsStore.getState().settings.systemMonitorPosition).toBeNull();

  expect(await screen.findByText("2.14.1+cpu")).toBeTruthy();
  expect(screen.getByText("cpu → cpu")).toBeTruthy();
  expect(screen.getByText("auto")).toBeTruthy();
  expect(screen.getByText("resident · native")).toBeTruthy();
  expect(screen.getAllByText("off").length).toBeGreaterThanOrEqual(2);
});

test("controls execution behavior and shows read-only limits", async () => {
  render(createElement(SettingsModal));

  fireEvent.click(screen.getByRole("tab", { name: "Execution" }));

  const autoOpen = screen.getByRole("switch", {
    name: "Open Execution panel on run",
  });
  expect(autoOpen.getAttribute("aria-checked")).toBe("true");
  fireEvent.click(autoOpen);
  expect(useSettingsStore.getState().settings.autoOpenConsole).toBe(false);

  const confirm = screen.getByRole("switch", {
    name: "Confirm canvas clearing",
  });
  fireEvent.click(confirm);
  expect(useSettingsStore.getState().settings.confirmClearCanvas).toBe(false);

  expect(await screen.findByText("600s")).toBeTruthy();
  expect(screen.getByText("3600s")).toBeTruthy();
  expect(screen.getByText("4096 MB")).toBeTruthy();
});

test("resets all preferences from the general tab", () => {
  useSettingsStore.getState().updateSetting("edgeAnimated", true);
  useSettingsStore.getState().updateSetting("showMiniMap", false);
  render(createElement(SettingsModal));

  fireEvent.click(screen.getByRole("tab", { name: "General" }));
  fireEvent.click(screen.getByRole("button", { name: /Reset all settings/i }));

  expect(useSettingsStore.getState().settings).toEqual(DEFAULT_SETTINGS);
});
