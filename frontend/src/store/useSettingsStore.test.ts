// @vitest-environment jsdom
import { beforeEach, expect, test } from "vitest";

import {
  DEFAULT_SETTINGS,
  SETTINGS_STORAGE_KEY,
  sanitiseSettings,
  useSettingsStore,
} from "./useSettingsStore";

beforeEach(() => {
  window.localStorage.clear();
  useSettingsStore.setState({
    settings: { ...DEFAULT_SETTINGS },
    isOpen: false,
  });
});

test("exposes default settings", () => {
  expect(useSettingsStore.getState().settings).toEqual(DEFAULT_SETTINGS);
});

test("updates settings, clamps ranges and persists them", () => {
  useSettingsStore.getState().updateSetting("edgeWidth", 99);
  useSettingsStore.getState().updateSetting("edgeStyle", "straight");
  useSettingsStore.getState().updateSetting("showEdges", false);
  useSettingsStore.getState().updateSetting("connectionDropSearch", false);

  const settings = useSettingsStore.getState().settings;
  expect(settings.edgeWidth).toBe(6);
  expect(settings.edgeStyle).toBe("straight");
  expect(settings.showEdges).toBe(false);
  expect(settings.connectionDropSearch).toBe(false);
  expect(window.localStorage.getItem(SETTINGS_STORAGE_KEY)).toContain(
    '"edgeStyle":"straight"',
  );
});

test("opens and closes the settings dialog without persisting that state", () => {
  useSettingsStore.getState().openSettings();
  expect(useSettingsStore.getState().isOpen).toBe(true);
  useSettingsStore.getState().closeSettings();
  expect(useSettingsStore.getState().isOpen).toBe(false);
  expect(window.localStorage.getItem(SETTINGS_STORAGE_KEY)).not.toContain(
    '"isOpen":true',
  );
});

test("resets settings to their defaults", () => {
  useSettingsStore.getState().updateSetting("edgeAnimated", true);
  useSettingsStore.getState().updateSetting("showMiniMap", false);
  useSettingsStore.getState().resetSettings();
  expect(useSettingsStore.getState().settings).toEqual(DEFAULT_SETTINGS);
});

test("sanitises monitor and execution settings", () => {
  const settings = sanitiseSettings({
    showSystemMonitor: false,
    systemMonitorIntervalMs: 1000,
    systemMonitorPosition: { x: 12.5, y: 400 },
    autoOpenConsole: false,
    confirmClearCanvas: false,
  });

  expect(settings.showSystemMonitor).toBe(false);
  expect(settings.systemMonitorIntervalMs).toBe(1000);
  expect(settings.systemMonitorPosition).toEqual({ x: 12.5, y: 400 });
  expect(settings.autoOpenConsole).toBe(false);
  expect(settings.confirmClearCanvas).toBe(false);

  const invalid = sanitiseSettings({
    systemMonitorIntervalMs: 1234,
    systemMonitorPosition: { x: "left", y: 400 },
  });
  expect(invalid.systemMonitorIntervalMs).toBe(2000);
  expect(invalid.systemMonitorPosition).toBeNull();
});

test("sanitises corrupted persisted values", () => {
  expect(
    sanitiseSettings({
      showEdges: "yes",
      edgeStyle: "wavy",
      edgeWidth: 42,
      edgeColorMode: "rainbow",
      accentColor: "red",
      edgeAnimated: 1,
      showArrowheads: true,
      handleSize: "gigantic",
      backgroundVariant: "plaid",
      gridSpacing: 1,
      snapToGrid: false,
      showMiniMap: false,
      connectionDropSearch: "sometimes",
    }),
  ).toEqual({
    ...DEFAULT_SETTINGS,
    edgeWidth: 6,
    showArrowheads: true,
    gridSpacing: 10,
    snapToGrid: false,
    showMiniMap: false,
  });
});
