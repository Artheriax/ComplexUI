// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test } from "vitest";

import {
  DEFAULT_SETTINGS,
  useSettingsStore,
} from "../store/useSettingsStore";
import { SettingsModal } from "./SettingsModal";

afterEach(cleanup);

beforeEach(() => {
  useSettingsStore.setState({
    settings: { ...DEFAULT_SETTINGS },
    isOpen: true,
  });
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

test("resets all preferences from the general tab", () => {
  useSettingsStore.getState().updateSetting("edgeAnimated", true);
  useSettingsStore.getState().updateSetting("showMiniMap", false);
  render(createElement(SettingsModal));

  fireEvent.click(screen.getByRole("tab", { name: "General" }));
  fireEvent.click(screen.getByRole("button", { name: /Reset all settings/i }));

  expect(useSettingsStore.getState().settings).toEqual(DEFAULT_SETTINGS);
});
