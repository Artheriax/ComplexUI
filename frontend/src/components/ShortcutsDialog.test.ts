// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, expect, test } from "vitest";

import { useUiStore } from "../store/useUiStore";
import { ShortcutsDialog } from "./ShortcutsDialog";

afterEach(cleanup);

beforeEach(() => {
  useUiStore.setState({ shortcutsOpen: true });
});

test("lists canvas and workflow shortcuts", () => {
  render(createElement(ShortcutsDialog));

  const dialog = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
  expect(dialog).toBeTruthy();
  expect(screen.getByText("Double-click empty canvas")).toBeTruthy();
  expect(screen.getByText("Ctrl+Y / Ctrl+Shift+Z")).toBeTruthy();
  expect(screen.getByText("Save the active workflow")).toBeTruthy();
});

test("closes from the header button", () => {
  render(createElement(ShortcutsDialog));

  fireEvent.click(
    screen.getByRole("button", { name: "Close keyboard shortcuts" }),
  );

  expect(useUiStore.getState().shortcutsOpen).toBe(false);
});
