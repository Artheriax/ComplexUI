// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useDialogA11y } from "./useDialogA11y";

afterEach(cleanup);

function TestDialog({
  label,
  onClose,
  children,
}: {
  label: string;
  onClose: () => void;
  children?: ReactNode;
}) {
  const ref = useDialogA11y<HTMLDivElement>(onClose);
  return createElement(
    "div",
    { ref, role: "dialog", "aria-label": label, tabIndex: -1 },
    createElement("button", { type: "button" }, "first"),
    createElement("button", { type: "button" }, "last"),
    children,
  );
}

function Harness({ open, onClose }: { open: boolean; onClose: () => void }) {
  return createElement(
    "div",
    null,
    createElement("button", { type: "button" }, "trigger"),
    open
      ? createElement(TestDialog, { label: "Test dialog", onClose })
      : null,
  );
}

describe("useDialogA11y", () => {
  it("focuses the first control and closes on Escape", () => {
    const onClose = vi.fn();
    render(createElement(TestDialog, { label: "Test dialog", onClose }));

    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "first" }),
    );

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("wraps Tab and Shift+Tab inside the dialog", () => {
    render(
      createElement(TestDialog, { label: "Test dialog", onClose: () => {} }),
    );
    const first = screen.getByRole("button", { name: "first" });
    const last = screen.getByRole("button", { name: "last" });

    last.focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(document.activeElement).toBe(first);

    first.focus();
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(last);
  });

  it("only lets the topmost dialog consume Escape", () => {
    const outerClose = vi.fn();
    const innerClose = vi.fn();
    render(
      createElement(
        "div",
        null,
        createElement(TestDialog, {
          key: "outer",
          label: "Outer",
          onClose: outerClose,
        }),
        createElement(TestDialog, {
          key: "inner",
          label: "Inner",
          onClose: innerClose,
        }),
      ),
    );

    fireEvent.keyDown(document, { key: "Escape" });
    expect(innerClose).toHaveBeenCalledTimes(1);
    expect(outerClose).not.toHaveBeenCalled();
  });

  it("restores focus when it unmounts", () => {
    const { rerender } = render(
      createElement(Harness, { open: false, onClose: () => {} }),
    );
    const trigger = screen.getByRole("button", { name: "trigger" });
    trigger.focus();

    rerender(createElement(Harness, { open: true, onClose: () => {} }));
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "first" }),
    );

    rerender(createElement(Harness, { open: false, onClose: () => {} }));
    expect(document.activeElement).toBe(trigger);
  });
});
