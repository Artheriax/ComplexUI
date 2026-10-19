// @vitest-environment jsdom
import { cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useGraphStore } from "../store/useGraphStore";
import { useViewportShortcuts } from "./useViewportShortcuts";

const zoomIn = vi.fn();
const zoomOut = vi.fn();

function press(
  init: KeyboardEventInit,
  target: EventTarget = window,
): KeyboardEvent {
  const event = new KeyboardEvent("keydown", {
    bubbles: true,
    cancelable: true,
    ...init,
  });
  target.dispatchEvent(event);
  return event;
}

beforeEach(() => {
  zoomIn.mockReset();
  zoomOut.mockReset();
  useGraphStore.setState({ editingNodeId: null, pendingApproval: null });
  renderHook(() => useViewportShortcuts(zoomIn, zoomOut));
});

afterEach(() => {
  cleanup();
});

describe("useViewportShortcuts", () => {
  it("zooms in and out with ctrl and meta", () => {
    const plus = press({ key: "+", ctrlKey: true });
    expect(plus.defaultPrevented).toBe(true);
    expect(zoomIn).toHaveBeenCalledOnce();

    press({ key: "=", ctrlKey: true });
    expect(zoomIn).toHaveBeenCalledTimes(2);

    const minus = press({ key: "-", ctrlKey: true });
    expect(minus.defaultPrevented).toBe(true);
    expect(zoomOut).toHaveBeenCalledOnce();

    press({ key: "_", ctrlKey: true });
    expect(zoomOut).toHaveBeenCalledTimes(2);

    press({ key: "+", metaKey: true });
    expect(zoomIn).toHaveBeenCalledTimes(3);
  });

  it("ignores text fields and open dialogs", () => {
    const input = document.createElement("input");
    document.body.appendChild(input);

    const typed = press({ key: "+", ctrlKey: true }, input);
    expect(typed.defaultPrevented).toBe(false);
    expect(zoomIn).not.toHaveBeenCalled();

    useGraphStore.setState({ editingNodeId: "node-1" });
    const modal = press({ key: "-", ctrlKey: true });
    expect(modal.defaultPrevented).toBe(false);
    expect(zoomOut).not.toHaveBeenCalled();

    input.remove();
  });

  it("ignores unmodified and alt-modified keys", () => {
    const plain = press({ key: "+" });
    expect(plain.defaultPrevented).toBe(false);

    const alt = press({ key: "-", ctrlKey: true, altKey: true });
    expect(alt.defaultPrevented).toBe(false);

    expect(zoomIn).not.toHaveBeenCalled();
    expect(zoomOut).not.toHaveBeenCalled();
  });
});
