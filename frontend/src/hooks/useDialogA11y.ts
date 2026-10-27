import { useEffect, useRef } from "react";
import type { RefObject } from "react";

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

const dialogStack: symbol[] = [];

/** True while a modal dialog is mounted and trapping focus. */
export function isDialogOpen(): boolean {
  return dialogStack.length > 0;
}

/**
 * Focus management for modal dialogs: initial focus, Tab trapping, Escape to
 * close and focus restoration. Only the topmost dialog handles keyboard input,
 * so stacked dialogs (code editor + security review) behave predictably.
 */
export function useDialogA11y<T extends HTMLElement>(
  onClose: () => void,
  enabled = true,
): RefObject<T | null> {
  const dialogRef = useRef<T>(null);
  const closeRef = useRef(onClose);

  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    const dialog = dialogRef.current;
    const token = Symbol("dialog");
    dialogStack.push(token);
    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;

    const focusable = (): HTMLElement[] => {
      if (!dialog) {
        return [];
      }
      return Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
    };

    const first = focusable()[0];
    if (first) {
      first.focus();
    } else {
      dialog?.focus();
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (dialogStack[dialogStack.length - 1] !== token) {
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        closeRef.current();
        return;
      }
      if (event.key !== "Tab" || !dialog) {
        return;
      }
      const items = focusable();
      if (items.length === 0) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const firstItem = items[0]!;
      const lastItem = items[items.length - 1]!;
      const active = document.activeElement;
      if (event.shiftKey && (active === firstItem || active === dialog)) {
        event.preventDefault();
        lastItem.focus();
      } else if (!event.shiftKey && active === lastItem) {
        event.preventDefault();
        firstItem.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      document.removeEventListener("keydown", handleKeyDown, true);
      const index = dialogStack.indexOf(token);
      if (index >= 0) {
        dialogStack.splice(index, 1);
      }
      if (previouslyFocused?.isConnected) {
        previouslyFocused.focus();
      }
    };
  }, [enabled]);

  return dialogRef;
}
