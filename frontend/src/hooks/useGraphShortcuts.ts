import { useEffect } from "react";

import { useGraphStore } from "../store/useGraphStore";
import { isFrameNode, isRunnableNode } from "../types/graph";

export function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return Boolean(
    target.closest("input, textarea, select, [contenteditable='true']"),
  );
}

/**
 * Canvas-level keyboard shortcuts. Ignored while a code/diff modal is open or
 * when the event originates from a text field (including Monaco's textarea).
 */
export function useGraphShortcuts(): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey)) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "s") {
        event.preventDefault();
        useGraphStore.getState().requestSave();
        return;
      }
      if (isTextEntryTarget(event.target)) {
        return;
      }
      const state = useGraphStore.getState();
      if (state.editingNodeId !== null || state.pendingApproval !== null) {
        return;
      }
      if (key === "z" && !event.shiftKey) {
        event.preventDefault();
        state.undo();
        return;
      }
      if ((key === "z" && event.shiftKey) || key === "y") {
        event.preventDefault();
        state.redo();
        return;
      }
      if (key === "a" && !event.shiftKey) {
        if (state.nodes.length > 0) {
          event.preventDefault();
          state.selectAllNodes();
        }
        return;
      }
      if (key === "g") {
        if (event.shiftKey) {
          if (state.nodes.some((node) => node.selected && isFrameNode(node))) {
            event.preventDefault();
            state.ungroupSelection();
          }
          return;
        }
        const groupable = state.nodes.filter(
          (node) =>
            node.selected && isRunnableNode(node) && node.parentId === undefined,
        );
        if (groupable.length >= 2) {
          event.preventDefault();
          state.groupSelection();
        }
        return;
      }
      if (key === "d") {
        if (state.nodes.some((node) => node.selected)) {
          event.preventDefault();
          void state.duplicateSelection();
        }
        return;
      }
      if (key === "c") {
        if (state.nodes.some((node) => node.selected)) {
          event.preventDefault();
          state.copySelection();
        }
        return;
      }
      if (key === "x") {
        if (state.nodes.some((node) => node.selected)) {
          event.preventDefault();
          state.cutSelection();
        }
        return;
      }
      if (key === "v" && state.clipboard !== null) {
        event.preventDefault();
        void state.pasteClipboard();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
