import { useEffect } from "react";

import { useGraphStore } from "../store/useGraphStore";
import { isTextEntryTarget } from "./useGraphShortcuts";

export function useViewportShortcuts(
  zoomIn: () => void,
  zoomOut: () => void,
): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey) {
        return;
      }
      if (isTextEntryTarget(event.target)) {
        return;
      }
      const state = useGraphStore.getState();
      if (state.editingNodeId !== null || state.pendingApproval !== null) {
        return;
      }
      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        zoomIn();
        return;
      }
      if (event.key === "-" || event.key === "_") {
        event.preventDefault();
        zoomOut();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [zoomIn, zoomOut]);
}
