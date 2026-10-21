import { useEffect } from "react";

import { isTextEntryTarget } from "./useGraphShortcuts";
import { useModelsStore } from "../store/useModelsStore";

const DEFAULT_INTERVAL_MS = 2500;

/**
 * Refresh model metadata while a model view is mounted. Polling is paused
 * while the document is hidden, and pressing R refreshes immediately unless
 * focus is inside a text field.
 */
export function useModelsAutoRefresh(
  modelType: string | null,
  enabled = true,
  intervalMs = DEFAULT_INTERVAL_MS,
): void {
  const refresh = useModelsStore((state) => state.refresh);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    let disposed = false;

    const refreshNow = () => {
      if (!disposed && document.visibilityState === "visible") {
        void refresh(modelType);
      }
    };

    refreshNow();
    const interval = window.setInterval(refreshNow, intervalMs);
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        refreshNow();
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.key.toLowerCase() !== "r" ||
        isTextEntryTarget(event.target)
      ) {
        return;
      }
      event.preventDefault();
      void refresh(modelType);
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      disposed = true;
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [enabled, intervalMs, modelType, refresh]);
}
