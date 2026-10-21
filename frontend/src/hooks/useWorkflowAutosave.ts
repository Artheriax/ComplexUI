import { useEffect } from "react";

import { saveWorkspaceSession } from "../lib/api";
import {
  AUTOSAVE_DELAY_MS,
  saveLocalProjectIdentity,
  saveLocalWorkflow,
  saveLocalWorkspace,
} from "../lib/persistence";
import { useGraphStore } from "../store/useGraphStore";

/**
 * Debounced localStorage autosave plus a browser-close workspace session.
 * Returns a disposer. Exported standalone so tests can drive it without React.
 */
export function startWorkflowAutosave(): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;

  const persistLocal = (): void => {
    const current = useGraphStore.getState();
    const windows = current.windows.flatMap((window) => {
      const workflow = current.getWindowDocument(window.id);
      if (!workflow) {
        return [];
      }
      const active = window.id === current.activeWindowId;
      const projectName = active ? current.projectName : window.projectName;
      const projectSlug = active ? current.projectSlug : window.projectSlug;
      return [
        {
          id: window.id,
          title: projectName ?? window.title,
          projectName,
          projectSlug,
          workflow,
        },
      ];
    });
    saveLocalWorkspace({
      version: 1,
      activeWindowId: current.activeWindowId,
      windows,
    });
    saveLocalWorkflow(current.getWorkflowDocument(current.projectName));
    saveLocalProjectIdentity({
      name: current.projectName,
      slug: current.projectSlug,
    });
  };

  const persistSession = (): void => {
    if (typeof window === "undefined") {
      return;
    }
    persistLocal();
    const current = useGraphStore.getState();
    void saveWorkspaceSession(
      current.getWorkflowDocument(current.projectName),
      current.projectSlug,
      current.projectName,
      { keepalive: true },
    );
  };

  const onVisibilityChange = (): void => {
    if (document.visibilityState === "hidden") {
      persistSession();
    }
  };

  const unsubscribe = useGraphStore.subscribe((state, previous) => {
    if (
      state.nodes === previous.nodes &&
      state.edges === previous.edges &&
      state.subgraphs === previous.subgraphs &&
      state.windows === previous.windows &&
      state.activeWindowId === previous.activeWindowId
    ) {
      return;
    }
    if (timer !== undefined) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = undefined;
      persistLocal();
    }, AUTOSAVE_DELAY_MS);
  });

  window.addEventListener("pagehide", persistSession);
  document.addEventListener("visibilitychange", onVisibilityChange);

  return () => {
    unsubscribe();
    if (timer !== undefined) {
      clearTimeout(timer);
    }
    window.removeEventListener("pagehide", persistSession);
    document.removeEventListener("visibilitychange", onVisibilityChange);
  };
}

export function useWorkflowAutosave(): void {
  useEffect(() => startWorkflowAutosave(), []);
}
