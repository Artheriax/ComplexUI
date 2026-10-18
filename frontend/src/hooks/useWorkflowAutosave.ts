import { useEffect } from "react";

import {
  AUTOSAVE_DELAY_MS,
  saveLocalWorkflow,
  serialiseWorkflow,
} from "../lib/persistence";
import { useGraphStore } from "../store/useGraphStore";

/**
 * Debounced autosave of the canvas to localStorage. Returns a disposer.
 * Exported standalone so tests can drive it without React.
 */
export function startWorkflowAutosave(): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;

  const unsubscribe = useGraphStore.subscribe((state, previous) => {
    if (state.nodes === previous.nodes && state.edges === previous.edges) {
      return;
    }
    if (timer !== undefined) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = undefined;
      const current = useGraphStore.getState();
      saveLocalWorkflow(serialiseWorkflow(current.nodes, current.edges));
    }, AUTOSAVE_DELAY_MS);
  });

  return () => {
    unsubscribe();
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  };
}

export function useWorkflowAutosave(): void {
  useEffect(() => startWorkflowAutosave(), []);
}
