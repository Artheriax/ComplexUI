import { create } from "zustand";

import { deleteOutput, fetchOutputs } from "../lib/api";
import type { OutputSummary } from "../lib/api";
import { errorMessage } from "../lib/errors";

interface OutputsStore {
  outputs: OutputSummary[];
  outputsLoaded: boolean;
  outputsLoading: boolean;
  outputsError: string | null;

  loadOutputs: (force?: boolean) => Promise<void>;
  removeOutput: (outputId: string) => Promise<boolean>;
  clearOutputError: () => void;
}

export const useOutputsStore = create<OutputsStore>((set, get) => ({
  outputs: [],
  outputsLoaded: false,
  outputsLoading: false,
  outputsError: null,

  loadOutputs: async (force = false) => {
    if (get().outputsLoading || (get().outputsLoaded && !force)) {
      return;
    }
    set({ outputsLoading: true, outputsError: null });
    try {
      set({
        outputs: await fetchOutputs(),
        outputsLoaded: true,
        outputsError: null,
      });
    } catch (cause) {
      set({ outputsError: errorMessage(cause) });
    } finally {
      set({ outputsLoading: false });
    }
  },

  removeOutput: async (outputId) => {
    set({ outputsError: null });
    try {
      await deleteOutput(outputId);
      set((state) => ({
        outputs: state.outputs.filter((output) => output.id !== outputId),
      }));
      return true;
    } catch (cause) {
      set({ outputsError: errorMessage(cause) });
      return false;
    }
  },

  clearOutputError: () => set({ outputsError: null }),
}));
