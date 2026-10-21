import { create } from "zustand";

import {
  fetchModelTypes,
  fetchModels,
} from "../lib/api";
import type { ModelSummary, ModelTypeSummary } from "../lib/api";

interface ModelsStore {
  types: ModelTypeSummary[];
  typesLoaded: boolean;
  modelsByType: Record<string, ModelSummary[]>;
  loadedTypes: Record<string, boolean>;
  loading: boolean;
  refreshing: boolean;
  error: string | null;
  lastUpdated: number | null;

  loadTypes: (force?: boolean) => Promise<void>;
  loadType: (modelType: string, force?: boolean) => Promise<void>;
  loadLibrary: (force?: boolean) => Promise<void>;
  refresh: (modelType?: string | null) => Promise<void>;
  ensureType: (modelType: string) => Promise<void>;
  hasModel: (modelType: string, path: string) => boolean;
  clearError: () => void;
}

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

const inFlightTypes = new Set<string>();
let libraryInFlight = false;

function groupModels(models: ModelSummary[]): Record<string, ModelSummary[]> {
  const grouped: Record<string, ModelSummary[]> = {};
  for (const model of models) {
    grouped[model.model_type] = [
      ...(grouped[model.model_type] ?? []),
      model,
    ];
  }
  return grouped;
}

export const useModelsStore = create<ModelsStore>((set, get) => ({
  types: [],
  typesLoaded: false,
  modelsByType: {},
  loadedTypes: {},
  loading: false,
  refreshing: false,
  error: null,
  lastUpdated: null,

  loadTypes: async (force = false) => {
    if (get().typesLoaded && !force) {
      return;
    }
    try {
      const types = await fetchModelTypes();
      set({ types, typesLoaded: true, error: null });
    } catch (cause) {
      set({ error: errorMessage(cause) });
    }
  },

  loadType: async (modelType, force = false) => {
    if (!modelType || (!force && get().loadedTypes[modelType])) {
      return;
    }
    if (inFlightTypes.has(modelType)) {
      return;
    }
    inFlightTypes.add(modelType);
    set({ refreshing: true, error: null });
    try {
      const models = await fetchModels(modelType);
      set((state) => ({
        modelsByType: { ...state.modelsByType, [modelType]: models },
        loadedTypes: { ...state.loadedTypes, [modelType]: true },
        lastUpdated: Date.now(),
      }));
    } catch (cause) {
      set({ error: errorMessage(cause) });
    } finally {
      inFlightTypes.delete(modelType);
      set({ refreshing: inFlightTypes.size > 0 });
    }
  },

  loadLibrary: async (force = false) => {
    if (libraryInFlight) {
      return;
    }
    const loaded = get().typesLoaded && Object.keys(get().loadedTypes).length > 0;
    if (loaded && !force) {
      return;
    }
    libraryInFlight = true;
    set({ loading: true, error: null });
    try {
      const [types, models] = await Promise.all([
        fetchModelTypes(),
        fetchModels(),
      ]);
      const modelsByType = groupModels(models);
      const loadedTypes = Object.fromEntries(
        types.map((modelType) => [modelType.name, true]),
      );
      set({
        types,
        typesLoaded: true,
        modelsByType,
        loadedTypes,
        lastUpdated: Date.now(),
      });
    } catch (cause) {
      set({ error: errorMessage(cause) });
    } finally {
      libraryInFlight = false;
      set({ loading: false });
    }
  },

  refresh: async (modelType = null) => {
    await get().loadTypes(true);
    if (modelType) {
      await get().loadType(modelType, true);
    } else {
      await get().loadLibrary(true);
    }
  },

  ensureType: async (modelType) => {
    await get().loadType(modelType);
  },

  hasModel: (modelType, path) => {
    return (get().modelsByType[modelType] ?? []).some(
      (model) => model.path === path,
    );
  },

  clearError: () => set({ error: null }),
}));
