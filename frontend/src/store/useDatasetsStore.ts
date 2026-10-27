import { create } from "zustand";

import {
  deleteDataset as deleteDatasetApi,
  fetchDatasets,
  uploadDataset,
} from "../lib/api";
import type { DatasetSummary } from "../lib/api";
import { errorMessage } from "../lib/errors";

interface DatasetStore {
  datasets: DatasetSummary[];
  datasetsLoaded: boolean;
  datasetsLoading: boolean;
  datasetsError: string | null;
  uploading: boolean;

  loadDatasets: (force?: boolean) => Promise<void>;
  uploadFiles: (files: File[]) => Promise<DatasetSummary[]>;
  removeDataset: (datasetId: string) => Promise<boolean>;
  clearDatasetError: () => void;
}

export const useDatasetsStore = create<DatasetStore>((set, get) => ({
  datasets: [],
  datasetsLoaded: false,
  datasetsLoading: false,
  datasetsError: null,
  uploading: false,

  loadDatasets: async (force = false) => {
    if (get().datasetsLoading || (get().datasetsLoaded && !force)) {
      return;
    }
    set({ datasetsLoading: true, datasetsError: null });
    try {
      const datasets = await fetchDatasets();
      set({ datasets, datasetsLoaded: true });
    } catch (cause) {
      set({ datasetsError: errorMessage(cause) });
    } finally {
      set({ datasetsLoading: false });
    }
  },

  uploadFiles: async (files) => {
    if (files.length === 0) {
      return [];
    }
    set({ uploading: true, datasetsError: null });
    const uploaded: DatasetSummary[] = [];
    try {
      for (const file of files) {
        const result = await uploadDataset(file);
        uploaded.push(result.dataset);
        set((state) => ({
          datasets: [
            result.dataset,
            ...state.datasets.filter(
              (dataset) => dataset.id !== result.dataset.id,
            ),
          ],
          datasetsLoaded: true,
        }));
      }
      return uploaded;
    } catch (cause) {
      set({ datasetsError: errorMessage(cause) });
      return uploaded;
    } finally {
      set({ uploading: false });
    }
  },

  removeDataset: async (datasetId) => {
    set({ datasetsError: null });
    try {
      await deleteDatasetApi(datasetId);
      set((state) => ({
        datasets: state.datasets.filter((dataset) => dataset.id !== datasetId),
      }));
      return true;
    } catch (cause) {
      set({ datasetsError: errorMessage(cause) });
      return false;
    }
  },

  clearDatasetError: () => set({ datasetsError: null }),
}));
