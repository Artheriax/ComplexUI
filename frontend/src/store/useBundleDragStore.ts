import { create } from "zustand";

import type { SocketType } from "../types/graph";

export interface BundleDrag {
  edgeIds: string[];
  sourceNodeId: string;
  sourceHandleId: string;
  dataType: SocketType;
  start: { x: number; y: number };
  current: { x: number; y: number };
}

interface BundleDragStore {
  drag: BundleDrag | null;
  start: (drag: BundleDrag) => void;
  move: (x: number, y: number) => void;
  clear: () => void;
}

export const useBundleDragStore = create<BundleDragStore>((set) => ({
  drag: null,
  start: (drag) => set({ drag }),
  move: (x, y) =>
    set((state) =>
      state.drag
        ? { drag: { ...state.drag, current: { x, y } } }
        : state,
    ),
  clear: () => set({ drag: null }),
}));
