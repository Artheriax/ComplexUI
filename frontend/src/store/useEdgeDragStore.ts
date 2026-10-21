import { create } from "zustand";

import type { SocketType } from "../types/graph";

export interface EdgeDragInfo {
  edgeId: string;
  sourceNodeId: string;
  sourceHandleId: string;
  targetNodeId: string;
  targetHandleId: string;
  dataType: SocketType;
  sourceScreen: { x: number; y: number };
}

export interface PendingEdgeDrag extends EdgeDragInfo {
  start: { x: number; y: number };
}

export interface EdgeDrag extends EdgeDragInfo {
  current: { x: number; y: number };
}

interface EdgeDragStore {
  pending: PendingEdgeDrag | null;
  drag: EdgeDrag | null;
  press: (pending: PendingEdgeDrag) => void;
  activate: (current: { x: number; y: number }) => void;
  move: (x: number, y: number) => void;
  clear: () => void;
}

export const useEdgeDragStore = create<EdgeDragStore>((set) => ({
  pending: null,
  drag: null,
  press: (pending) => set({ pending, drag: null }),
  activate: (current) =>
    set((state) => {
      if (!state.pending) {
        return state;
      }
      const pending = state.pending;
      return {
        pending: null,
        drag: {
          edgeId: pending.edgeId,
          sourceNodeId: pending.sourceNodeId,
          sourceHandleId: pending.sourceHandleId,
          targetNodeId: pending.targetNodeId,
          targetHandleId: pending.targetHandleId,
          dataType: pending.dataType,
          sourceScreen: pending.sourceScreen,
          current,
        },
      };
    }),
  move: (x, y) =>
    set((state) =>
      state.drag
        ? { drag: { ...state.drag, current: { x, y } } }
        : state,
    ),
  clear: () => set({ pending: null, drag: null }),
}));
