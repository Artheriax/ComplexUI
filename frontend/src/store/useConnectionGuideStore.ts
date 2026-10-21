import { create } from "zustand";

import type { SocketType } from "../types/graph";

export interface ConnectionGuide {
  mode: "connect" | "bundle";
  direction: "source" | "target";
  dataType: SocketType;
  originNodeId: string;
  originHandleId: string;
  ignoredTargetNodeId?: string;
  ignoredTargetHandleId?: string;
}

interface ConnectionGuideStore {
  guide: ConnectionGuide | null;
  begin: (guide: ConnectionGuide) => void;
  clear: () => void;
}

export const useConnectionGuideStore = create<ConnectionGuideStore>((set) => ({
  guide: null,
  begin: (guide) => set({ guide }),
  clear: () => set({ guide: null }),
}));
